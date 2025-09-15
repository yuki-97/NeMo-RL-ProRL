# Standard library imports for async operations, logging, and networking
import asyncio
import logging
import os
import socket
import threading
from typing import Any, Dict, List, Tuple

# Ray framework for distributed computing
import ray
from omegaconf import DictConfig

# VERL framework imports
from verl.protocol import DataProto  # Data protocol for tensor/non-tensor data exchange
from verl.single_controller.ray.base import RayWorkerGroup  # Ray worker group management
from verl.workers.rollout.async_server import async_server_class  # Async server implementation
from async_generator import asynccontextmanager
# Logging configuration
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

# Chat template management for different model types
from .chat_template_manager import get_chat_template

# PyTorch and HTTP client imports
import torch
import aiohttp
from verl.utils.model import compute_position_id_with_mask
from .utils import convert_right_padding_to_left, pad_to_max_length_right


import fastapi
import uvicorn
from starlette.requests import Request
from abc import ABC, abstractmethod

def _get_free_port():
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


class AsyncLLMServerManager:
    """
    AsyncLLMServerManager manages a distributed group of async LLM server instances.
    
    This class provides a high-level interface for managing multiple vLLM server instances
    running in a Ray cluster. It handles:
    
    1. Server Management:
       - Starting/stopping LLM servers with proper resource allocation
       - Managing server addresses and load balancing
       - Health monitoring and restart capabilities
    
    2. OpenHands Integration:
       - Distributing LLM server addresses to OpenHands servers
       - Managing OpenHands worker pools for parallel processing
       - Handling multi-turn conversations with tool usage
    
    3. Request Processing:
       - Converting DataProto inputs to message format
       - Distributing requests across available servers
       - Aggregating results and converting back to DataProto
    
    4. Tokenization and Formatting:
       - Applying chat templates for different model types
       - Handling prompt/response tokenization and padding
       - Managing sequence length constraints
    
    The manager supports both tensor parallel and data parallel deployments,
    automatically detecting the optimal server configuration based on the
    worker group topology.
    """

    def __init__(self, config: DictConfig, worker_group: RayWorkerGroup):
        """
        Initialize AsyncLLMServerManager with configuration and worker group.

        Args:
            config (DictConfig): Complete configuration object containing:
                - actor_rollout_ref: Rollout configuration settings
                - data: Data processing configuration (max lengths, etc.)
                - model: Model configuration (path, parameters, etc.)
            worker_group (RayWorkerGroup): Ray worker group for distributed execution
                - Contains worker topology information
                - Manages worker lifecycle and communication
                
        The initialization process:
        1. Extracts relevant configuration parameters
        2. Calculates tensor/data parallel topology
        3. Starts LLM server instances on appropriate nodes
        4. Initializes tokenizer and chat templates
        5. Configures OpenHands server integration
        6. Sets up sampling parameters for generation
        """
        # Store complete configuration and extract rollout-specific config
        self.full_config = config
        self.config = config.actor_rollout_ref
        self.worker_group = worker_group

        # Calculate parallel topology from worker group
        # tensor_model_parallel_size: Number of GPUs per model instance (for large models)
        # rollout_dp_size: Number of data parallel replicas (for throughput scaling)
        self.rollout_tp_size = self.config.rollout.tensor_model_parallel_size
        self.rollout_dp_size = self.worker_group.world_size // self.rollout_tp_size

        # Initialize server management data structures
        self.async_llm_servers = [None] * self.rollout_dp_size  # Ray actor references
        self.server_addresses = [None] * self.rollout_dp_size   # HTTP server addresses

        # Start LLM server instances and initialize their engines
        self.start_llm_servers()
        # All server instances are ready, initialize AsyncLLM engines
        ray.get([server.init_engine.remote() for server in self.async_llm_servers])
        
        # Extract generation and processing parameters
        self.num_trajectories = self.config.rollout.n  # Number of trajectories per prompt
        self.remove_think_tokens = self.config.rollout.remove_think_tokens  # Token filtering
        
        # Get tokenizer from the first server instance via Ray remote call
        # This ensures we use the same tokenizer as the inference engines
        self.tokenizer = ray.get(self.async_llm_servers[0].get_tokenizer.remote())
        self.tokenizer = self.tokenizer.tokenizer  # Extract the actual tokenizer object
        
        # Set sequence length constraints from configuration
        self.max_prompt_length = self.full_config.data.max_prompt_length
        self.max_response_length = self.full_config.data.max_response_length
        self.total_len = self.max_prompt_length + self.max_response_length
        self.max_starting_message_length = self.config.rollout.max_starting_message_length
        
        # Use CPU device for tensor operations (data preparation)
        self.device = torch.device('cpu')
        
        # OpenHands worker configuration for parallel processing
        self.openhands_num_workers = self.config.rollout.openhands_num_workers

        # Extract model name from path for API identification
        model_name = '/'.join(self.config.model.path.split('/')[-2:])

        # Configure sampling parameters for generation
        # These parameters control the randomness and quality of generated text
        self.sampling_params = {
            "model": f"hosted_vllm/{model_name}",
            "api_key": "dummy_key",  # Placeholder for local servers
            "modify_params": False,
            "log_completions": False,
            "native_tool_calling": self.config.rollout.get("native_tool_calling", True),
            "temperature": self.config.rollout.temperature,  # Randomness control
            "top_p": self.config.rollout.top_p,              # Nucleus sampling
            "max_iterations": self.config.rollout.max_iterations,  # Max tool calls
            "max_output_tokens": self.max_response_length,   # Response length limit
            'token_level_generation': self.config.rollout.get("token_level_generation", False), # This is new
            'custom_tokenizer': self.config.rollout.get("custom_tokenizer", 'Qwen/Qwen3-8B'),
            'max_model_len': self.total_len,
            'ensure_thinking_end_properly': False if self.config.rollout.get("token_level_generation", False) else True,
        }
        # Parse and validate OpenHands server addresses
        # OpenHands servers handle multi-turn conversations with tool usage
        openhands_base_urls = self.config.rollout.openhands_base_url
        if isinstance(openhands_base_urls, str):
            # Support multiple URLs separated by '+' 
            self.openhands_urls = [url.strip() for url in openhands_base_urls.split('+') if url.strip()]
        else:
            self.openhands_urls = [openhands_base_urls] if openhands_base_urls else []
        
        # Initialize OpenHands servers by clearing existing LLM servers
        asyncio.run(self.clear_llm_servers())
        
        # Distribute LLM server addresses to OpenHands servers for load balancing
        self._send_llm_addresses_to_openhands()
        
        # Initialize chat template for message formatting
        # Different models require different conversation formats
        self.chat_template_name = getattr(self.config.rollout, 'chat_template_name', 'qwen3_chat_template_generation')
        self.chat_template = get_chat_template(self.chat_template_name)
        logger.info(f"Using chat template: {self.chat_template_name}")

    def start_llm_servers(self):
        """
        Start LLM server instances across the Ray cluster with proper resource allocation.
        
        This method handles the complex process of starting distributed LLM servers:
        
        1. Resource Discovery:
           - Queries the Ray register center for worker node information
           - Determines optimal server placement based on worker topology
           
        2. Server Instantiation:
           - Creates async server instances using the appropriate backend
           - Applies node affinity scheduling to colocate servers with workers
           - Handles naming and resource allocation
           
        3. Error Handling and Retry:
           - Implements retry logic for address conflicts
           - Gracefully handles server startup failures
           - Ensures all servers are successfully started before proceeding
           
        4. Address Management:
           - Collects and stores server HTTP addresses
           - Maintains mapping between data parallel ranks and addresses
           
        The method uses a retry loop to handle common startup issues like
        port conflicts, ensuring robust server initialization.
        """
        # Get worker information from Ray register center
        # This provides the mapping between worker ranks and node IDs
        register_center = ray.get_actor(f"{self.worker_group.name_prefix}_register_center")
        workers_info = ray.get(register_center.get_worker_info.remote())
        assert len(workers_info) == self.worker_group.world_size
        
        # Get the appropriate server class based on rollout backend configuration
        # This allows switching between different inference backends (vLLM, TensorRT-LLM, etc.)
        from verl.nvidia.rollout.vllm_async_server import AsyncvLLMServer

        # Start all server instances with retry logic for address conflicts
        # Track which data parallel ranks still need successful server startup
        unready_dp_ranks = set(range(self.rollout_dp_size))
        
        while len(unready_dp_ranks) > 0:
            # Create server instances for all unready ranks
            servers = {
                rollout_dp_rank: AsyncvLLMServer.options(
                    # Ensure AsyncvLLMServer colocates with its corresponding workers
                    # This optimizes memory usage and reduces network overhead
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        # Use the first worker in the tensor parallel group for this DP rank
                        node_id=workers_info[rollout_dp_rank * self.rollout_tp_size],
                        soft=False,  # Hard constraint - must run on this node
                    ),
                    # Assign unique name for Ray actor registry
                    name=f"async_llm_server_{rollout_dp_rank}",
                ).remote(
                    self.full_config,           # Complete configuration
                    self.rollout_dp_size,       # Total number of data parallel replicas
                    rollout_dp_rank,            # This server's data parallel rank
                    self.worker_group.name_prefix  # Worker group identifier
                )
                for rollout_dp_rank in unready_dp_ranks
            }

            # Attempt to start each server and collect successful addresses
            for rollout_dp_rank, server in servers.items():
                try:
                    # Get the HTTP server address from the started server
                    # This is a blocking call that waits for server initialization
                    address = ray.get(server.get_server_address.remote())
                    
                    # Store successful server reference and address
                    self.server_addresses[rollout_dp_rank] = address
                    self.async_llm_servers[rollout_dp_rank] = server
                    
                    # Remove from unready set - this server is now operational
                    unready_dp_ranks.remove(rollout_dp_rank)
                    
                except Exception:
                    # Server startup failed (likely port conflict), clean up and retry
                    ray.kill(server)
                    logger.error(f"rollout server {rollout_dp_rank} failed, maybe address already in use, restarting...")

    def assign_llm_addresses_to_openhands(self):
        """
        Distribute LLM server addresses to OpenHands servers for optimal load balancing.
        
        This method implements a two-phase assignment strategy:
        
        Phase 1 - Locality-First Assignment:
        - Assigns LLM servers to OpenHands servers running on the same IP address
        - Minimizes network latency and maximizes bandwidth utilization
        - Reduces cross-node communication overhead
        
        Phase 2 - Load Distribution:
        - Distributes remaining LLM servers evenly across all OpenHands servers
        - Ensures balanced workload distribution
        - Handles cases where server counts don't divide evenly
        
        Returns:
            dict: Mapping of OpenHands server URLs to their assigned LLM server addresses
            
        The assignment algorithm prioritizes network locality while maintaining
        load balance across the entire OpenHands server pool.
        """
        
        def url_to_ip(url):
            """Extract IP address from URL by removing protocol and port."""
            # Remove HTTP/HTTPS protocol prefixes
            url = url.replace("http://", "").replace("https://", "")
            # Remove port number if present
            url = url.split(":")[0]
            return url
        
        # Create IP address mappings for both OpenHands and LLM servers
        openhands_urls2ips = {url: url_to_ip(url) for url in self.openhands_urls}
        server_addresses2ips = {url: url_to_ip(url) for url in self.server_addresses}
        
        num_openhands = len(self.openhands_urls)
        num_llm_servers = len(self.server_addresses)
        
        # Initialize assignment tracking
        address_assignments = {}
        used_addresses = set()
        
        # Phase 1: Assign LLM servers to OpenHands servers on the same IP
        # This optimizes for network locality and reduces latency
        for i, openhands_url in enumerate(self.openhands_urls):            
            assigned_addresses = []
            
            # Find all LLM servers on the same IP as this OpenHands server
            for server_address in self.server_addresses:
                if server_addresses2ips[server_address] == openhands_urls2ips[openhands_url]:
                    assigned_addresses.append(server_address)
                    used_addresses.add(server_address)
                    
            address_assignments[openhands_url] = assigned_addresses
            logger.info(f"Assigned same-IP LLM server addresses to {openhands_url}: {assigned_addresses}")
        
        # Phase 2: Distribute remaining LLM servers evenly among all OpenHands servers
        remaining_addresses = [addr for addr in self.server_addresses if addr not in used_addresses]
        
        if remaining_addresses:
            logger.info(f"Distributing {len(remaining_addresses)} remaining LLM servers among {num_openhands} OpenHands servers")
            
            # Calculate distribution parameters
            additional_per_openhands = len(remaining_addresses) // num_openhands
            remainder = len(remaining_addresses) % num_openhands
            
            # Distribute remaining addresses with even load balancing
            addr_index = 0
            for i, openhands_url in enumerate(self.openhands_urls):
                # Calculate how many additional addresses this OpenHands server gets
                num_additional = additional_per_openhands
                if i < remainder:  # First 'remainder' servers get one extra
                    num_additional += 1
                
                # Assign additional addresses to this OpenHands server
                for j in range(num_additional):
                    if addr_index < len(remaining_addresses):
                        address_assignments[openhands_url].append(remaining_addresses[addr_index])
                        addr_index += 1
                
                logger.info(f"Final LLM server addresses assigned to {openhands_url}: {address_assignments[openhands_url]}")
        
        # Log assignment summary for monitoring and debugging
        total_assigned = sum(len(addrs) for addrs in address_assignments.values())
        logger.info(f"Assignment complete: {total_assigned}/{num_llm_servers} LLM servers assigned to {num_openhands} OpenHands servers")
        
        return address_assignments
    
    def _send_llm_addresses_to_openhands(self):
        """
        Distribute and send LLM server addresses to multiple OpenHands servers.
        
        This method orchestrates the distribution of LLM server addresses to OpenHands servers
        for load balancing and fault tolerance. It performs the following steps:
        
        1. Validates that OpenHands servers are configured
        2. Calculates optimal address assignments using locality-aware algorithm
        3. Sends addresses asynchronously to all OpenHands servers
        4. Provides detailed logging for monitoring and debugging
        
        The method ensures that each OpenHands server receives appropriate LLM server
        addresses, enabling distributed processing of conversation requests.
        """
        if not self.openhands_urls:
            logger.warning("No OpenHands base URLs configured, skipping LLM address distribution")
            return
            
        # Calculate optimal address assignments based on network topology
        address_assignments = self.assign_llm_addresses_to_openhands()
        
        # Send addresses asynchronously to all OpenHands servers
        asyncio.run(self._send_addresses_async(address_assignments))
        logger.info("Successfully sent LLM addresses to all OpenHands servers")

    async def _send_addresses_async(self, address_assignments: Dict[str, List[str]]):
        """
        Asynchronously send LLM addresses to multiple OpenHands servers.
        
        This method implements concurrent address distribution to minimize setup time:
        
        1. Creates async tasks for each LLM server address assignment
        2. Sends all requests concurrently using asyncio.gather
        3. Handles exceptions gracefully and reports success/failure statistics
        4. Provides detailed logging for monitoring
        
        Args:
            address_assignments (Dict[str, List[str]]): Mapping of OpenHands server URLs
                to their assigned LLM server addresses
                
        The concurrent approach significantly reduces setup time compared to
        sequential address distribution, especially with many servers.
        """
        tasks = []
        
        # Create async tasks for each address assignment
        for openhands_url, addresses in address_assignments.items():
            for address in addresses:
                wrapped_address = f"http://{address}" if self.config.rollout.get("token_level_generation", False) else f"http://{address}/v1"
                task = self._add_llm_server_to_openhands(openhands_url, wrapped_address)
                tasks.append(task)
        
        # Send all requests concurrently for maximum efficiency
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and provide detailed feedback
        success_count = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to send address {i}: {result}")
            else:
                success_count += 1
        
        logger.info(f"Successfully sent {success_count}/{len(results)} LLM addresses to OpenHands servers")

    async def _add_llm_server_to_openhands(self, openhands_base_url: str, address: str):
        """
        Send a single LLM server address to an OpenHands server.
        
        This method handles the HTTP communication with OpenHands servers to register
        LLM server addresses. It includes comprehensive error handling and timeout management.
        
        Args:
            openhands_base_url (str): Base URL of the OpenHands server
            address (str): HTTP address of the LLM server to register
            
        Returns:
            dict: Response from the OpenHands server on successful registration
            
        Raises:
            Exception: On HTTP errors, timeouts, or communication failures
            
        The method uses proper session management and timeout handling to ensure
        reliable communication with OpenHands servers.
        """
        try:
            # Configure reasonable timeout for server communication
            timeout = aiohttp.ClientTimeout(total=30)
            session = aiohttp.ClientSession(timeout=timeout)
            
            try:
                # Prepare request URL and payload
                url = f"{openhands_base_url}/add_llm_server"
                payload = {"address": address}
                
                # Send POST request to register LLM server
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"LLM server {address} added successfully to {openhands_base_url}: {result}")
                        return result
                    else:
                        # Handle HTTP error responses
                        error_text = await response.text()
                        logger.error(f"Failed to add LLM server {address} to {openhands_base_url}, HTTP {response.status}: {error_text}")
                        raise Exception(f"HTTP {response.status}: {error_text}")
                        
            finally:
                # Ensure session is properly closed
                await session.close()
                
        except asyncio.TimeoutError:
            error_msg = f"Timeout adding LLM server {address} to {openhands_base_url}"
            logger.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            logger.error(f"Error adding LLM server {address} to {openhands_base_url}: {e}")
            raise

    def wake_up(self):
        """
        Wake up all vLLM instances from sleep mode.
        
        This method activates all LLM servers that may have been put into sleep mode
        for resource conservation. It's useful for resuming operations after
        periods of inactivity.
        
        The wake_up operation is performed synchronously across all servers to ensure
        they are ready before returning control to the caller.
        """
        ray.get([server.wake_up.remote() for server in self.async_llm_servers])

    def sleep(self):
        """
        Put all vLLM instances into sleep mode.
        
        This method puts all LLM servers into a low-power state to conserve resources
        when they are not actively processing requests. This is useful for:
        - Reducing GPU memory usage during idle periods
        - Conserving power in multi-tenant environments
        - Allowing other processes to use GPU resources temporarily
        
        The sleep operation is performed synchronously across all servers.
        """
        ray.get([server.sleep.remote() for server in self.async_llm_servers])

    def _convert_results_to_dataproto(self, results) -> DataProto:
        """
        Convert OpenHands conversation results to DataProto format for training.
        
        This method performs complex data transformation to convert conversation results
        from OpenHands servers into the structured DataProto format required by the
        training pipeline. The conversion process includes:
        
        1. Result Organization:
           - Groups results by instance ID and trajectory ID
           - Handles missing or empty results with intelligent fallback
           - Maintains consistent ordering matching the input batch
        
        2. Message Processing:
           - Extracts conversation messages from results
           - Separates prompts from responses based on assistant messages
           - Handles multi-turn conversations with tool usage
        
        3. Tokenization:
           - Applies chat templates for proper conversation formatting
           - Tokenizes prompts and responses with appropriate masks
           - Handles different tokenization requirements for prompts vs responses
        
        4. Tensor Creation:
           - Pads sequences to consistent lengths
           - Converts to PyTorch tensors with proper attention masks
           - Computes position IDs for transformer models
        
        5. Data Packaging:
           - Combines tensor and non-tensor data into DataProto
           - Includes metadata like success flags, error messages, etc.
           - Maintains batch structure for distributed training
        
        Args:
            results (dict): Dictionary with structure {instance_id: {trajectory_id: result_dict}}
                Each result_dict contains:
                - messages: List of conversation messages
                - resolved: Boolean indicating if task was completed
                - success: Boolean indicating if execution succeeded
                - error: Optional error message
                - tools: Optional tool definitions
                
        Returns:
            DataProto: Structured data containing:
                - Tensor data: input_ids, attention_mask, position_ids, etc.
                - Non-tensor data: success flags, error messages, metadata
                - Proper formatting for downstream training/evaluation
        """
        # Initialize lists for non-tensor data collection
        success_list = []
        error_list = []
        resolved_list = []
        has_finish_action_list = []
        
        # Create a mapping of instance_id -> list of trajectories for organization
        instance_trajectories = {}
        for instance_id, trajectories in results.items():
            instance_trajectories[instance_id] = []
            for trajectory_id, result in trajectories.items():
                instance_trajectories[instance_id].append(result)

        # Create final results in the same order as the input batch
        # This ensures consistent ordering for training
        matched_results = []
        instance_list = []
        for batch_item in self.batch:
            instance_id = batch_item.non_tensor_batch['instance']['instance_id']
            instance = batch_item.non_tensor_batch['instance']
            if instance_id in instance_trajectories:
                # Add all trajectories for this instance
                traj_results = instance_trajectories[instance_id]
                matched_results.extend(traj_results)
                instance_list.extend([instance] * len(traj_results))
        
        # Validate that we have the expected number of results
        expected_count = self.num_trajectories * len(self.batch)
        assert len(matched_results) == expected_count, f"Expected {expected_count} results, got {len(matched_results)}"
        
        # Group results by instance_id for intelligent message handling
        results_by_instance = {}
        for i, result in enumerate(matched_results):
            instance_id = instance_list[i]['instance_id']
            if instance_id not in results_by_instance:
                results_by_instance[instance_id] = []
            results_by_instance[instance_id].append((i, result))
        
        # Handle empty messages by copying from another trajectory of the same instance
        # This provides robustness against individual trajectory failures
        for instance_id, results in results_by_instance.items():
            # Find a valid messages list to use as fallback
            valid_messages = None
            for _, result in results:
                messages = result.get('messages', [])
                if messages and len(messages) > 0:
                    valid_messages = messages
                    valid_resolved = result.get('resolved', False)
                    valid_finish = result.get('finish', False)
                    valid_error = result.get('error', None)
                    break
            
            # If we found valid messages, use them for trajectories with empty messages
            if valid_messages:
                for idx, result in results:
                    if not result.get('messages') or len(result.get('messages', [])) == 0:
                        print(f"Got empty messages for instance_id {instance_id}, trajectory {idx}. "
                              f"Copying messages array from a valid trajectory.")
                        # Copy messages from the valid trajectory
                        matched_results[idx]['messages'] = valid_messages.copy()
                        matched_results[idx]['resolved'] = valid_resolved
                        matched_results[idx]['error'] = valid_error
                        matched_results[idx]['finish'] = valid_finish
                        matched_results[idx]['is_padded'] = True  # Mark as padded sample
        
        # Initialize data structures for tokenization and tensor creation
        all_messages = []
        all_prompts = []
        all_responses = []
        prompt_encodings = {'input_ids': [], 'attention_mask': []}
        response_encodings = {'input_ids': [], 'attention_mask': [], 'assistant_masks': []}
        is_padded_list = []  # Track which samples are padded for training logic

        # Process each result to extract and tokenize messages
        for result in matched_results:
            messages = result.get('messages', [])
            all_messages.append(messages)
            
            # Check if this is a padded sample (copied from another trajectory)
            is_padded = result.get('is_padded', False)
            is_padded_list.append(is_padded)
            
            # Separate prompt from response based on first assistant message
            # The prompt includes all messages before the first assistant response
            starting_index = 0
            for i, msg in enumerate(messages):
                if msg["role"] == 'assistant':
                    starting_index = i
                    break
                    
            if starting_index == 0:
                # If no assistant message found, treat all messages as prompts
                print(f'ERROR: Found no assistant message. len(messages) == {len(messages)} '
                      f'and roles are {[msg["role"] for msg in messages]}')
                starting_index = len(messages)
                
            # Split into prompt and response parts
            prompt = messages[:starting_index]
            response = messages[starting_index:]
            all_prompts.append(prompt)
            all_responses.append(response)

            # Collect non-tensor metadata
            success_list.append(result.get('success', False))
            error_list.append(result.get('error', None))
            resolved_list.append(result.get('resolved', False))
            has_finish_action_list.append(result.get('finish', False))
            
            # Get tool definitions for this conversation
            tools = result.get('tools', None)
            
            # Tokenize the prompt using chat template
            prompt_encoding = self.tokenizer.apply_chat_template(
                prompt, 
                add_generation_prompt=False,
                return_dict=True,
                tools=tools,
                chat_template=self.chat_template
            )
            
            # Tokenize the full conversation with assistant mask
            message_encoding = self.tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=False,
                return_assistant_tokens_mask=True,  # Mark assistant tokens for loss calculation
                return_dict=True,
                tools=tools,
                chat_template=self.chat_template
            )
            
            # Extract response encoding by removing prompt portion
            response_encoding = dict()
            prompt_length = len(prompt_encoding['input_ids'])
            response_encoding['input_ids'] = message_encoding['input_ids'][prompt_length:]
            response_encoding['attention_mask'] = message_encoding['attention_mask'][prompt_length:]
            response_encoding['assistant_masks'] = message_encoding['assistant_masks'][prompt_length:]
            
            # Store encodings for batch processing
            response_encodings['input_ids'].append(response_encoding['input_ids'])
            response_encodings['attention_mask'].append(response_encoding['attention_mask'])
            response_encodings['assistant_masks'].append(response_encoding['assistant_masks'])
            prompt_encodings['input_ids'].append(prompt_encoding['input_ids'])
            prompt_encodings['attention_mask'].append(prompt_encoding['attention_mask'])
            
        # Pad sequences to consistent lengths for batch processing
        
        # Pad prompts to maximum prompt length in batch
        max_len_prompt = max(len(input_ids) for input_ids in prompt_encodings['input_ids'])
        for idx in range(len(prompt_encodings['input_ids'])):
            pad_length = max_len_prompt - len(prompt_encodings['input_ids'][idx])
            prompt_encodings['input_ids'][idx].extend([self.tokenizer.pad_token_id] * pad_length)
            prompt_encodings['attention_mask'][idx].extend([0] * pad_length)
            
        # Pad responses to maximum response length in batch
        max_len_response = max(len(input_ids) for input_ids in response_encodings['input_ids'])
        for idx in range(len(response_encodings['input_ids'])):
            pad_length = max_len_response - len(response_encodings['input_ids'][idx])
            response_encodings['input_ids'][idx].extend([self.tokenizer.pad_token_id] * pad_length)
            response_encodings['attention_mask'][idx].extend([0] * pad_length)
            response_encodings['assistant_masks'][idx].extend([0] * pad_length)
        
        # Convert to PyTorch tensors and apply proper padding
        prompt_input_ids = torch.tensor(prompt_encodings['input_ids'], device=self.device)
        prompt_attention_mask = torch.tensor(prompt_encodings['attention_mask'], device=self.device)
        
        # Convert right padding to left padding for prompts (required by some models)
        prompt_input_ids, prompt_attention_mask = convert_right_padding_to_left(
            self.tokenizer, prompt_input_ids, prompt_attention_mask, 
            self.device, self.max_starting_message_length
        )
        
        # Pad responses to total sequence length
        response_ids, response_attention_mask, response_assistant_mask = pad_to_max_length_right(
            self.tokenizer, response_encodings, self.total_len, self.device
        )
        
        # Combine prompts and responses into full sequences
        input_ids = torch.cat([prompt_input_ids, response_ids], dim=1)
        attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=1)
        position_ids = compute_position_id_with_mask(attention_mask)

        # Log tensor shapes for debugging and validation
        logger.info(f"input_ids shape: {input_ids.shape}, response_ids shape: {response_ids.shape}, "
                   f"max_starting_message_length: {self.max_starting_message_length}, "
                   f"max_response_length: {self.total_len}")
        
        # Validate tensor shape consistency
        assert input_ids.shape[1] == attention_mask.shape[1] == position_ids.shape[1], \
            f"Shape mismatch: input_ids {input_ids.shape}, attention_mask {attention_mask.shape}, " \
            f"position_ids {position_ids.shape}"
        assert response_ids.shape[1] == response_assistant_mask.shape[1], \
            f"Response shape mismatch: response_ids {response_ids.shape}, " \
            f"response_assistant_mask {response_assistant_mask.shape}"
        
        # Create tensor dictionary for training
        tensor_dict = {
            'input_ids': input_ids,                    # Full sequence tokens
            'responses': response_ids,                 # Response tokens only
            'attention_mask': attention_mask,          # Attention mask for full sequence
            'position_ids': position_ids,              # Position IDs for transformer
            'loss_mask': response_assistant_mask,      # Mask for loss calculation
        }

        # Create non-tensor dictionary for metadata
        non_tensor_dict = {
            'success': success_list,                   # Task completion status
            'error': error_list,                       # Error messages if any
            'instance': instance_list,                 # Original instance data
            'resolved': resolved_list,                 # Problem resolution status
            'finish': has_finish_action_list,          # Finish action detection
            'is_padded': is_padded_list,              # Padding indicators for training
        }
        
        # Create and return DataProto with combined tensor and non-tensor data
        result_dataproto = DataProto.from_dict(
            tensors=tensor_dict,
            non_tensors=non_tensor_dict
        )
        
        return result_dataproto


    def _convert_results_to_dataproto_token(self, results) -> DataProto:
        """
        Convert OpenHands conversation results to DataProto format for training.
        
        This method is another version of _convert_results_to_dataproto, where the messages contains token ids.
        Args:
            results (dict): Dictionary with structure {instance_id: {trajectory_id: result_dict}}
                Each result_dict contains:
                - messages: List of conversation messages, each message contains token ids.
                - resolved: Boolean indicating if task was completed
                - success: Boolean indicating if execution succeeded
                - error: Optional error message
                - tools: Optional tool definitions
                
        Returns:
            DataProto: Structured data containing:
                - Tensor data: input_ids, attention_mask, position_ids, etc.
                - Non-tensor data: success flags, error messages, metadata
                - Proper formatting for downstream training/evaluation
        """
        # Initialize lists for non-tensor data collection
        success_list = []
        error_list = []
        resolved_list = []
        has_finish_action_list = []
        
        # Create a mapping of instance_id -> list of trajectories for organization
        instance_trajectories = {}
        for instance_id, trajectories in results.items():
            instance_trajectories[instance_id] = []
            for trajectory_id, result in trajectories.items():
                instance_trajectories[instance_id].append(result)

        # Create final results in the same order as the input batch
        # This ensures consistent ordering for training
        matched_results = []
        instance_list = []
        for batch_item in self.batch:
            instance_id = batch_item.non_tensor_batch['instance']['instance_id']
            instance = batch_item.non_tensor_batch['instance']
            if instance_id in instance_trajectories:
                # Add all trajectories for this instance
                traj_results = instance_trajectories[instance_id]
                matched_results.extend(traj_results)
                instance_list.extend([instance] * len(traj_results))
        
        # Validate that we have the expected number of results
        expected_count = self.num_trajectories * len(self.batch)
        assert len(matched_results) == expected_count, f"Expected {expected_count} results, got {len(matched_results)}"
        
        # Group results by instance_id for intelligent message handling
        results_by_instance = {}
        for i, result in enumerate(matched_results):
            instance_id = instance_list[i]['instance_id']
            if instance_id not in results_by_instance:
                results_by_instance[instance_id] = []
            results_by_instance[instance_id].append((i, result))
        
        # Handle empty messages by copying from another trajectory of the same instance
        # This provides robustness against individual trajectory failures
        for instance_id, results in results_by_instance.items():
            # Find a valid messages list to use as fallback
            valid_messages = None
            for _, result in results:
                messages = result.get('messages', [])
                if messages and len(messages) > 0:
                    valid_messages = messages
                    valid_resolved = result.get('resolved', False)
                    valid_finish = result.get('finish', False)
                    valid_error = result.get('error', None)
                    break
            
            # If we found valid messages, use them for trajectories with empty messages
            if valid_messages:
                for idx, result in results:
                    if not result.get('messages') or len(result.get('messages', [])) == 0:
                        print(f"Got empty messages for instance_id {instance_id}, trajectory {idx}. "
                              f"Copying messages array from a valid trajectory.")
                        # Copy messages from the valid trajectory
                        matched_results[idx]['messages'] = valid_messages.copy()
                        matched_results[idx]['resolved'] = valid_resolved
                        matched_results[idx]['error'] = valid_error
                        matched_results[idx]['finish'] = valid_finish
                        matched_results[idx]['is_padded'] = True  # Mark as padded sample
        
        # Initialize data structures for tokenization and tensor creation
        prompt_encodings = {'input_ids': [], 'attention_mask': [], 'log_probs': []}
        response_encodings = {'input_ids': [], 'attention_mask': [], 'assistant_masks': [], 'log_probs': []}
        is_padded_list = []  # Track which samples are padded for training logic

        # Process each result to extract and tokenize messages
        for result in matched_results:
            messages = result.get('messages', [])
            
            # Check if this is a padded sample (copied from another trajectory)
            is_padded = result.get('is_padded', False)
            is_padded_list.append(is_padded)
            
            # Separate prompt from response based on first assistant message
            # The prompt includes all messages before the first assistant response
            starting_index = 0
            for i, msg in enumerate(messages):
                if msg["role"] == 'assistant':
                    starting_index = i
                    break
                    
            if starting_index == 0:
                # If no assistant message found, treat all messages as prompts
                print(f'ERROR: Found no assistant message. len(messages) == {len(messages)} '
                      f'and roles are {[msg["role"] for msg in messages]}')
                starting_index = len(messages)
            
            assert messages[-1]["role"] == "assistant", f"The last message should be an assistant message. " \
                f"len(messages) == {len(messages)} and roles are {[msg['role'] for msg in messages]}"
            
            all_ids = [msg['token_ids'] for msg in messages]
            all_log_probs = [msg['logprobs'] if 'logprobs' in msg and msg['logprobs'] is not None else [0.0] * len(msg['token_ids']) for msg in messages]
            # find assistant turn indices
            assistant_turn_indices = []
            for i, msg in enumerate(messages):
                if msg["role"] == 'assistant':
                    assistant_turn_indices.append(i)
            assistant_masks = [[0] * len(input_ids) for input_ids in all_ids]
            for i in assistant_turn_indices:
                assistant_masks[i] = [1] * len(all_ids[i])
            
            # Split into prompt and response parts
            prompt_ids = all_ids[:starting_index]
            prompt_ids = sum(prompt_ids, [])
            response_ids = all_ids[starting_index:]
            response_ids = sum(response_ids, [])
            assistant_masks = sum(assistant_masks, [])
            len_prompt_ids = len(prompt_ids)
            assistant_masks = assistant_masks[len_prompt_ids:]
            attention_mask = [[1] * len(input_ids) for input_ids in all_ids]
            attention_mask = sum(attention_mask, [])
            prompt_attention_mask = attention_mask[:len_prompt_ids]
            response_attention_mask = attention_mask[len_prompt_ids:]

            response_log_probs = all_log_probs[starting_index:]
            response_log_probs = sum(response_log_probs, [])

            # Collect non-tensor metadata
            success_list.append(result.get('success', False))
            error_list.append(result.get('error', None))
            resolved_list.append(result.get('resolved', False))
            has_finish_action_list.append(result.get('finish', False))
            
            # Store encodings for batch processing
            response_encodings['input_ids'].append(response_ids)
            response_encodings['attention_mask'].append(response_attention_mask)
            response_encodings['assistant_masks'].append(assistant_masks)
            response_encodings['log_probs'].append(response_log_probs)
            prompt_encodings['input_ids'].append(prompt_ids)
            prompt_encodings['attention_mask'].append(prompt_attention_mask)
            
        # Pad sequences to consistent lengths for batch processing
        
        # Pad prompts to maximum prompt length in batch
        max_len_prompt = max(len(input_ids) for input_ids in prompt_encodings['input_ids'])
        for idx in range(len(prompt_encodings['input_ids'])):
            pad_length = max_len_prompt - len(prompt_encodings['input_ids'][idx])
            prompt_encodings['input_ids'][idx].extend([self.tokenizer.pad_token_id] * pad_length)
            prompt_encodings['attention_mask'][idx].extend([0] * pad_length)
            
        # Pad responses to maximum response length in batch
        max_len_response = max(len(input_ids) for input_ids in response_encodings['input_ids'])
        for idx in range(len(response_encodings['input_ids'])):
            pad_length = max_len_response - len(response_encodings['input_ids'][idx])
            response_encodings['input_ids'][idx].extend([self.tokenizer.pad_token_id] * pad_length)
            response_encodings['attention_mask'][idx].extend([0] * pad_length)
            response_encodings['assistant_masks'][idx].extend([0] * pad_length)
            response_encodings['log_probs'][idx].extend([0.0] * pad_length)

        # Convert to PyTorch tensors and apply proper padding
        prompt_input_ids = torch.tensor(prompt_encodings['input_ids'], device=self.device)
        prompt_attention_mask = torch.tensor(prompt_encodings['attention_mask'], device=self.device)
        # Convert right padding to left padding for prompts (required by some models)
        prompt_input_ids, prompt_attention_mask = convert_right_padding_to_left(
            self.tokenizer, prompt_input_ids, prompt_attention_mask,
            self.device, self.max_starting_message_length
        )

        # Pad responses to total sequence length
        response_ids, response_attention_mask, response_assistant_mask, response_log_probs = pad_to_max_length_right(
            self.tokenizer, response_encodings, self.total_len, self.device
        )
        
        # Combine prompts and responses into full sequences
        input_ids = torch.cat([prompt_input_ids, response_ids], dim=1)
        attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=1)
        position_ids = compute_position_id_with_mask(attention_mask)

        # Log tensor shapes for debugging and validation
        logger.info(f"input_ids shape: {input_ids.shape}, response_ids shape: {response_ids.shape}, "
                   f"max_starting_message_length: {self.max_starting_message_length}, "
                   f"max_response_length: {self.total_len}")
        
        # Validate tensor shape consistency
        assert input_ids.shape[1] == attention_mask.shape[1] == position_ids.shape[1], \
            f"Shape mismatch: input_ids {input_ids.shape}, attention_mask {attention_mask.shape}, " \
            f"position_ids {position_ids.shape}"
        assert response_ids.shape[1] == response_assistant_mask.shape[1] == response_log_probs.shape[1], \
            f"Response shape mismatch: response_ids {response_ids.shape}, " \
            f"response_assistant_mask {response_assistant_mask.shape}, " \
            f"response_log_probs {response_log_probs.shape}"
        
        # Create tensor dictionary for training
        tensor_dict = {
            'input_ids': input_ids,                    # Full sequence tokens
            'responses': response_ids,                 # Response tokens only
            'attention_mask': attention_mask,          # Attention mask for full sequence
            'position_ids': position_ids,              # Position IDs for transformer
            'loss_mask': response_assistant_mask,      # Mask for loss calculation
            'rollout_log_probs': response_log_probs,   # Log probabilities for training
        }

        # Create non-tensor dictionary for metadata
        non_tensor_dict = {
            'success': success_list,                   # Task completion status
            'error': error_list,                       # Error messages if any
            'instance': instance_list,                 # Original instance data
            'resolved': resolved_list,                 # Problem resolution status
            'finish': has_finish_action_list,          # Finish action detection
            'is_padded': is_padded_list,              # Padding indicators for training
        }
        
        # Create and return DataProto with combined tensor and non-tensor data
        result_dataproto = DataProto.from_dict(
            tensors=tensor_dict,
            non_tensors=non_tensor_dict
        )
        
        return result_dataproto


    def DataProto2Messages(self, prompts: DataProto):
        """
        Convert DataProto input to OpenHands message format.
        
        This method transforms the structured DataProto input into the message format
        expected by OpenHands servers. It handles trajectory expansion to generate
        multiple conversation variants from each input prompt.
        
        Args:
            prompts (DataProto): Input data containing instance information
                Each instance typically contains:
                - instance_id: Unique identifier for the problem instance
                - Problem description, context, and other metadata
                
        Returns:
            list: List of message dictionaries formatted for OpenHands processing
                Each message contains:
                - Original instance data
                - trajectory_id: Unique identifier for this conversation variant
                - All necessary context for OpenHands processing
                
        The method creates multiple trajectories per instance to enable diverse
        solution exploration and improved training data generation.
        """
        # Extract instance data from each prompt in the batch
        messages = [prompt.non_tensor_batch['instance'] for prompt in prompts] 
        
        # Expand each instance into multiple trajectories
        new_messages = []
        for i in range(len(messages)):
            for j in range(self.num_trajectories):
                # Create a copy of the original message for each trajectory
                tmp_message = messages[i].copy()
                tmp_message['trajectory_id'] = j  # Add trajectory identifier
                new_messages.append(tmp_message)

        return new_messages

    def generate_sequences(self, prompts: DataProto, **sampling_params) -> DataProto:
        """
        Generate multiple conversation sequences in parallel via OpenHands servers.
        
        This is the main entry point for conversation generation. It orchestrates
        the entire pipeline from input processing to result aggregation:
        
        1. Input Processing:
           - Converts DataProto to OpenHands message format
           - Expands single instances to multiple trajectories
           
        2. Parallel Generation:
           - Distributes requests across OpenHands servers
           - Manages concurrent processing with load balancing
           - Handles retries and error recovery
           
        3. Result Processing:
           - Aggregates responses from all servers
           - Converts back to DataProto format
           - Includes timing information and metadata
        
        Args:
            prompts (DataProto): Input batch containing problem instances
            **sampling_params: Additional parameters for generation (currently unused)
                
        Returns:
            DataProto: Generated conversation sequences with:
                - Tensor data: tokenized conversations ready for training
                - Non-tensor data: metadata, success flags, timing info
                - meta_info: Detailed timing breakdown for performance analysis
                
        The method includes comprehensive timing instrumentation to help
        identify performance bottlenecks in the generation pipeline.
        """
        import time
        
        # Start total timing for performance analysis
        total_start_time = time.time()
        
        # Store batch reference for result processing
        self.batch = prompts
        
        # Time message conversion phase
        convert_start_time = time.time()
        messages = self.DataProto2Messages(prompts)
        convert_end_time = time.time()
        
        # Time OpenHands request processing phase
        request_start_time = time.time()
        output_messages = asyncio.run(self.request_from_openhands(messages))
        request_end_time = time.time()
        
        # Time result conversion phase
        convert_results_start_time = time.time()
        if self.config.rollout.get("token_level_generation", False):
            response = self._convert_results_to_dataproto_token(output_messages)
        else:
            response = self._convert_results_to_dataproto(output_messages)
        convert_results_end_time = time.time()
        total_end_time = time.time()
        
        # Log detailed timing information for performance monitoring
        logger.info(f"convert_results_to_dataproto time: {convert_results_end_time - convert_results_start_time:.3f}s")

        # Ensure meta_info exists for timing data
        if not hasattr(response, 'meta_info') or response.meta_info is None:
            response.meta_info = {}
        
        # Add comprehensive timing breakdown
        response.meta_info["timing"] = {
            "total": total_end_time - total_start_time,
            "convert_messages": convert_end_time - convert_start_time,
            "openhands_request": request_end_time - request_start_time,
            "convert_results_to_dataproto": convert_results_end_time - convert_results_start_time,
            "generate_sequences": total_end_time - total_start_time  # Main timing metric
        }

        return response

    
    async def clear_llm_servers(self):
        """
        Clear all LLM servers from OpenHands servers.
        
        This method removes all previously registered LLM server addresses from
        OpenHands servers. It's typically called during initialization to ensure
        a clean state before registering new servers.
        
        The operation is performed concurrently across all OpenHands servers
        to minimize setup time and ensure consistent state.
        """
        if not self.openhands_urls:
            logger.warning("No OpenHands base URLs configured, skipping clear llm servers")
            return
        
        logger.info(f"Clearing llm servers from {len(self.openhands_urls)} OpenHands servers")
        
        # Clear all llm servers concurrently for efficiency
        tasks = []
        for openhands_url in self.openhands_urls:
            task = self._clear_llm_servers_single_server(openhands_url)
            tasks.append(task)
        
        # Wait for all clear operations to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and provide feedback
        success_count = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to clear llm servers from server {self.openhands_urls[i]}: {result}")
            else:
                success_count += 1
        
        logger.info(f"Successfully cleared llm servers from {success_count}/{len(self.openhands_urls)} OpenHands servers")    
    
    async def _clear_llm_servers_single_server(self, openhands_base_url: str):
        """
        Clear LLM servers from a single OpenHands server.
        
        Args:
            openhands_base_url (str): Base URL of the OpenHands server
            
        Returns:
            dict: Response from the OpenHands server
            
        Raises:
            Exception: On HTTP errors or communication failures
        """
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            session = aiohttp.ClientSession(timeout=timeout)
            
            try:
                url = f"{openhands_base_url}/clear_llm_server"
                
                async with session.post(url) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"Cleared llm servers from OpenHands server {openhands_base_url}: {result}")
                        return result
                    else:
                        error_text = await response.text()
                        logger.error(f"Failed to clear llm servers from OpenHands server {openhands_base_url}, "
                                   f"HTTP {response.status}: {error_text}")
                        raise Exception(f"HTTP {response.status}: {error_text}")
            finally:
                await session.close()
        except Exception as e:
            logger.error(f"Error clearing llm servers from OpenHands server {openhands_base_url}: {e}")
            raise
    
    async def start_servers(self):
        """
        Start all OpenHands servers concurrently.
        
        This method activates all configured OpenHands servers, preparing them
        for request processing. It's called before beginning conversation generation
        to ensure all servers are ready to handle requests.
        
        The startup process is performed concurrently to minimize initialization time.
        """
        if not self.openhands_urls:
            logger.warning("No OpenHands base URLs configured, skipping server start")
            return
        
        logger.info(f"Starting {len(self.openhands_urls)} OpenHands servers")
        
        # Start all servers concurrently
        tasks = []
        for openhands_url in self.openhands_urls:
            task = self._start_single_server(openhands_url)
            tasks.append(task)
        
        # Wait for all servers to start
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and provide feedback
        success_count = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to start server {self.openhands_urls[i]}: {result}")
            else:
                success_count += 1
        
        logger.info(f"Successfully started {success_count}/{len(self.openhands_urls)} OpenHands servers")

    async def stop_servers(self):
        """
        Stop all OpenHands servers concurrently.
        
        This method gracefully shuts down all OpenHands servers after request
        processing is complete. It ensures proper cleanup and resource deallocation.
        
        The shutdown process is performed concurrently to minimize cleanup time.
        """
        if not self.openhands_urls:
            logger.warning("No OpenHands base URLs configured, skipping server stop")
            return
        
        logger.info(f"Stopping {len(self.openhands_urls)} OpenHands servers")
        
        # Stop all servers concurrently
        tasks = []
        for openhands_url in self.openhands_urls:
            task = self._stop_single_server(openhands_url)
            tasks.append(task)
        
        # Wait for all servers to stop
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and provide feedback
        success_count = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to stop server {self.openhands_urls[i]}: {result}")
            else:
                success_count += 1
        
        logger.info(f"Successfully stopped {success_count}/{len(self.openhands_urls)} OpenHands servers")

    async def _start_single_server(self, openhands_base_url: str):
        """
        Start a single OpenHands server.
        
        This method sends a start request to a specific OpenHands server and waits
        for successful activation. It includes proper error handling and timeout management.
        
        Args:
            openhands_base_url (str): Base URL of the OpenHands server to start
            
        Returns:
            dict: Response from the OpenHands server on successful startup
            
        Raises:
            Exception: On HTTP errors, timeouts, or communication failures
        """
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            session = aiohttp.ClientSession(timeout=timeout)
            
            try:
                url = f"{openhands_base_url}/start"
                
                async with session.post(url) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"OpenHands server {openhands_base_url} started successfully: {result}")
                        return result
                    else:
                        error_text = await response.text()
                        logger.error(f"Failed to start OpenHands server {openhands_base_url}, "
                                   f"HTTP {response.status}: {error_text}")
                        raise Exception(f"HTTP {response.status}: {error_text}")
                        
            finally:
                await session.close()
                
        except asyncio.TimeoutError:
            error_msg = f"Timeout starting OpenHands server {openhands_base_url}"
            logger.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            logger.error(f"Error starting OpenHands server {openhands_base_url}: {e}")
            raise

    async def _stop_single_server(self, openhands_base_url: str):
        """
        Stop a single OpenHands server.
        
        This method sends a stop request to a specific OpenHands server and waits
        for graceful shutdown. It includes proper error handling and timeout management.
        
        Args:
            openhands_base_url (str): Base URL of the OpenHands server to stop
            
        Returns:
            dict: Response from the OpenHands server on successful shutdown
            
        Raises:
            Exception: On HTTP errors, timeouts, or communication failures
        """
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            session = aiohttp.ClientSession(timeout=timeout)
            
            try:
                url = f"{openhands_base_url}/stop"
                
                async with session.post(url) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"OpenHands server {openhands_base_url} stopped successfully: {result}")
                        return result
                    else:
                        error_text = await response.text()
                        logger.error(f"Failed to stop OpenHands server {openhands_base_url}, "
                                   f"HTTP {response.status}: {error_text}")
                        raise Exception(f"HTTP {response.status}: {error_text}")
                        
            finally:
                await session.close()
                
        except asyncio.TimeoutError:
            error_msg = f"Timeout stopping OpenHands server {openhands_base_url}"
            logger.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            logger.error(f"Error stopping OpenHands server {openhands_base_url}: {e}")
            raise

    async def request_from_openhands(self, messages: List):
        """
        Process conversation requests using OpenHands servers with intelligent load balancing.
        
        This method implements a sophisticated request distribution system that:
        
        1. Server Management:
           - Starts all OpenHands servers concurrently
           - Tracks server availability in real-time
           - Gracefully shuts down servers after processing
        
        2. Load Balancing:
           - Distributes requests across available servers
           - Uses job queue for fair work distribution
           - Implements server availability tracking
        
        3. Error Handling:
           - Retries failed requests up to 3 times
           - Handles server communication failures gracefully
           - Provides detailed error reporting and logging
        
        4. Progress Monitoring:
           - Tracks processing progress in real-time
           - Reports active tasks and server utilization
           - Provides comprehensive timing information
        
        Args:
            messages (List): List of message dictionaries to process
                Each message contains instance data and trajectory information
                
        Returns:
            dict: Mapping of instance_id to trajectory results
                Structure: {instance_id: {trajectory_id: result_dict}}
                Each result_dict contains conversation outcomes and metadata
                
        The method uses asyncio queues and locks to ensure thread-safe operation
        and optimal resource utilization across the server pool.
        """
        import time
        
        # Start total timing for performance analysis
        total_start_time = time.time()
        
        # Start all OpenHands servers concurrently
        server_start_time = time.time()
        await self.start_servers()
        server_end_time = time.time()
        logger.info(f"Starting OpenHands servers took: {server_end_time - server_start_time:.2f} seconds")
        
        try:
            # Initialize result storage
            all_responses = {}
            
            if not self.openhands_urls:
                logger.error("No OpenHands base URLs configured")
                return {}
                
            # Thread-safe queue management with asyncio locks
            job_queue_lock = asyncio.Lock()
            results_queue_lock = asyncio.Lock()
            available_servers_queue_lock = asyncio.Lock()
            
            # Create job queue and populate with all messages
            job_queue = asyncio.Queue()
            for i, message in enumerate(messages):
                await job_queue.put((i, message, 0))  # (index, message, retry_count)
            # Create results queue for completed work
            results_queue = asyncio.Queue()
            
            # Create server availability queue
            available_servers_queue = asyncio.Queue()
            
            # Initialize server pool - all servers start as available
            max_active_tasks = 0
            for server_url in self.openhands_urls:
                for worker_idx in range(self.openhands_num_workers):
                    await available_servers_queue.put(f"{server_url}|worker_{worker_idx}")
                    max_active_tasks += 1
            
            logger.info(f"Initialized {available_servers_queue.qsize()} available server workers")
            
            # Job dispatcher - assigns jobs to available servers
            async def job_dispatcher():
                """Continuously dispatch jobs to available servers."""
                while True:
                    try:
                        # Check for available work and servers
                        async with job_queue_lock:
                            async with available_servers_queue_lock:
                                if job_queue.empty() or available_servers_queue.empty():
                                    await asyncio.sleep(0.1)  # Brief pause before retry
                                    continue
                                    
                                # Get next job and available server
                                job_task = asyncio.create_task(job_queue.get())
                                server_task = asyncio.create_task(available_servers_queue.get())

                                # Wait for both to be ready
                                done, pending = await asyncio.wait(
                                    [job_task, server_task], 
                                    return_when=asyncio.ALL_COMPLETED
                                )
                                
                                # Extract job and server information
                                message_index, message, retry_count = job_task.result()
                                server_worker_id = server_task.result()
                                server_url = server_worker_id.split('|')[0]
                                
                                logger.debug(f"Dispatching message {message_index} to {server_worker_id}")
                                logger.debug(f"Available servers: {available_servers_queue.qsize()}")
                        
                        # Create async task to process job on assigned server
                        asyncio.create_task(
                            process_job_on_server(message_index, message, retry_count, server_url, server_worker_id)
                        )
                        
                    except asyncio.CancelledError:
                        break
                    except Exception as e:
                        logger.error(f"Job dispatcher error: {e}")
            
            # Process individual job on specific server
            async def process_job_on_server(message_index, message, retry_count, server_url, server_worker_id):
                """Process a single job on an assigned server."""
                try:
                    # Send request to OpenHands server
                    result, should_retry = await self._send_single_message_to_openhands(
                        message=message,
                        message_index=message_index,
                        openhands_base_url=server_url
                    )
                    
                    # Handle retry logic
                    if should_retry and retry_count < 2:  # Maximum 3 attempts
                        async with job_queue_lock:
                            await job_queue.put((message_index, message, retry_count + 1))
                        logger.info(f"Retrying message {message_index}, attempt {retry_count + 1}")
                    else:
                        # Job completed (success or max retries reached)
                        async with results_queue_lock:
                            await results_queue.put((message_index, message, result))
                        async with job_queue_lock:
                            job_queue.task_done()
                    
                    # Return server to available pool
                    async with available_servers_queue_lock:
                        await available_servers_queue.put(server_worker_id)
                        logger.debug(f"Returned server to pool: {server_worker_id}")
                        
                except Exception as e:
                    logger.error(f"Error processing job {message_index}: {e}")
                    # Return server to pool even on error
                    async with available_servers_queue_lock:
                        await available_servers_queue.put(server_worker_id)

            # Start job dispatcher
            dispatcher_task = asyncio.create_task(job_dispatcher())
            
            logger.info(f"Started job dispatcher with {max_active_tasks} concurrent workers "
                       f"to process {len(messages)} messages")
            
            # Collect results as they arrive
            processing_start_time = time.time()
            completed_count = 0
            
            while completed_count < len(messages):
                async with results_queue_lock:
                    if results_queue.empty():
                        await asyncio.sleep(0.1)
                        continue
                    message_index, message, result = await results_queue.get()


                # Process and store result
                instance_id = message['instance_id']
                trajectory_id = message['trajectory_id']
                
                if instance_id not in all_responses:
                    all_responses[instance_id] = {}
                
                # Store result with proper error handling
                if isinstance(result, Exception):
                    logger.error(f"Error processing message {message_index}: {result}")
                    all_responses[instance_id][trajectory_id] = {
                        "error": str(result),
                        "success": False,
                        "messages": [],
                        "resolved": False,
                        "finish": False
                    }
                elif result is not None:
                    all_responses[instance_id][trajectory_id] = result
                else:
                    logger.warning(f"Message {message_index} returned empty response")
                    all_responses[instance_id][trajectory_id] = {
                        "error": "Empty response", 
                        "success": False,
                        "messages": [],
                        "resolved": False,
                        "finish": False
                    }
                
                completed_count += 1
                
                # Log progress periodically
                if completed_count % 10 == 0 or completed_count == len(messages):
                    current_time = time.time()
                    elapsed_time = current_time - processing_start_time
                    progress_percent = (completed_count / len(messages)) * 100
                    
                    async with job_queue_lock:
                        pending_jobs = job_queue.qsize()
                    async with available_servers_queue_lock:
                        available_servers = available_servers_queue.qsize()
                    
                    active_tasks = max_active_tasks - available_servers
                    
                    logger.info(
                        f"Progress: {completed_count}/{len(messages)} ({progress_percent:.1f}%), "
                        f"pending: {pending_jobs}, available: {available_servers}, "
                        f"active: {active_tasks}/{max_active_tasks}, elapsed: {elapsed_time:.2f}s"
                    )
            
            # Clean up dispatcher
            dispatcher_task.cancel()
            await asyncio.gather(dispatcher_task, return_exceptions=True)
            
            processing_end_time = time.time()
            total_end_time = time.time()
            
            logger.info(f"Request processing completed in {processing_end_time - processing_start_time:.2f}s")
            logger.info(f"Total request time: {total_end_time - total_start_time:.2f}s")
            logger.info(f"Processed {len(all_responses)} instances successfully")
            
            return all_responses
            
        finally:
            # Always stop servers for cleanup
            stop_start_time = time.time()
            await self.stop_servers()
            stop_end_time = time.time()
            logger.info(f"Server shutdown took: {stop_end_time - stop_start_time:.2f}s")

    async def _send_single_message_to_openhands(self, message: dict, message_index: int, openhands_base_url: str):
        """
        Send single message to openhands server.
        
        This method handles the HTTP communication with OpenHands servers to process
        a single conversation request. It includes comprehensive error handling,
        timeout management, and retry logic.
        
        Args:
            message (dict): A single message dictionary containing the conversation
                to be processed by OpenHands.
            message_index (int): The index of the message in the input batch.
            openhands_base_url (str): The base URL of the OpenHands server.
            
        Returns:
            tuple: A tuple (result, should_retry) where:
                - result (dict or Exception): The response from OpenHands or an exception.
                - should_retry (bool): True if the message should be retried, False otherwise.
        """
        try:
            request_data = {
                "instance": message,
                "sampling_params": self.sampling_params
            }
            
            timeout = aiohttp.ClientTimeout(total=None)
            session = aiohttp.ClientSession(timeout=timeout)
            
            try:
                async with session.post(
                    url=f"{openhands_base_url}/process",
                    headers={"Content-Type": "application/json"},
                    json=request_data,
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        out_message=data.get("messages",[])
                        if out_message and len(out_message)>0:
                            return data, False  # No retry needed
                        else:
                            return None, True  # Retry needed
                    else:
                        error_text = await resp.text()
                        print("Failed to process request:", error_text)
                        return None, True  # Retry needed
                        
            finally:
                await session.close()
                
        except asyncio.TimeoutError as e:
            logger.error(f"Error sending message {message_index} to OpenHands: {e}")
            return None, True  # Retry needed
        except Exception as e:
            logger.error(f"Error sending message {message_index} to OpenHands: {e}")
            return None, True  # Retry needed

    def get_server_info(self) -> List[Tuple[str, int]]:
        """
        Get address and port information for all LLM servers.
        
        This method provides access to the network addresses of all running LLM servers,
        useful for monitoring, debugging, and external integrations.
        
        Returns:
            List[Tuple[str, int]]: List of (hostname, port) tuples for each server
                Each tuple contains:
                - hostname (str): The hostname or IP address of the server
                - port (int): The port number where the server is listening
                
        The information can be used for:
        - Health monitoring and status checks
        - Direct API access for debugging
        - Load balancing configuration
        - Service discovery integration
        """
        server_info = []
        for address in self.server_addresses:
            if address:  # Ensure address is not None
                host, port = address.split(":")
                server_info.append((host, int(port)))
        return server_info
