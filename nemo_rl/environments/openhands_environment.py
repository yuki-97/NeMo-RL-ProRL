import asyncio
import logging
import os
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import aiohttp
import torch
from transformers import PreTrainedTokenizerBase

from nemo_rl.algorithms.grpo import MasterConfig
from nemo_rl.data.llm_message_utils import _pad_tensor
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import _get_node_ip_local

logger = logging.getLogger(__name__)


class OpenhandsEnvironment:
    """OpenhandsEnvironment manages a distributed group of async LLM server instances.

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
       - Converting BatchedDataDict inputs to message format
       - Distributing requests across available servers
       - Aggregating results and converting back to BatchedDataDict

    4. Tokenization and Formatting:
       - Applying chat templates for different model types
       - Handling prompt/response tokenization and padding
       - Managing sequence length constraints

    The manager supports both tensor parallel and data parallel deployments,
    automatically detecting the optimal server configuration based on the
    worker group topology.
    """

    def __init__(
        self,
        config: MasterConfig,
        tokenizer: PreTrainedTokenizerBase,
        server_addresses: List[str],
        dp_size: int,
    ):
        """Initialize OpenhandsEnvironment with configuration and worker group.

        Args:
            config (MasterConfig): Complete configuration object
            tokenizer (PreTrainedTokenizerBase): Tokenizer that contains custom chat template
            server_addresses (List[str]): List of server addresses
            dp_size (int): Data parallel size

        The initialization process:
        1. Extracts relevant configuration parameters
        2. Calculates tensor/data parallel topology
        3. Starts LLM server instances on appropriate nodes
        4. Initializes tokenizer and chat templates
        5. Configures OpenHands server integration
        6. Sets up sampling parameters for generation
        """
        # Store configuration
        self.full_config = config
        self.config = config["policy"]["generation"]
        self.server_addresses = server_addresses
        self.dp_size = dp_size

        # current implementation only supports token level generation
        token_level_generation = self.config.get("token_level_generation", True)
        assert token_level_generation, "Only token level generation is supported now."

        # Extract generation and processing parameters
        # Number of trajectories per prompt
        self.num_trajectories = self.full_config["grpo"]["num_generations_per_prompt"]
        # Token filtering
        # Not supported and not used yet
        # self.remove_think_tokens = False

        # Use the tokenizer passed in the constructor
        self.tokenizer = tokenizer

        # Set sequence length constraints from configuration
        self.max_prompt_length = self.config["max_prompt_length"]
        self.max_response_length = self.config["max_response_length"]
        self.total_len = self.config["vllm_cfg"]["max_model_len"]
        assert self.max_prompt_length + self.max_response_length <= self.total_len
        # TODO: check
        self.max_starting_message_length = None

        # Use CPU device for tensor operations (data preparation)
        self.device = torch.device("cpu")

        # OpenHands worker configuration for parallel processing
        self.openhands_num_workers = self.config.get("openhands_num_workers", 64)

        # Extract model name from path for API identification
        model_name = "/".join(self.full_config["policy"]["model_name"].split("/")[-2:])

        # Configure sampling parameters for generation
        # These parameters control the randomness and quality of generated text
        self.max_turns = self.full_config["grpo"]["max_rollout_turns"]
        self.sampling_params = {
            "model": f"hosted_vllm/{model_name}",
            # Placeholder for local servers
            "api_key": "dummy_key",
            "modify_params": False,
            "log_completions": False,
            "native_tool_calling": self.config.get("native_tool_calling", True),
            # Randomness control
            "temperature": self.config["temperature"],
            # Nucleus sampling
            "top_p": self.config["top_p"],
            # Max tool calls
            "max_iterations": self.max_turns,
            # Response length limit
            "max_output_tokens": self.max_response_length,
            "token_level_generation": token_level_generation,
            "custom_tokenizer": self.full_config["policy"]["tokenizer"]["name"],
            "max_model_len": self.total_len,
            "ensure_thinking_end_properly": not token_level_generation,
            "strict_loop_detector": self.config.get("strict_loop_detector", True),
            "is_reasoning_task": self.config.get("is_reasoning_task", False),
        }
        # Parse and validate OpenHands server addresses
        # OpenHands servers handle multi-turn conversations with tool usage
        openhands_base_urls = os.environ.get("OPENHANDS_URLS", None)
        if openhands_base_urls is None:
            local_ip = _get_node_ip_local()
            openhands_base_urls = f"http://{local_ip}:8006"

        if isinstance(openhands_base_urls, str):
            # Support multiple URLs separated by '+'
            self.openhands_urls = [
                url.strip() for url in openhands_base_urls.split("+") if url.strip()
            ]
        else:
            self.openhands_urls = [openhands_base_urls] if openhands_base_urls else []

        # Initialize OpenHands servers by clearing existing LLM servers
        asyncio.run(self.clear_llm_servers())

        # Distribute LLM server addresses to OpenHands servers for load balancing
        self._send_llm_addresses_to_openhands()

        # Initialize chat template for message formatting
        # Different models require different conversation formats
        # It is set at nemo_rl/algorithms/utils.py:get_tokenizer
        self.chat_template = self.tokenizer.chat_template

    # ===============================================================================
    # Assign LLM addresses to OpenHands servers
    # ===============================================================================
    def assign_llm_addresses_to_openhands(self):
        """Distribute LLM server addresses to OpenHands servers for optimal load balancing.

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
                if (
                    server_addresses2ips[server_address]
                    == openhands_urls2ips[openhands_url]
                ):
                    assigned_addresses.append(server_address)
                    used_addresses.add(server_address)

            address_assignments[openhands_url] = assigned_addresses
            logger.info(
                f"Assigned same-IP LLM server addresses to {openhands_url}: {assigned_addresses}"
            )

        # Phase 2: Distribute remaining LLM servers evenly among all OpenHands servers
        remaining_addresses = [
            addr for addr in self.server_addresses if addr not in used_addresses
        ]

        if remaining_addresses:
            logger.info(
                f"Distributing {len(remaining_addresses)} remaining LLM servers among {num_openhands} OpenHands servers"
            )

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
                        address_assignments[openhands_url].append(
                            remaining_addresses[addr_index]
                        )
                        addr_index += 1

                logger.info(
                    f"Final LLM server addresses assigned to {openhands_url}: {address_assignments[openhands_url]}"
                )

        # Log assignment summary for monitoring and debugging
        total_assigned = sum(len(addrs) for addrs in address_assignments.values())
        logger.info(
            f"Assignment complete: {total_assigned}/{num_llm_servers} LLM servers assigned to {num_openhands} OpenHands servers"
        )

        return address_assignments

    def _send_llm_addresses_to_openhands(self):
        """Distribute and send LLM server addresses to multiple OpenHands servers.

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
            logger.warning(
                "No OpenHands base URLs configured, skipping LLM address distribution"
            )
            return

        # Calculate optimal address assignments based on network topology
        address_assignments = self.assign_llm_addresses_to_openhands()

        # Send addresses asynchronously to all OpenHands servers
        asyncio.run(self._send_addresses_async(address_assignments))
        logger.info("Successfully sent LLM addresses to all OpenHands servers")

    async def _send_addresses_async(self, address_assignments: Dict[str, List[str]]):
        """Asynchronously send LLM addresses to multiple OpenHands servers.

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
                wrapped_address = f"http://{address}"
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

        logger.info(
            f"Successfully sent {success_count}/{len(results)} LLM addresses to OpenHands servers"
        )

    async def _add_llm_server_to_openhands(self, openhands_base_url: str, address: str):
        """Send a single LLM server address to an OpenHands server.

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
                        logger.info(
                            f"LLM server {address} added successfully to {openhands_base_url}: {result}"
                        )
                        return result
                    else:
                        # Handle HTTP error responses
                        error_text = await response.text()
                        logger.error(
                            f"Failed to add LLM server {address} to {openhands_base_url}, HTTP {response.status}: {error_text}"
                        )
                        raise Exception(f"HTTP {response.status}: {error_text}")

            finally:
                # Ensure session is properly closed
                await session.close()

        except asyncio.TimeoutError:
            error_msg = f"Timeout adding LLM server {address} to {openhands_base_url}"
            logger.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            logger.error(
                f"Error adding LLM server {address} to {openhands_base_url}: {e}"
            )
            raise

    # ===============================================================================
    # Formatting & Rollout
    # ===============================================================================
    def BatchedDataDict2Messages(self, prompts: BatchedDataDict) -> list[dict]:
        """Convert BatchedDataDict input to OpenHands message format.

        This method transforms the structured BatchedDataDict input into the message format
        expected by OpenHands servers. It handles trajectory expansion to generate
        multiple conversation variants from each input prompt.

        Args:
            prompts (BatchedDataDict): Input data containing instance information
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
        messages = prompts["instance"]

        # Expand each instance into multiple trajectories
        new_messages = []
        for i in range(len(messages)):
            for j in range(self.num_trajectories):
                # Create a copy of the original message for each trajectory
                tmp_message = messages[i].copy()
                tmp_message["trajectory_id"] = j  # Add trajectory identifier
                new_messages.append(tmp_message)

        return new_messages

    def Results2BatchedDataDict(self, results: dict) -> BatchedDataDict:
        """Convert OpenHands conversation results to BatchedDataDict format for training.

        Args:
            results (dict): Dictionary with structure {instance_id: {trajectory_id: result_dict}}
                Each result_dict contains:
                - messages: List of conversation messages, each message contains token ids.
                - resolved: Boolean indicating if task was completed
                - success: Boolean indicating if execution succeeded
                - error: Optional error message
                - tools: Optional tool definitions

        Returns:
            BatchedDataDict: Structured data containing:
                - prompt_ids, message_log, extra_env_info, task_name, total_reward, truncated, loss_multiplier
                - Proper formatting for downstream training/evaluation
        """

        def handle_empty_messages(results: dict, instance_id: str):
            # Find a valid messages list to use as fallback
            valid_messages = None
            for result in results[instance_id].values():
                messages = result.get("messages", [])
                if messages and len(messages) > 0:
                    valid_messages = messages
                    valid_resolved = result.get("resolved", False)
                    valid_finish = result.get("finish", False)
                    valid_error = result.get("error", None)
                    break

            # If we found valid messages, use them for trajectories with empty messages
            if valid_messages:
                for idx, result in results[instance_id].items():
                    if len(result.get("messages", [])) == 0:
                        print(
                            f"Got empty messages for instance_id {instance_id}, trajectory {idx}. "
                            f"Copying messages array from a valid trajectory."
                        )
                        # Copy messages from the valid trajectory
                        results[instance_id][idx]["messages"] = valid_messages.copy()
                        results[instance_id][idx]["resolved"] = valid_resolved
                        results[instance_id][idx]["error"] = valid_error
                        results[instance_id][idx]["finish"] = valid_finish
                        # Mark as padded sample
                        results[instance_id][idx]["is_padded"] = True

        def split_prompt_and_response(
            messages: list[dict],
        ) -> tuple[list[dict], list[dict]]:
            # Separate prompt from response based on first assistant message
            # The prompt includes all messages before the first assistant response
            starting_index = 0
            for i, msg in enumerate(messages):
                if msg["role"] == "assistant":
                    starting_index = i
                    break

            if starting_index == 0:
                # If no assistant message found, treat all messages as prompts
                print(
                    f"ERROR: Found no assistant message. len(messages) == {len(messages)} "
                    f"and roles are {[msg['role'] for msg in messages]}"
                )
                starting_index = len(messages)

            # Split into prompt and response parts
            prompt = messages[:starting_index]
            response = messages[starting_index:]

            return prompt, response

        def get_prompt_ids(messages: list[dict]) -> torch.Tensor:
            prompt, _ = split_prompt_and_response(messages)
            if len(prompt) != 0:
                input_ids = torch.cat([msg["token_ids"] for msg in prompt])
            else:
                input_ids = torch.tensor([])
            return input_ids

        result_dict = defaultdict(list)

        # Create final results in the same order as the input batch
        # This ensures consistent ordering for training
        for instance in self.batch["instance"]:
            instance_id = instance["instance_id"]

            # Handle empty messages by copying from another trajectory of the same instance
            # This provides robustness against individual trajectory failures
            handle_empty_messages(results, instance_id)

            for trajectory in results[instance_id].values():
                messages = trajectory.get("messages", [])
                for message in messages:
                    # convert to tensors
                    message["token_ids"] = torch.tensor(
                        message["token_ids"], dtype=torch.int64
                    )
                    # convert to tensors and rename to generation_logprobs
                    if "logprobs" in message:
                        if message["logprobs"] is not None:
                            message["generation_logprobs"] = torch.tensor(
                                message["logprobs"], dtype=torch.float32
                            )
                        message.pop("logprobs")
                    # remove input_ids to avoid duplicate in training
                    if message["input_ids"] is not None:
                        input_length = len(message["input_ids"])
                        message["token_ids"] = message["token_ids"][input_length:]
                        message["generation_logprobs"] = message["generation_logprobs"][
                            input_length:
                        ]

                resolved = trajectory.get("resolved", False)
                prompt_ids = get_prompt_ids(messages)

                # append to result_dict
                result_dict["message_log"].append(messages)
                result_dict["prompt_ids"].append(prompt_ids)
                # prompt length
                result_dict["length"].append(len(prompt_ids))
                result_dict["extra_env_info"].append(
                    {
                        "success": trajectory.get("success", False),
                        "error": trajectory.get("error", None),
                        # "instance": instance,
                        "resolved": resolved,
                        "finish": trajectory.get("finish", False),
                        "is_padded": trajectory.get("is_padded", False),
                    }
                )
                result_dict["task_name"].append("openhands")
                # TODO
                result_dict["total_reward"].append(float(resolved))
                # result_dict["idx"].append(trajectory["idx"])
                result_dict["truncated"].append(trajectory.get("end_properly", False))
                # TODO: check
                result_dict["loss_multiplier"].append(1.0)

        # concat prompt_ids
        prompt_ids = result_dict["prompt_ids"]
        pad_value = self.tokenizer.pad_token_id
        max_len = max(len(data) for data in prompt_ids)
        padded_prompt_ids = [
            _pad_tensor(data, max_len, "right", pad_value) for data in prompt_ids
        ]
        result_dict["prompt_ids"] = torch.stack(padded_prompt_ids)

        # convert to tensors
        result_dict["length"] = torch.tensor(result_dict["length"], dtype=torch.int32)
        result_dict["total_reward"] = torch.tensor(
            result_dict["total_reward"], dtype=torch.float32
        )
        result_dict["truncated"] = torch.tensor(
            result_dict["truncated"], dtype=torch.bool
        )
        result_dict["loss_multiplier"] = torch.tensor(
            result_dict["loss_multiplier"], dtype=torch.bool
        )

        return BatchedDataDict(result_dict)

    def calculate_metrics(self, batch: BatchedDataDict) -> dict:
        """Calculate metrics for the batch.

        Args:
            batch (BatchedDataDict): The batch to calculate metrics for

        Returns:
            dict: The metrics for the batch
        """
        batch_size = len(batch["message_log"])

        turns = []
        max_turns_reached = []
        total_tokens = []
        assistant_tokens = []
        env_tokens = []
        for messages in batch["message_log"]:
            turn = 0
            total_token = 0
            assistant_token = 0
            env_token = 0
            for idx, message in enumerate(messages):
                total_token += len(message["token_ids"])
                if message["role"] == "assistant":
                    turn += 1
                    assistant_token += len(message["token_ids"])
                elif idx >= 2:
                    env_token += len(message["token_ids"])
            turns.append(turn)
            max_turns_reached.append(turn == self.max_turns)
            total_tokens.append(total_token)
            assistant_tokens.append(assistant_token)
            env_tokens.append(env_token)

        rollout_metrics = {
            # Overall metrics
            "total_turns": sum(turns),
            "avg_turns_per_sample": sum(turns) / batch_size,
            "max_turns_per_sample": max(turns),
            # TODO: check the logic of terminate and truncated
            "natural_termination_rate": batch["truncated"].float().mean().item(),
            "truncation_rate": batch["truncated"].float().mean().item(),
            "max_turns_reached_rate": sum(max_turns_reached) / batch_size,
            # Token usage metrics
            "mean_total_tokens_per_sample": sum(total_tokens) / batch_size,
            "mean_gen_tokens_per_sample": sum(assistant_tokens) / batch_size,
            "mean_env_tokens_per_sample": sum(env_tokens) / batch_size,
            # Reward metrics
            "mean_total_reward": batch["total_reward"].float().mean().item(),
            "max_total_reward": batch["total_reward"].float().max().item(),
            "min_total_reward": batch["total_reward"].float().min().item(),
        }

        return rollout_metrics

    def run_async_rollout(self, batch: BatchedDataDict) -> tuple[BatchedDataDict, dict]:
        """Generate multiple conversation sequences in parallel via OpenHands servers.

        This is the main entry point for conversation generation. It orchestrates
        the entire pipeline from input processing to result aggregation:

        1. Input Processing:
           - Converts BatchedDataDict to OpenHands message format
           - Expands single instances to multiple trajectories

        2. Parallel Generation:
           - Distributes requests across OpenHands servers
           - Manages concurrent processing with load balancing
           - Handles retries and error recovery

        3. Result Processing:
           - Aggregates responses from all servers
           - Converts back to BatchedDataDict format
           - Includes timing information and metadata

        Args:
            prompts (BatchedDataDict): Input batch containing problem instances
            **sampling_params: Additional parameters for generation (currently unused)

        Returns:
            BatchedDataDict: Generated conversation sequences with:
                - Tensor data: tokenized conversations ready for training
                - Non-tensor data: metadata, success flags, timing info
                - meta_info: Detailed timing breakdown for performance analysis

        The method includes comprehensive timing instrumentation to help
        identify performance bottlenecks in the generation pipeline.
        """
        # Start total timing for performance analysis
        total_start_time = time.time()

        # Store batch reference for result processing
        self.batch = batch

        # Time message conversion phase
        convert_start_time = time.time()
        messages = self.BatchedDataDict2Messages(batch)
        convert_end_time = time.time()

        logger.info(
            f"BatchedDataDict2Messages time: {convert_end_time - convert_start_time:.3f}s"
        )

        # Time OpenHands request processing phase
        request_start_time = time.time()
        output_messages = asyncio.run(self.request_from_openhands(messages))
        request_end_time = time.time()

        logger.info(
            f"request_from_openhands time: {request_end_time - request_start_time:.3f}s"
        )

        # Time result conversion phase
        convert_results_start_time = time.time()
        response = self.Results2BatchedDataDict(output_messages)
        rollout_metrics = self.calculate_metrics(response)
        convert_results_end_time = time.time()
        total_end_time = time.time()

        logger.info(
            f"Results2BatchedDataDict time: {convert_results_end_time - convert_results_start_time:.3f}s"
        )
        logger.info(f"Total rollout time: {total_end_time - total_start_time:.3f}s")

        return response, rollout_metrics

    # ===============================================================================
    # Helper Functions
    # ===============================================================================
    async def clear_llm_servers(self):
        """Clear all LLM servers from OpenHands servers.

        This method removes all previously registered LLM server addresses from
        OpenHands servers. It's typically called during initialization to ensure
        a clean state before registering new servers.

        The operation is performed concurrently across all OpenHands servers
        to minimize setup time and ensure consistent state.
        """
        if not self.openhands_urls:
            logger.warning(
                "No OpenHands base URLs configured, skipping clear llm servers"
            )
            return

        logger.info(
            f"Clearing llm servers from {len(self.openhands_urls)} OpenHands servers"
        )

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
                logger.error(
                    f"Failed to clear llm servers from server {self.openhands_urls[i]}: {result}"
                )
            else:
                success_count += 1

        logger.info(
            f"Successfully cleared llm servers from {success_count}/{len(self.openhands_urls)} OpenHands servers"
        )

    async def _clear_llm_servers_single_server(self, openhands_base_url: str):
        """Clear LLM servers from a single OpenHands server.

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
                        logger.info(
                            f"Cleared llm servers from OpenHands server {openhands_base_url}: {result}"
                        )
                        return result
                    else:
                        error_text = await response.text()
                        logger.error(
                            f"Failed to clear llm servers from OpenHands server {openhands_base_url}, "
                            f"HTTP {response.status}: {error_text}"
                        )
                        raise Exception(f"HTTP {response.status}: {error_text}")
            finally:
                await session.close()
        except Exception as e:
            logger.error(
                f"Error clearing llm servers from OpenHands server {openhands_base_url}: {e}"
            )
            raise

    async def start_servers(self):
        """Start all OpenHands servers concurrently.

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
                logger.error(
                    f"Failed to start server {self.openhands_urls[i]}: {result}"
                )
            else:
                success_count += 1

        logger.info(
            f"Successfully started {success_count}/{len(self.openhands_urls)} OpenHands servers"
        )

    async def stop_servers(self):
        """Stop all OpenHands servers concurrently.

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
                logger.error(
                    f"Failed to stop server {self.openhands_urls[i]}: {result}"
                )
            else:
                success_count += 1

        logger.info(
            f"Successfully stopped {success_count}/{len(self.openhands_urls)} OpenHands servers"
        )

    async def _start_single_server(self, openhands_base_url: str):
        """Start a single OpenHands server.

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
                        logger.info(
                            f"OpenHands server {openhands_base_url} started successfully: {result}"
                        )
                        return result
                    else:
                        error_text = await response.text()
                        logger.error(
                            f"Failed to start OpenHands server {openhands_base_url}, "
                            f"HTTP {response.status}: {error_text}"
                        )
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
        """Stop a single OpenHands server.

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
                        logger.info(
                            f"OpenHands server {openhands_base_url} stopped successfully: {result}"
                        )
                        return result
                    else:
                        error_text = await response.text()
                        logger.error(
                            f"Failed to stop OpenHands server {openhands_base_url}, "
                            f"HTTP {response.status}: {error_text}"
                        )
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
        """Process conversation requests using OpenHands servers with intelligent load balancing.

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
        logger.info(
            f"Starting OpenHands servers took: {server_end_time - server_start_time:.2f} seconds"
        )

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
                    await available_servers_queue.put(
                        f"{server_url}|worker_{worker_idx}"
                    )
                    max_active_tasks += 1

            logger.info(
                f"Initialized {available_servers_queue.qsize()} available server workers"
            )

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
                                server_task = asyncio.create_task(
                                    available_servers_queue.get()
                                )

                                # Wait for both to be ready
                                done, pending = await asyncio.wait(
                                    [job_task, server_task],
                                    return_when=asyncio.ALL_COMPLETED,
                                )

                                # Extract job and server information
                                message_index, message, retry_count = job_task.result()
                                server_worker_id = server_task.result()
                                server_url = server_worker_id.split("|")[0]

                                logger.debug(
                                    f"Dispatching message {message_index} to {server_worker_id}"
                                )
                                logger.debug(
                                    f"Available servers: {available_servers_queue.qsize()}"
                                )

                        # Create async task to process job on assigned server
                        asyncio.create_task(
                            process_job_on_server(
                                message_index,
                                message,
                                retry_count,
                                server_url,
                                server_worker_id,
                            )
                        )

                    except asyncio.CancelledError:
                        break
                    except Exception as e:
                        logger.error(f"Job dispatcher error: {e}")

            # Process individual job on specific server
            async def process_job_on_server(
                message_index, message, retry_count, server_url, server_worker_id
            ):
                """Process a single job on an assigned server."""
                try:
                    # Send request to OpenHands server
                    result, should_retry = await self._send_single_message_to_openhands(
                        message=message,
                        message_index=message_index,
                        openhands_base_url=server_url,
                    )

                    # Handle retry logic
                    if should_retry and retry_count < 2:  # Maximum 3 attempts
                        async with job_queue_lock:
                            await job_queue.put(
                                (message_index, message, retry_count + 1)
                            )
                        logger.info(
                            f"Retrying message {message_index}, attempt {retry_count + 1}"
                        )
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

            logger.info(
                f"Started job dispatcher with {max_active_tasks} concurrent workers "
                f"to process {len(messages)} messages"
            )

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
                instance_id = message["instance_id"]
                trajectory_id = message["trajectory_id"]

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
                        "finish": False,
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
                        "finish": False,
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

            logger.info(
                f"Request processing completed in {processing_end_time - processing_start_time:.2f}s"
            )
            logger.info(f"Total request time: {total_end_time - total_start_time:.2f}s")
            logger.info(f"Processed {len(all_responses)} instances successfully")

            return all_responses

        finally:
            # Always stop servers for cleanup
            stop_start_time = time.time()
            await self.stop_servers()
            stop_end_time = time.time()
            logger.info(f"Server shutdown took: {stop_end_time - stop_start_time:.2f}s")

    async def _send_single_message_to_openhands(
        self, message: dict, message_index: int, openhands_base_url: str
    ):
        """Send single message to openhands server.

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
                "sampling_params": self.sampling_params,
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
                        out_message = data.get("messages", [])
                        if out_message and len(out_message) > 0:
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
        """Get address and port information for all LLM servers.

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
