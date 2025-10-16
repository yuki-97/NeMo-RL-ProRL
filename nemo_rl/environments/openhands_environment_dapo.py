import asyncio
import logging
import time
from typing import List

import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizerBase

from nemo_rl.algorithms.grpo import MasterConfig
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.openhands_environment import OpenhandsEnvironment

logger = logging.getLogger(__name__)


class OpenhandsEnvironmentDAPO(OpenhandsEnvironment):
    """OpenhandsEnvironmentDAPO extends OpenhandsEnvironment with DAPO filtering capabilities.

    This class provides difficulty-aware policy optimization (DAPO) support by:
    1. Filtering easy/hard instances based on solved trajectories
    2. Managing job queues with priority scheduling
    3. Streaming data from dataloader to maintain batch size
    4. Handling partial batch completion with dynamic refilling

    The DAPO approach filters out instances where all or none of the trajectories are solved,
    focusing training on instances with mixed success rates that provide the most learning signal.
    """

    def __init__(
        self,
        config: MasterConfig,
        tokenizer: PreTrainedTokenizerBase,
        server_addresses: List[str],
        dp_size: int,
        dataloader: StatefulDataLoader,
    ):
        """Initialize OpenhandsEnvironmentDAPO.

        Args:
            config (MasterConfig): Complete configuration object
            tokenizer: Tokenizer instance
            server_addresses (List[str]): List of server addresses
            dp_size (int): Data parallel size
            dataloader (StatefulDataLoader): Data loader for streaming data
        """
        super().__init__(config, tokenizer, server_addresses, dp_size)

        # DAPO-specific initialization
        self.dataloader = dataloader
        self.dataloader_iter = iter(self.dataloader)
        self.all_input_batch = None  # Accumulated input batches
        self.last_data_index = 0  # Track position in data stream
        self.job_queue = asyncio.PriorityQueue()  # Priority queue for job scheduling

    def run_async_rollout_dapo(
        self, requested_batch_size: int
    ) -> tuple[BatchedDataDict, dict]:
        """Generate conversation sequences with DAPO filtering.

        This method implements the DAPO training loop that:
        1. Continuously processes instances from the dataloader
        2. Filters easy/hard instances (all solved or all failed)
        3. Returns a batch when enough mixed-difficulty instances are collected

        Args:
            requested_batch_size (int): Number of instances to return

        Returns:
            tuple[BatchedDataDict, dict]: Generated conversation sequences and rollout metrics
        """
        # Start total timing for performance analysis
        total_start_time = time.time()

        # Time OpenHands request processing phase
        request_start_time = time.time()
        output_messages, input_batch = asyncio.run(
            self.request_from_openhands_dapo(requested_batch_size)
        )
        request_end_time = time.time()

        logger.info(
            f"request_from_openhands_dapo time: {request_end_time - request_start_time:.3f}s"
        )

        # Time result conversion phase
        convert_results_start_time = time.time()
        response = self.Results2BatchedDataDict(output_messages, input_batch)
        rollout_metrics = self.calculate_metrics(response)
        convert_results_end_time = time.time()
        total_end_time = time.time()

        logger.info(
            f"Results2BatchedDataDict time: {convert_results_end_time - convert_results_start_time:.3f}s"
        )
        logger.info(f"Total rollout time: {total_end_time - total_start_time:.3f}s")

        return response, rollout_metrics

    async def refill_job_queue(self, data_index: int) -> tuple[BatchedDataDict, int]:
        """Refill the job queue with new data from the dataloader.

        Args:
            data_index (int): Current data index for priority ordering

        Returns:
            tuple[BatchedDataDict, int]: New batch and updated data index
        """
        # Get next batch from dataloader
        try:
            next_batch_data = next(self.dataloader_iter)
        except StopIteration:
            self.dataloader_iter = iter(self.dataloader)
            next_batch_data = next(self.dataloader_iter)

        # Convert batch to messages
        batch = BatchedDataDict(next_batch_data)
        messages = self.BatchedDataDict2Messages(batch)

        # Add messages to job queue with priority based on data_index
        for message in messages:
            await self.job_queue.put((data_index, (data_index, message, 0)))
            data_index += 1

        return batch, data_index

    async def request_from_openhands_dapo(self, requested_batch_size: int):
        """Process conversation requests using OpenHands servers with DAPO filtering.

        This method implements the core DAPO logic:
        1. Start OpenHands servers
        2. Dispatch jobs to available servers
        3. Collect results and filter easy/hard instances
        4. Continue until requested_batch_size instances are collected
        5. Stop servers and return results

        Args:
            requested_batch_size (int): Number of instances to return

        Returns:
            tuple: (all_responses dict, output_batch BatchedDataDict)
        """
        # Start total timing for performance analysis
        total_start_time = time.time()

        # Start all OpenHands servers concurrently
        server_start_time = time.time()
        await self.start_servers()
        server_end_time = time.time()
        logger.info(
            f"Starting OpenHands servers took: {server_end_time - server_start_time:.2f} seconds"
        )

        self.existing_ids = set()

        try:
            # Initialize result storage
            all_responses = {}

            if not self.openhands_urls:
                logger.error("No OpenHands base URLs configured")
                return {}, BatchedDataDict({})

            # Thread-safe queue management with asyncio locks
            job_queue_lock = asyncio.Lock()
            results_queue_lock = asyncio.Lock()
            available_servers_queue_lock = asyncio.Lock()

            # Create results queue for completed work
            results_queue = asyncio.PriorityQueue()

            # Create server availability queue
            available_servers_queue = asyncio.Queue()

            # Track all active process_job_on_server tasks for cleanup
            active_job_tasks = []
            active_job_tasks_lock = asyncio.Lock()

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
                data_index = self.last_data_index

                while True:
                    try:
                        # Check for available work and servers
                        async with job_queue_lock:
                            async with available_servers_queue_lock:
                                if available_servers_queue.empty():
                                    await asyncio.sleep(0.1)
                                    continue

                                if self.job_queue.empty():
                                    # Refill job queue from dataloader
                                    (
                                        input_batch,
                                        data_index,
                                    ) = await self.refill_job_queue(data_index)
                                    if self.all_input_batch is None:
                                        self.all_input_batch = input_batch
                                    else:
                                        # Concatenate with existing batch
                                        combined_dict = {}
                                        for key in self.all_input_batch.keys():
                                            if isinstance(
                                                self.all_input_batch[key], torch.Tensor
                                            ):
                                                combined_dict[key] = torch.cat(
                                                    [
                                                        self.all_input_batch[key],
                                                        input_batch[key],
                                                    ]
                                                )
                                            elif isinstance(
                                                self.all_input_batch[key], list
                                            ):
                                                combined_dict[key] = (
                                                    self.all_input_batch[key]
                                                    + input_batch[key]
                                                )
                                        self.all_input_batch = BatchedDataDict(
                                            combined_dict
                                        )
                                    continue

                                # Get next job and available server
                                job_task = asyncio.create_task(self.job_queue.get())
                                server_task = asyncio.create_task(
                                    available_servers_queue.get()
                                )

                                # Wait for both to be ready
                                done, pending = await asyncio.wait(
                                    [job_task, server_task],
                                    return_when=asyncio.ALL_COMPLETED,
                                )

                                # Extract job and server information
                                priority, (message_index, message, retry_count) = (
                                    job_task.result()
                                )
                                server_worker_id = server_task.result()
                                server_url = server_worker_id.split("|")[0]

                                logger.debug(
                                    f"Dispatching message {message_index} to {server_worker_id}"
                                )
                                logger.debug(
                                    f"Available servers: {available_servers_queue.qsize()}"
                                )

                        # Create async task to process job on assigned server
                        task = asyncio.create_task(
                            process_job_on_server(
                                message_index,
                                message,
                                retry_count,
                                server_url,
                                server_worker_id,
                            )
                        )
                        # Track task for cleanup
                        async with active_job_tasks_lock:
                            active_job_tasks.append(task)

                    except asyncio.CancelledError:
                        break
                    except Exception as e:
                        logger.error(f"Job dispatcher error: {e}")

            # Process individual job on specific server
            async def process_job_on_server(
                message_index, message, retry_count, server_url, server_worker_id
            ):
                """Process a single job on an assigned server."""
                current_task = asyncio.current_task()
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
                            await self.job_queue.put(
                                (
                                    message_index,
                                    (message_index, message, retry_count + 1),
                                )
                            )
                        logger.info(
                            f"Retrying message {message_index}, attempt {retry_count + 1}"
                        )
                    else:
                        # Job completed (success or max retries reached)
                        async with results_queue_lock:
                            await results_queue.put(
                                (message_index, (message_index, message, result))
                            )
                        async with job_queue_lock:
                            self.job_queue.task_done()

                    # Return server to available pool
                    async with available_servers_queue_lock:
                        await available_servers_queue.put(server_worker_id)
                        logger.debug(f"Returned server to pool: {server_worker_id}")

                except asyncio.CancelledError:
                    logger.info(f"Job {message_index} processing was cancelled")
                    # Return server to pool even when cancelled
                    async with available_servers_queue_lock:
                        await available_servers_queue.put(server_worker_id)
                    raise
                except Exception as e:
                    logger.error(f"Error processing job {message_index}: {e}")
                    # Return server to pool even on error
                    async with available_servers_queue_lock:
                        await available_servers_queue.put(server_worker_id)
                finally:
                    # Remove task from tracking list
                    async with active_job_tasks_lock:
                        if current_task in active_job_tasks:
                            active_job_tasks.remove(current_task)

            # Start job dispatcher
            dispatcher_task = asyncio.create_task(job_dispatcher())

            logger.info(
                f"Started job dispatcher with {max_active_tasks} concurrent workers"
            )

            # Collect results as they arrive
            processing_start_time = time.time()
            completed_count = 0
            filtered_instance_ids = []

            while True:
                async with results_queue_lock:
                    if results_queue.empty():
                        await asyncio.sleep(0.1)
                        continue
                    (
                        priority,
                        (message_index, message, result),
                    ) = await results_queue.get()

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
                num_completed_instances = 0

                # Filter easy/hard instances
                all_responses, filtered_instance_ids_tmp = (
                    self.filter_easy_hard_instance(all_responses)
                )
                filtered_instance_ids.extend(filtered_instance_ids_tmp)

                for instance_id in all_responses:
                    if len(all_responses[instance_id]) == self.num_trajectories:
                        num_completed_instances += 1

                # Log progress periodically
                if (
                    completed_count % 4 == 0
                    or num_completed_instances == requested_batch_size
                ):
                    current_time = time.time()
                    elapsed_time = current_time - processing_start_time
                    progress_percent = (
                        num_completed_instances / requested_batch_size
                    ) * 100

                    async with job_queue_lock:
                        pending_jobs = self.job_queue.qsize()
                    async with available_servers_queue_lock:
                        available_servers = available_servers_queue.qsize()

                    active_tasks = max_active_tasks - available_servers

                    # Calculate resolved and finish ratios
                    resolved_count = 0
                    finish_count = 0
                    all_results_count = 0
                    resolved_instance_count = 0
                    for instance_id in all_responses:
                        resolved_instance_count_tmp = 0
                        for trajectory_id in all_responses[instance_id]:
                            all_results_count += 1
                            if all_responses[instance_id][trajectory_id]["resolved"]:
                                resolved_count += 1
                                resolved_instance_count_tmp = 1
                            if all_responses[instance_id][trajectory_id]["finish"]:
                                finish_count += 1
                        if resolved_instance_count_tmp == 1:
                            resolved_instance_count += 1

                    resolved_ratio = (
                        resolved_count / all_results_count if all_results_count else 0
                    )
                    finish_ratio = (
                        finish_count / all_results_count if all_results_count else 0
                    )
                    resolved_instance_ratio = (
                        resolved_instance_count / len(all_responses)
                        if all_responses
                        else 0
                    )

                    logger.info(
                        f"Progress: {num_completed_instances}/{requested_batch_size} ({progress_percent:.1f}%), "
                        f"pending: {pending_jobs}, available: {available_servers}, "
                        f"active: {active_tasks}/{max_active_tasks}, elapsed: {elapsed_time:.2f}s, "
                        f"resolved: {resolved_ratio:.2f}, finish: {finish_ratio:.2f}, "
                        f"resolved_instance: {resolved_instance_ratio:.2f}"
                    )

                if num_completed_instances >= requested_batch_size:
                    logger.info(
                        f"Completed {num_completed_instances} instances, requested batch size is {requested_batch_size}"
                    )
                    break

            # Clean up all tasks
            logger.info("Stopping all tasks...")

            # Cancel job dispatcher
            dispatcher_task.cancel()

            # Cancel all active process_job_on_server tasks
            async with active_job_tasks_lock:
                active_tasks_to_cancel = active_job_tasks.copy()

            if active_tasks_to_cancel:
                logger.info(
                    f"Cancelling {len(active_tasks_to_cancel)} active job processing tasks"
                )
                for task in active_tasks_to_cancel:
                    task.cancel()

            # Wait for all tasks to complete or be cancelled
            all_tasks = [dispatcher_task] + active_tasks_to_cancel
            if all_tasks:
                await asyncio.gather(*all_tasks, return_exceptions=True)
                logger.info("All tasks have been stopped")

            processing_end_time = time.time()
            total_end_time = time.time()

            logger.info(
                f"Request processing completed in {processing_end_time - processing_start_time:.2f}s"
            )
            logger.info(f"Total request time: {total_end_time - total_start_time:.2f}s")
            logger.info(f"Processed {len(all_responses)} instances successfully")

            # Get instance IDs before filtering
            instance_ids_before_filtering = set(
                [
                    instance["instance_id"]
                    for instance in self.all_input_batch["instance"]
                ]
            )

            # Filter uncompleted instances
            all_responses, output_batch = self.filter_uncompleted_instances(
                all_responses, filtered_instance_ids
            )

            # Validate instance ID sets
            instance_ids_in_output_batch = set(
                [instance["instance_id"] for instance in output_batch["instance"]]
            )
            instance_ids_after_filtering = set(
                [
                    instance["instance_id"]
                    for instance in self.all_input_batch["instance"]
                ]
            )
            instance_ids_in_filtered_instance_ids = set(filtered_instance_ids)

            assert instance_ids_before_filtering == (
                instance_ids_after_filtering
                | instance_ids_in_output_batch
                | instance_ids_in_filtered_instance_ids
            ), (
                f"Instance ID mismatch: {instance_ids_before_filtering} != "
                f"{instance_ids_after_filtering} + {instance_ids_in_output_batch} + "
                f"{instance_ids_in_filtered_instance_ids}"
            )

            # Put remaining train data back to job queue
            await self.push_remaining_train_data_to_job_queue()

            return all_responses, output_batch

        except Exception as e:
            logger.error(f"Error in request_from_openhands_dapo: {e}")
            raise

        finally:
            # Always stop servers for cleanup
            stop_start_time = time.time()
            await self.stop_servers()
            stop_end_time = time.time()
            logger.info(f"Server shutdown took: {stop_end_time - stop_start_time:.2f}s")

    async def push_remaining_train_data_to_job_queue(self):
        """Push the remaining train data back to the job queue."""
        self.last_data_index = 0
        messages = self.BatchedDataDict2Messages(self.all_input_batch)
        for message in messages:
            await self.job_queue.put(
                (self.last_data_index, (self.last_data_index, message, 0))
            )
            self.last_data_index += 1

    def filter_uncompleted_instances(
        self, all_responses: dict, filtered_instance_ids: list
    ) -> tuple[dict, BatchedDataDict]:
        """Filter out uncompleted instances and return completed ones.

        Args:
            all_responses (dict): Dictionary of instance responses
            filtered_instance_ids (list): List of filtered instance IDs

        Returns:
            tuple[dict, BatchedDataDict]: Filtered responses and output batch
        """
        # Create mapping from instance_id to batch index
        instance_id_to_batch_idx = {}
        for i, instance in enumerate(self.all_input_batch["instance"]):
            instance_id = instance["instance_id"]
            instance_id_to_batch_idx[instance_id] = i

        # Find completed instances
        instance_ids_to_keep = []
        batch_idx_to_keep = []
        instance_ids = list(all_responses.keys())

        for instance_id in instance_ids:
            if len(all_responses[instance_id]) != self.num_trajectories:
                all_responses.pop(instance_id)
            else:
                instance_ids_to_keep.append(instance_id)
                batch_idx_to_keep.append(instance_id_to_batch_idx[instance_id])

        # Select completed instances from batch
        output_dict = {}
        for key in self.all_input_batch.keys():
            if isinstance(self.all_input_batch[key], torch.Tensor):
                output_dict[key] = self.all_input_batch[key][batch_idx_to_keep]
            elif isinstance(self.all_input_batch[key], list):
                output_dict[key] = [
                    self.all_input_batch[key][i] for i in batch_idx_to_keep
                ]
        output_batch = BatchedDataDict(output_dict)

        # Remain instances that are not filtered and not completed
        batch_idx_to_remain = []
        for i, instance in enumerate(self.all_input_batch["instance"]):
            instance_id = instance["instance_id"]
            if (
                instance_id not in filtered_instance_ids
                and instance_id not in instance_ids_to_keep
            ):
                batch_idx_to_remain.append(i)

        # Update all_input_batch to only keep remaining instances
        remain_dict = {}
        for key in self.all_input_batch.keys():
            if isinstance(self.all_input_batch[key], torch.Tensor):
                remain_dict[key] = self.all_input_batch[key][batch_idx_to_remain]
            elif isinstance(self.all_input_batch[key], list):
                remain_dict[key] = [
                    self.all_input_batch[key][i] for i in batch_idx_to_remain
                ]
        self.all_input_batch = BatchedDataDict(remain_dict)

        assert len(output_batch["instance"]) == len(all_responses), (
            f"Batch size mismatch: {len(output_batch['instance'])} != {len(all_responses)}"
        )

        return all_responses, output_batch

    def filter_easy_hard_instance(self, all_responses: dict) -> tuple[dict, list]:
        """Filter instances where all or none of the trajectories are resolved.

        This is the core DAPO filtering logic. We want to focus on instances
        with mixed success rates, as they provide the most learning signal.

        Args:
            all_responses (dict): Dictionary of instance responses

        Returns:
            tuple[dict, list]: Filtered responses and list of filtered instance IDs
        """
        filtered_instance_ids = []
        keys = list(all_responses.keys())

        for instance_id in keys:
            if len(all_responses[instance_id]) == self.num_trajectories:
                resolved_count = 0
                for trajectory_id in all_responses[instance_id]:
                    if all_responses[instance_id][trajectory_id]["resolved"]:
                        resolved_count += 1

                # Filter if all resolved or none resolved
                if resolved_count == self.num_trajectories or resolved_count == 0:
                    all_responses.pop(instance_id)
                    filtered_instance_ids.append(instance_id)
                    logger.info(
                        f"Filtered instance {instance_id} with resolved ratio "
                        f"{resolved_count}/{self.num_trajectories}"
                    )

        return all_responses, filtered_instance_ids
