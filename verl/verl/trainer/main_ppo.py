# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl import DataProto
import torch
import ray
import hydra

from deepscaler.rewards.math_reward import deepscaler_reward_fn
import os
import json
import uuid
from datetime import datetime
import random
import math as math_o


Failed_RESULTS_DIR = "failed_results_14b"
Correct_RESULTS_DIR = "correct_results_14b"

os.makedirs(Failed_RESULTS_DIR, exist_ok=True)
os.makedirs(Correct_RESULTS_DIR, exist_ok=True)

def get_custom_reward_fn(config):
    import importlib.util, os

    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    if not file_path:
        return None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error loading module from '{file_path}': {e}")

    function_name = reward_fn_config.get("name")

    if not hasattr(module, function_name):
        raise AttributeError(f"Reward function '{function_name}' not found in '{file_path}'.")

    print(f"using customized reward function '{function_name}' from '{file_path}'")

    return getattr(module, function_name)

class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, test=False) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.test = test

    def get_reweight_rewards(self, rewards, completions_length, index):
        """
        rewards: list of rewards (1-d tensor)
        completions_length: list of completions length (1-d tensor)
        index: list indicating which question each sample belongs to
        """

        unique_questions = set(index)
        
        for q in unique_questions:
            group_indices = [i for i, q_idx in enumerate(index) if q_idx == q]

            # 初始化统计量
            sum_len = 0
            var_len = 0
            num_correct = 0
            min1, min2, min3 = float('inf'), float('inf'), float('inf')
            
            for i in group_indices:
                L = completions_length[i]
                if L < min1:
                    min3 = min2
                    min2 = min1
                    min1 = L
                elif L < min2:
                    min3 = min2
                    min2 = L
                elif L < min3:
                    min3 = L
                if rewards[i] > 0:
                    num_correct += 1
                    sum_len += L

            avg_min = (min1 + min2 + min3) / 3
            if avg_min <= 4000:
                continue
            
            avg_len = sum_len / num_correct if num_correct > 0 else 0.0
            std_len = 1.0
            if num_correct > 1:
                for i in group_indices:
                    if rewards[i] > 0:
                        var_len += (completions_length[i] - avg_len) ** 2
                val_len = var_len / (num_correct - 1)
                std_len = math_o.sqrt(val_len)
                
            alpha = 0.05
            
            min_three_avg = (min1 + min2 + min3) / 3
            
            for i in group_indices:
                if rewards[i] > 0:
                    L_i = completions_length[i]
                    epsilon = 1e-6
                    z_i = (L_i - avg_len) / (std_len + epsilon)
                    rewards[i] = rewards[i] * math_o.exp(-alpha * z_i)
                    
        return rewards


    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']


        already_print_data_sources = {}

        from concurrent.futures import ThreadPoolExecutor
        from typing import Dict, Any
        #import threading
        # Thread-safe dict for tracking printed data sources
        # print_lock = threading.Lock()
        
        def process_item(args):
            i, data_item, already_print_data_sources = args
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses'] 
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)
            print(sequences_str)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = deepscaler_reward_fn
            score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth)

            if score < 1:

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  
                unique_id = uuid.uuid4().hex 
                file_name = f"{Failed_RESULTS_DIR}/failed_{timestamp}_{unique_id}.json"

                data_to_save = {
                    "sequences_str": sequences_str,
                    "ground_truth": ground_truth
                }

                with open(file_name, "w", encoding="utf-8") as f:
                    json.dump(data_to_save, f, ensure_ascii=False, indent=4)
            
            if score >= 1:

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  
                unique_id = uuid.uuid4().hex 
                file_name = f"{Correct_RESULTS_DIR}/corrected_{timestamp}_{unique_id}.json"

                data_to_save = {
                    "sequences_str": sequences_str,
                    "ground_truth": ground_truth
                }

                with open(file_name, "w", encoding="utf-8") as f:
                    json.dump(data_to_save, f, ensure_ascii=False, indent=4)
            
            
            return i, score, valid_response_length

        # Process items in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=96) as executor:
            args = [(i, data[i], already_print_data_sources) for i in range(len(data))]
            results = list(executor.map(process_item, args))


        # Fill reward tensor with results
        rewards = []
        completions_length = []
        skipped_idx = []

        if self.test:
            for i, score, valid_response_length in results:
                if score < 0:
                    score = 0.0
                    # skipped_idx.append(i)
                    # continue
                rewards.append(score)
                completions_length.append(valid_response_length)
            reweight_rewards = rewards
        else:
            for i, score, valid_response_length in results:
                if score == -1:
                    rewards.append(0.0)
                elif score == 0:
                    rewards.append(-1) # control the reward of negative answer
                # elif score == -2:
                #     rewards.append(-1.5) # format penalty, if there are repetition.
                else:
                    rewards.append(score)
                completions_length.append(valid_response_length)
            index = data.non_tensor_batch['uid']
            # print(rewards)
            reweight_rewards = self.get_reweight_rewards(rewards, completions_length, index)
            # reweight_rewards = rewards # no reweight
            

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        mask = torch.ones(reward_tensor.shape[0], dtype=torch.bool)
        mask[skipped_idx] = False
        reward_tensor = reward_tensor[mask]

        skipped_idx = set(skipped_idx)
        cnt_skipped = 0
        for i, score, valid_response_length in results:
            if i in skipped_idx:
                cnt_skipped += 1
                continue
            reward_tensor[i-cnt_skipped, valid_response_length - 1] = reweight_rewards[i-cnt_skipped]

        return reward_tensor



@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:

    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
def main_task(config):
    from verl.utils.fs import copy_to_local
    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    local_path = copy_to_local(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer, hf_processor
    tokenizer = hf_tokenizer(local_path)
    processor = hf_processor(local_path, use_fast=True)  # used for multimodal LLM, could be none

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker)
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_manager_name = config.reward_model.get("reward_manager", "naive")
    if reward_manager_name == 'naive':
        from verl.workers.reward_manager import NaiveRewardManager
        reward_manager_cls = NaiveRewardManager
    elif reward_manager_name == 'prime':
        from verl.workers.reward_manager import PrimeRewardManager
        reward_manager_cls = PrimeRewardManager
    else:
        raise NotImplementedError

    # compute_score = get_custom_reward_fn(config)
    # reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, compute_score=compute_score)

    # Note that we always use function-based RM for validation
    # val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, compute_score=compute_score)



    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0)

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1, test=True)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            processor=processor,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn)
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
