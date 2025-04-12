"""Script to prepare DeepScaler training and test datasets.

This script processes math problem datasets into a standardized format for training
and testing DeepScaler models. It loads problems from specified datasets, adds
instruction prompts, and saves the processed data as parquet files.
"""

import argparse
import os
from copy import deepcopy
from typing import Dict, List, Optional, Any

import pandas as pd
from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed

from deepscaler.data.utils import load_dataset
from deepscaler.data.dataset_types import TrainDataset, TestDataset
import random


def extract_solution(solution_str: str) -> str:
    """Extract the final boxed solution from a solution string.

    Args:
        solution_str: Raw solution string that may contain multiple boxed answers

    Returns:
        The final boxed answer with box notation removed
    """
    return remove_boxed(last_boxed_only_string(solution_str))


def make_map_fn(split: str):
    """Create a mapping function to process dataset examples.

    Args:
        split: Dataset split name ('train' or 'test')

    Returns:
        Function that processes individual dataset examples
    """
    def process_fn(example: Dict[str, Any], idx: int, use_chinese=False, test=False) -> Optional[Dict[str, Any]]:
        question = example.pop('problem')
        print(1)
        # prompt_type = int(random.random() * 2)
        prompt_type = 0
        sys_prompt = ['''You are Qwen, a helpful assistant for solving challenging math problems. You first thinks about the reasoning process in the mind and then provides the user with the answer.''', 
            ''' A conversation between User and Assistant. The user asks a question, and the Assistant solves it using LATEX.
            The assistant first thinks about the reasoning process in the mind in LATEX and then provides the user
            with the answer. Only work with exact numbers. Only give your final answer if you are certain it is correct. After you get your final answer, return the final answer within \\boxed{}.''']
        user_prmopt = [ question + "\nLet's think step by step and put the final answer within \\boxed{}", 
                        question.strip() + "\nWhat type of problem is this? Can you solve it? Think carefully and \\boxed{} the final answer. You can \\boxed{} any answer during thinking."]
        instruction_test = "Let's think step by step and output the final answer within \\boxed{}, after taking modulo 1000."
        answer = example.pop('answer')

        data = {
            "data_source": "",
            "prompt": [
                # {"role": "system", 
                # "content": sys_prompt[prompt_type]},
                {
                "role": "user",
                "content": user_prmopt[prompt_type]
            }],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": answer
            },
            "extra_info": {
                'split': split,
                'index': idx
            }
        }
        return data
    # def process_fn(example: Dict[str, Any], idx: int, use_chinese=False, test=False) -> Optional[Dict[str, Any]]:
    #     question = example.pop('problem')
    #     prompt_type = int(random.random() * 4)
    #     print(prompt_type)
    #     if prompt_type >= 2:
    #         # print(1)
    #         sys_prompt = ['''A conversation between User and Assistant. The user asks a question, and the Assistant solves it using LATEX.
    #             The assistant first thinks about the reasoning process in the mind in LATEX and then provides the user
    #             with the answer. i.e., <think> reasoning process here </think> answer here.''', 
    #             ''' Solve the math problem from the user. Only work with exact numbers. Only submit an answer if you are sure. After you get your final answer, return the final answer within \\boxed{}.''']
    #         user_prmopt = [question + "\nWhat type of problem is this? Can you solve it? Think carefully and \\boxed{} the final answer. You can \\boxed{} any answer during thinking.", 
    #                         question]
    #         instruction_test = "Let's think step by step and output the final answer within \\boxed{}, after taking modulo 1000."
    #         answer = example.pop('answer')

    #         data = {
    #             "data_source": "",
    #             "prompt": [
    #                 {"role": "system", 
    #                 "content": sys_prompt[prompt_type - 2]},
    #                 {
    #                 "role": "user",
    #                 "content": user_prmopt[prompt_type - 2]
    #             }],
    #             "ability": "math",
    #             "reward_model": {
    #                 "style": "rule",
    #                 "ground_truth": answer
    #             },
    #             "extra_info": {
    #                 'split': split,
    #                 'index': idx
    #             }
    #         }
    #     else:
    #         # print(2)
    #         # instruction = "Let's think step by step and output the final answer within \\boxed{}."
    #         # instruction_test = "Let's think step by step and output the final answer within \\boxed{}, after taking modulo 1000."
    #         # chinese_instruction = "请一步一步思考，并将最终答案写在 \\boxed{} 中。"
    #         # instruction = instruction if not test else instruction_test
    #         # question = f"{question} {instruction}" if not use_chinese else f"{question} {chinese_instruction}"
    #         answer = example.pop('answer')
    #         user_prmopt = [question + "Let's think step by step and output the final answer within \\boxed{}.", 
    #             question.strip() + " Please think carefully, and only work with exact numbers. and return an answer if you are sure it is correct. and \\boxed{} the final answer"]

    #         data = {
    #             "data_source": "",
    #             "prompt": [{
    #                 "role": "user",
    #                 "content": user_prmopt[prompt_type]
    #             }],
    #             "ability": "math",
    #             "reward_model": {
    #                 "style": "rule",
    #                 "ground_truth": answer
    #             },
    #             "extra_info": {
    #                 'split': split,
    #                 'index': idx
    #             }
    #         }
    #     return data
    return process_fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process datasets for DeepScaler training')
    parser.add_argument('--local_dir', default=os.path.expanduser('~/deepscaler/data'),
                       help='Local directory to save processed datasets')
    parser.add_argument('--hdfs_dir', default=None,
                       help='Optional HDFS directory to copy datasets to')
    args = parser.parse_args()

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir
    
    # Make local directory if it doesn't exist
    # makedirs(local_dir)

    # Initialize datasets
    train_datasets = [TrainDataset.DEEPSCALER]
    # train_datasets = [TrainDataset.final_problems_nonzero]
    train_dataset = load_dataset(train_datasets[0])
    print(len(train_dataset))
    # test_datasets = [TestDataset.AIME, TestDataset.AMC, TestDataset.MATH, TestDataset.MINERVA, TestDataset.OLYMPIAD_BENCH]
    # test_datasets = [TestDataset.OURS]
    
    # test_datasets_data = [load_dataset(d) for d in test_datasets]

    # Process training data
    train_data: List[Dict[str, Any]] = []
    process_fn = make_map_fn('train')
    for idx, example in enumerate(train_dataset):
        # example_copy = deepcopy(example)
        processed_example = process_fn(example, idx)
        # processed_example_ch = process_fn(example_copy, idx, use_chinese=True)
        if processed_example is not None:
            train_data.append(processed_example)
            # train_data.append(processed_example_ch)

    # Process and save each test dataset separately
    # for test_dataset, test_data_list in zip(test_datasets, test_datasets_data):
    #     test_data: List[Dict[str, Any]] = []
    #     process_fn = make_map_fn('test')
    #     for idx, example in enumerate(test_data_list):
    #         processed_example = process_fn(example, idx, test=True)
    #         if processed_example is not None:
    #             test_data.append(processed_example)

        # dataset_name = test_dataset.value.lower()
        # test_df = pd.DataFrame(test_data)
        # test_df.to_parquet(os.path.join(local_dir, f'{dataset_name}.parquet'))
        # print(f"{dataset_name} test data size:", len(test_data))

    # Save training dataset
    print("train data size:", len(train_data))
    train_df = pd.DataFrame(train_data)
    train_df.to_parquet(os.path.join(local_dir, 'train.parquet'))

    # Optionally copy to HDFS
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)