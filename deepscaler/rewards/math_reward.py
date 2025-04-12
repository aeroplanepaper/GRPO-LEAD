"""
This module contains the RewardMathFn class, which evaluates mathematical answers
and assigns rewards based on their correctness. It utilizes a language model to 
validate answers when necessary.
"""
from typing import List, Union

from deepscaler.globals import THOUGHT_DELIMITER_START, THOUGHT_DELIMITER_END, OAI_RM_MODEL
from deepscaler.rewards import RewardConfig, RewardFn, RewardInput, RewardOutput, RewardType
from deepscaler.rewards.math_utils.utils import extract_answer, grade_answer_sympy, grade_answer_mathd
from deepscaler.system_prompts import ORM_PROMPT
from deepscaler.utils import call_gemini_llm, call_oai_rm_llm

ORM_USER_TEMPLATE = """
Problem: {problem}
Answer 1: {answer_1}
Answer 2: {answer_2}
"""

class RewardMathFn(RewardFn):
    """
    Reward function for evaluating mathematical answers.

    This class implements the __call__ method to process the input and determine
    the reward based on the correctness of the provided answer compared to the ground truth.
    """
    def __call__(self, input: RewardInput) -> RewardOutput:
        assert input.problem_type == RewardType.MATH, \
            "Invalid problem type: expected 'MATH', but got '{}'".format(input.problem_type)
        
        problem = input.problem
        model_response = input.model_response
        
        # Extract solution.
        if THOUGHT_DELIMITER_END in model_response:
            model_solution = model_response.split(THOUGHT_DELIMITER_END)[0]
        else:
            return RewardOutput(reward=-1, is_correct=False)
        # print("find solution")
        model_answer = extract_answer(model_solution)
        if model_answer is None:
            # print("No model answer")
            return RewardOutput(reward=-1, is_correct=False)

        # Process the ground truth(s)
        ground_truths = input.ground_truth.get("answer", None)
        if ground_truths is None:
            return RewardOutput(reward=self.config.unk_error_reward, is_correct=False)

        # print("Model Answer", model_answer)
        # print("Ground Truth", ground_truths)
        # Convert single answer to list for uniform processing
        if isinstance(ground_truths, (str, float, int)):
            ground_truths = [ground_truths]
            
        # Process each ground truth
        processed_ground_truths = []
        for truth in ground_truths:
            truth = str(truth)
            if "\\boxed" in truth:
                processed_truth = extract_answer(truth)
                if processed_truth is not None:
                    processed_ground_truths.append(processed_truth)
            else:
                processed_ground_truths.append(truth)
        
        if not processed_ground_truths:
            return RewardOutput(reward=0, is_correct=False)

        # Check against all possible correct answers
        for ground_truth in processed_ground_truths:
            # print(model_answer)
            # print(ground_truth)
            is_correct = grade_answer_mathd(model_answer, ground_truth) or grade_answer_sympy(model_answer, ground_truth)
            if is_correct:
                return RewardOutput(reward=self.config.correct_reward, is_correct=True)

        # If latex heuristics fail and ORM is enabled, use LLM as ORM to evaluate correctness
        if self.config.use_math_orm:
            for ground_truth in processed_ground_truths:
                try:
                    orm_response = call_gemini_llm(
                        system_prompt=ORM_PROMPT,
                        prompt=ORM_USER_TEMPLATE.format(problem=problem, answer_1=model_answer, answer_2=ground_truth),
                        temperature=0.0,
                    )

                    if "[[YES]]" in orm_response:
                        return RewardOutput(reward=self.config.correct_reward, is_correct=True)
                except Exception as e:
                    print ("Error calling Gemini ORM, trying OAI RM")
                    orm_response = call_oai_rm_llm(
                        system_prompt=ORM_PROMPT,
                        prompt=ORM_USER_TEMPLATE.format(problem=problem, answer_1=model_answer, answer_2=ground_truth),
                        temperature=0.0,
                        model_id=OAI_RM_MODEL,
                    )
                    
                    if "[[YES]]" in orm_response:
                        return RewardOutput(reward=self.config.correct_reward, is_correct=True)
                    continue
                
        return RewardOutput(reward=self.config.incorrect_reward, is_correct=False)

def max_continuous_ngram(text: str, max_ngram: int = 30):
    """
    Find the n-gram (for n from 1 to max_ngram) that has the longest adjacent (continuous)
    repetition when partitioning the text into non-overlapping chunks.

    The text is preprocessed (converted to lowercase and stripped of punctuation),
    then for each n (1 to max_ngram) we split the word list into non-overlapping segments
    of length n. We then scan these segments to find the maximum consecutive repetition.
    
    Returns a tuple: (best_ngram, best_count, best_n)
    where best_ngram is the repeated n-gram (as a tuple of words),
    best_count is the number of consecutive repetitions,
    and best_n is the size of the n-gram.
    
    If no n-gram repeats continuously (i.e. maximum chain length is less than 2),
    the function returns (None, 0, 0).
    """
    # Preprocess: convert text to lowercase and extract words (ignoring punctuation)
    # words = re.findall(r'\w+', text.lower())
    words = text.split()
    if not words:
        return (None, 0, 0)
    
    best_ngram = None
    best_count = 0
    best_n = 0

    # Consider each n-gram size from 1 to max_ngram.
    for n in range(1, max_ngram + 1):
        if len(words) < n:
            continue
        # Partition words into non-overlapping segments of length n.
        segments = [tuple(words[i:i+n]) for i in range(0, len(words) - n + 1, n)]
        if not segments:
            continue
        
        current_count = 1
        for i in range(1, len(segments)):
            if segments[i] == segments[i-1]:
                current_count += 1
            else:
                if current_count > best_count and current_count >= 2:
                    best_count = current_count
                    best_ngram = segments[i-1]
                    best_n = n
                current_count = 1
        # Check the last chain
        if current_count > best_count and current_count >= 2:
            best_count = current_count
            best_ngram = segments[-1]
            best_n = n

    if best_count < 2:
        return (None, 0, 0)
    return (best_ngram, best_count, best_n)


def find_largest_word(text: str) -> str:
    """
    Find the largest "word" in the text based on character length.
    
    The function first attempts to extract words using alphanumeric characters.
    If no such words are found (e.g. when the text is only punctuation),
    it falls back to splitting the text using str.split(), which splits on any whitespace.
    
    Args:
        text (str): The input text.
    
    Returns:
        str: The largest word or segment found in the text.
    """
    # First, extract alphanumeric words
    words = text.split()
    if not words:
        return ""
    return max(words, key=len)

def deepscaler_reward_fn(solution_str: str, ground_truth: Union[str, List[str]], enable_llm = False):
    reward_config = RewardConfig()
    reward_config.use_math_orm = enable_llm
    reward_fn = RewardMathFn(reward_config)
    reward_response = reward_fn(RewardInput(problem=solution_str, problem_type=RewardType.MATH, model_response=solution_str, ground_truth={"answer": ground_truth}))
    if reward_response.reward < 0:
        best_ngram, best_count, best_n = max_continuous_ngram(solution_str, max_ngram=30)
        if best_count >= 10:
            return -2
        largest_word = find_largest_word(solution_str)
        if len(largest_word) >= 120:
            return -2
        return -1
    return int(reward_response.is_correct)
    # return reward_response.is_correct

# def deepscaler_test_reward_fn(solution_str: str, ground_truth: Union[str, List[str]], enable_llm = False):
# 	reward_config = RewardConfig()
# 	reward_config.use_math_orm = enable_llm
# 	reward_fn = RewardMathFn(reward_config, test=True)
# 	reward_response = reward_fn(RewardInput(problem=solution_str, problem_type=RewardType.MATH, model_response=solution_str, ground_truth={"answer": ground_truth}))
# 	if reward_response.reward < 0:
# 		return -1
# 	return int(reward_response.is_correct)

if __name__ == "__main__":
    reward = RewardMathFn(RewardConfig)
    input = RewardInput(problem="Let $P(x)=x^{4}+2 x^{3}-13 x^{2}-14 x+24$ be a polynomial with roots $r_{1}, r_{2}, r_{3}, r_{4}$. Let $Q$ be the quartic polynomial with roots $r_{1}^{2}, r_{2}^{2}, r_{3}^{2}, r_{4}^{2}$, such that the coefficient of the $x^{4}$ term of $Q$ is 1. Simplify the quotient $Q\\left(x^{2}\\right) / P(x)$, leaving your answer in terms of $x$. (You may assume that $x$ is not equal to any of $\\left.r_{1}, r_{2}, r_{3}, r_{4}\\right)$.", problem_type=RewardType.MATH, model_response="<think> I am omniscient. </think> The answer is \\boxed{24 + 14*x + (-13)*x^2 - 2*x^3 + x^4}.", ground_truth={"answer": ["10", "$x^{4}-2 x^{3}-13 x^{2}+14 x+24$"]})
    output = reward(input)
    print(output)
