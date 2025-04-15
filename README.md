# GRPO-LEAD 🎯

[![🤗 Checkpoints on HuggingFace](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/PlanePaper/LEAD-14B)

---

## Overview

**GRPO-LEAD** (**GRPO** with **L**ength-dependent rewards, **E**xplicit penalties, and **A**dvantage reweighting for **D**ifficulty) is a reinforcement learning pipeline for fine-tuning LLMs for more concise and accurate reasoning in mathematical tasks.

**Our work builds upon and extends the [DeepScaler](https://github.com/agentica-project/rllm) framework — many of our components, including data preprocessing and training scripts, are adapted or extended from it.**
![image-20250412005231869](./figrue/validation_trend.png)

**Figure 1**: Validation Pass@1 over training steps for three configurations: GRPO, GRPO with length reward, and GRPO with length reward plus advantage reweighting.

---

## Getting Started 🔧

### Installation

```
# Installing Python 3.10 Environment.
conda create -n grpo-lead python=3.10 -y
conda activate grpo-lead

# Installing grpo-lead dependencies.
cd grpo-lead
pip install -e ./verl
pip install -e .
```

---

## Data 📊

**Raw training data is in the directory:**

```
grpo-lead/data/train/
```

**The data preprocessing notebook is located at, you can process the dataset on your own:**

```
grpo-lead/data/preprocess/data_preprocess.ipynb
```

**To convert raw `.json` data into `.parquet` format for training, use the following script**:

```
python scripts/data/process_dataset.py
```

---

## Training Scripts

**Training script used for the main experiments:**

```
scripts/train/ds_14b_sft_stage1.sh
```

---

## Results 📈

We evaluate our model with AIME24/25,  all evaluation are conducted with 14k max tokens, 0.6 temperature and 0.01 min-p with 32 samples per question. 

| **Model Name**   | **AIME24 Cons@32** | **AIME24 Pass@1** | **AIME24 Len_avg** | **AIME25 Cons@32** | **AIME25 Pass@1** | **AIME25 Len_avg** |
|:----------------:|:------------------:|:------------------:|:------------------:|:------------------:|:------------------:|:------------------:|
| **DeepSeek-Distilled-14B** | 0.800              | 0.614              | 9182               | 0.633              | 0.429              | 10046              |
| **Light-R1-14B** | 0.833              | 0.641              | 9571               | 0.767              | 0.505              | 10194              |
| **LEAD-14B-stage1**  | 0.833              | 0.629              | 8790               | 0.767              | 0.523              | 9371               |
| **LEAD-14B-stage3**  | 0.867              | 0.650              | 8267               | 0.767              | 0.539              | 8668               |
