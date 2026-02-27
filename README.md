# GRRM: Group Relative Reward Modeling for Machine Translation

<a href="https://arxiv.org/abs/2602.14028">
  <img src="https://img.shields.io/badge/Paper-arXiv-blue"></a>
<a href="https://huggingface.co/collections/double7/grrm">
  <img src="https://img.shields.io/badge/Models-Hugging%20Face-brightgreen"></a>
<a href="https://huggingface.co/collections/double7/grrm">
  <img src="https://img.shields.io/badge/Data-Hugging%20Face-orange"></a>
<a href="LICENSE">
  <img src="https://img.shields.io/badge/License-MIT-yellow"></a>

Official implementation of the paper: **GRRM: Group Relative Reward Modeling for Machine Translation**.

This repository contains the code for **GRRM (Group Relative Reward Model)**, a Generative Reward Model** instantiated under the proposed **Group Quality Metric (GQM) paradigm for **reinforcement learning-based machine translation (MT) optimization** with **GRPO**.

## Highlights

- **Improved reward accuracy:** large gains especially on challenging samples (idioms, slang, terminology) and restores reward variance needed by GRPO advantages. Also Robust to reward hacking.
- **GRRM**: efficient GQM-based reward model trained with SFT + RLVR.
- **Cross-lingual generalization**: GRRM trained on Zh–En can support multilingual MT optimization.
- **MT optimization**: GRPO training with GRRM yields strong improvements on WMT-style benchmarks and challenge sets.


Ranking accuracy performance and downstream translation performance on Seed-X-Challenge:
<p align="center">
  <img src="static/teaser.png" alt="Teaser" width="600">
</p>

## Method Overview


**Core idea:** Standard generative reward models often evaluate candidates independently (**Scalar Quality Metric, SQM**) and suffer from score saturation, which causes vanishing advantages in GRPO and stalls optimization.  
We propose **GQM**, which evaluates a *group* of candidates jointly to produce reliable fine-grained intra-group ranking, and implement it as **GRRM**, a high-throughput reward model with explicit comparative reasoning.

**GQM** evaluates the *entire group* together and outputs:
- comparative analysis
- predicted ranking
- scores consistent with that ranking

<p align="center">
  <img src="static/GRRM_framework.png" alt="GRRM Framework" width="600">
</p>

## Performance Overview


---

## Getting Started

### Installation

```bash
git clone https://github.com/NJUNLP/GRRM.git
cd GRRM
pip install -e . --no-build-isolation
```

Extra dependencies available:
- `infer`: install vllm for inference.
- `eval`: sacrebleu and bleurt for translation evaluation.

For model training, additional dependencies are required. We use [Llama-Factory](https://github.com/hiyouga/LlamaFactory) for SFT training and [verl](https://github.com/DoubleVII/verl) for reinforcement learning training. The latter is a forked version of the original repository with some custom modifications.


---

## Quick Use

### 1) Use GRRM to rank a group of translations (GQM inference)

[This script](inference/run_rm_GQM.py) performs Group Quality Metric (GQM) inference using vLLM to evaluate and rank multiple translation candidates. It includes prompt templates, result parsing and return the scores and raw model outputs.

Example usage:
```python
import inference.run_rm_GQM as rm_GQM

output = rm_GQM.func_call(
    model_path="double7/GRRM-Qwen2.5-7B",
    src_list=["I have a frog in my throat."],
    mt_list=[["我嗓子有点哑。", "我嗓子眼里好像有只青蛙。"]],
    src_langs=["en"],
    trg_langs=["zh"],
    temperature=1.0,
    top_p=1.0,
    max_new_tokens=8192,
    retry=6,
    prompt_type="ranking_score",
)

# output["scores"] -> [[8, 3]]  # scores for each candidate
# output["responses"] -> ["...model response text..."]
```


> [!NOTE]  
> - Inference at low `temperature` may fail. Set `retry` to automatically retry with higher temperature.
> - For GRRM, set `prompt_type` to `ranking_score`.

### 2) Use GRRM-Optimized MT model for translation inference

[This script](inference/run_mt.py) performs machine translation inference using vLLM. It supports multiple prompt formats and answer extraction methods for different model types. It returns a dictionary with translation responses and raw model outputs.

Example usage:
```python
import inference.run_mt as mt

output = mt.func_call(
    model_path="double7/Qwen2.5-7B-MT-GRRM-Optimized",
    src_list=["The grass is always greener on the other side.", "INTJ总是装E"],
    src_langs=["en", "zh"],
    trg_langs=["zh", "en"],
    temperature=0.7,
    top_p=0.9,
    max_new_tokens=8192,
    retry=6,
    prompt_type="codeblock-think",
    use_chat_template=True,
)

# output["responses"] -> ["这山望着那山高。", "INTJs are always putting on an extroverted front."]
# output["raw_outputs"] -> ["...raw model output...", "...raw model output..."]
```

> [!NOTE]  
> - Inference at low `temperature` may fail. Set `retry` to automatically retry with higher temperature.
> - For GRRM-Optimized MT models, set `prompt_type` to `codeblock-think`.

## GRRM Training Pipeline

### SFT Cold-start

<details><summary>SFT Cold-start Pipeline</summary>

#### Data Preprocessing

We have prepared the translation ranking data based on TowerBlocks-MT. Run the following command to download it:
```bash
hf download double7/TowerBlocks-MT-Ranking --repo-type dataset --local-dir parquet_data/TowerBlocks-MT-Ranking
```

Run the command below to construct GQM training data for supervised fine-tuning:
```bash
python scripts/prepare_SFT_GQM_training_data.py \
  --data_path  parquet_data/TowerBlocks-MT-Ranking/train.parquet \
  --output_path sft_towerblocks_mt_ranking.parquet \
  --mt_key mt_texts \
  --score_key scores \
  --analysis_key analysis
```
#### Training

We employ [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for supervised fine-tuning.

**Setup:**
Install the LLaMA-Factory runtime environment according to the [documentation](https://github.com/hiyouga/LLaMA-Factory?tab=readme-ov-file#getting-started).

**Dataset:**
- Place the SFT data in the `LLaMA-Factory/data` directory.
- Add the dataset information to `LLaMA-Factory/data/dataset_info.json`:
```json
{
    "sft_towerblocks_mt_ranking": {
        "file_name": "sft_towerblocks_mt_ranking.parquet",
        "formatting": "sharegpt",
        "columns": {
            "messages": "messages"
        },
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant"
        }
    }
}
``` 

**Training Config:**
Create a yaml file to configure training hyperparameters, e.g., `qwen7b_full_sft_ds3.yaml`:
```yaml
### model
model_name_or_path: Qwen/Qwen2.5-7B
flash_attn: fa2

### method
stage: sft
do_train: true
finetuning_type: full
deepspeed: examples/deepspeed/ds_z3_config.json

### dataset
dataset: sft_towerblocks_mt_ranking
template: chatml
cutoff_len: 12000
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: saves/qwen-7b/full/grrm_sft
logging_steps: 1
save_steps: 500000
plot_loss: true
overwrite_output_dir: true
report_to: wandb  # choices: [none, wandb, tensorboard, swanlab, mlflow]

### train
per_device_train_batch_size: 4
gradient_accumulation_steps: 1
learning_rate: 6.0e-6
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
# warmup_steps: 300
bf16: true
ddp_timeout: 180000000
resume_from_checkpoint: null
```

> [!NOTE]  
> Depending on the number of computing devices, the total batch_size is calculated as `your_device_num * per_device_train_batch_size * gradient_accumulation_steps`.


**Run Training:**

Run the following command to start training:
```bash
FORCE_TORCHRUN=1 llamafactory-cli train path/to/qwen7b_full_sft_ds3.yaml 
```

> [!TIP]
> For Multiple Nodes training, refer to [Supervised Fine-Tuning on Multiple Nodes](https://github.com/hiyouga/LLaMA-Factory/tree/main/examples#supervised-fine-tuning-on-multiple-nodes).

</details>


### RLVR Training

<details><summary>RLVR Training Pipeline</summary>

#### Data Preprocessing

Run the command below to construct GQM training data for RLVR learning:
```bash
python scripts/prepare_RL_GQM_training_data.py \
  --data_path  parquet_data/TowerBlocks-MT-Ranking/train.parquet \
  --output_path rl_towerblocks_mt_ranking.parquet \
  --mt_key mt_texts \
  --score_key scores \
  --analysis_key analysis \
  --subgroup_augment 4 \
  --shuffle_augment 4
```

> [!NOTE]  
> The arguments `subgroup_augment` and `shuffle_augment` control the data augmentation mentioned in the paper, where `subgroup_augment` denotes the number of subgroups to sample from each original annotated group, `shuffle_augment` denotes the number of times to randomly shuffle the candidate order.

Run the command below to construct GQM test data for validation:
```bash
python scripts/prepare_RL_GQM_training_data.py \
  --data_path  parquet_data/TowerBlocks-MT-Ranking/test.parquet \
  --output_path rl_towerblocks_mt_ranking.test.parquet \
  --mt_key mt_texts \
  --score_key scores \
  --analysis_key analysis \
  --subgroup_augment 0 \
  --shuffle_augment 0
```

#### Training

We employ [verl with Generative Reward Models support](https://github.com/DoubleVII/verl/tree/gen_rm) for RLVR training. Our **Ranking Accuracy Reward** is defined in [this script](https://github.com/DoubleVII/verl/blob/gen_rm/reward_utils/ranking_score_reward.py).

**Setup:**

```bash
git clone https://github.com/DoubleVII/verl.git
git checkout gen_rm # switch to gen_rm branch is mandatory
```

Install the verl runtime environment according to the [documentation](https://verl.readthedocs.io/en/latest/start/install.html).

**Training Config:**

Here we list the most frequently used arguments. For reward settings, refer to the *Reward Control* sections.

<details><summary>RLVR Training Config</summary>

```bash
# data
TRAIN_FILES=path/to/rl_towerblocks_mt_ranking.parquet
VAL_FILES=path/to/rl_towerblocks_mt_ranking.test.parquet
TRAIN_BATCH_SIZE=512
VAL_BATCH_SIZE=256
MAX_PROMPT_LENGTH=1600
MAX_RESPONSE_LENGTH=4096

# Actor/Optimization
ACTOR_MODEL_PATH=path/to/grrm_sft
ACTOR_OPT_LR=1e-5
ACTOR_LR_SCHEDULER_TYPE=cosine
ACTOR_MIN_LR_RATIO=0.2
ACTOR_PPO_MINI_BSZ=128
ACTOR_PPO_MICRO_BSZ_PER_GPU=8
ACTOR_LOSS_MODE=gspo
ACTOR_LOSS_AGG_MODE=seq-mean-token-mean
ACTOR_CLIP_RATIO_LOW=0.0003
ACTOR_CLIP_RATIO_HIGH=0.0004
ACTOR_STRATEGY=fsdp2
ACTOR_MODEL_DTYPE=bf16
ACTOR_ULYSSES_SP_SIZE=1

# Rollout
ROLLOUT_NAME=vllm
ROLLOUT_MODE=sync
ROLLOUT_CALC_LOG_PROBS=True
ROLLOUT_LOG_PROB_MB_PER_GPU=128
ROLLOUT_TENSOR_MP_SIZE=1
ROLLOUT_GPU_MEM_UTIL=0.5
ROLLOUT_N=8
ROLLOUT_TEMPERATURE=1.0
ROLLOUT_TOP_P=1.0
ROLLOUT_TOP_K=-1
VAL_TEMPERATURE=1.0
VAL_TOP_P=0.7
VAL_TOP_K=-1
VAL_DO_SAMPLE=true



# Algorithm
ADV_ESTIMATOR=grpo
KL_COEF=0.001
NORM_ADV_BY_STD_IN_GRPO=False
ACTOR_USE_KL_LOSS=False

# Reward Control
REWARD_MANAGER=dapo
REWARD_OVERLONG_BUFFER_ENABLE=false
REWARD_OVERLONG_BUFFER_LEN=1024
REWARD_OVERLONG_BUFFER_PENALTY_FACTOR=1.0
REWARD_OVERLONG_BUFFER_LOG=false
REWARD_MAX_RESP_LEN=3200
CUSTOM_REWARD_FN_PATH=reward_utils/ranking_score_reward.py
CUSTOM_REWARD_FN_NAME=ranking_score_reward_fn

# Training/Logging
PROJECT_NAME=grrm_verl
EXPERIMENT_NAME=grrm_rlvr.exp
N_GPUS_PER_NODE=8
VAL_BEFORE_TRAIN=True
NNODES=2
SAVE_FREQ=200
TEST_FREQ=20
RESUME_MODE=auto
TOTAL_EPOCHS=1
DEFAULT_LOCAL_DIR=path/to/grrm_rlvr
```
</details>

**Run Training:**

Run the following command to start training:
<details><summary>verl Launch Command</summary>

```bash
ray job submit \
    --runtime-env=verl/trainer/runtime_env.yaml \
    --no-wait \
    -- \
    python3 -m \
    verl.trainer.main_ppo \
    data.train_files=${TRAIN_FILES} \
    data.val_files=${VAL_FILES} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.val_batch_size=${VAL_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.filter_overlong_prompts=True \
    data.shuffle=True \
    data.truncation=error \
    actor_rollout_ref.model.path=${ACTOR_MODEL_PATH} \
    actor_rollout_ref.actor.shuffle=True \
    actor_rollout_ref.actor.optim.lr=${ACTOR_OPT_LR} \
    actor_rollout_ref.actor.optim.lr_scheduler_type=${ACTOR_LR_SCHEDULER_TYPE} \
    actor_rollout_ref.actor.optim.min_lr_ratio=${ACTOR_MIN_LR_RATIO} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ACTOR_PPO_MINI_BSZ} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ACTOR_PPO_MICRO_BSZ_PER_GPU} \
    actor_rollout_ref.actor.policy_loss.loss_mode=${ACTOR_LOSS_MODE} \
    actor_rollout_ref.actor.loss_agg_mode=${ACTOR_LOSS_AGG_MODE} \
    actor_rollout_ref.actor.use_kl_loss=${ACTOR_USE_KL_LOSS} \
    actor_rollout_ref.actor.kl_loss_type=mse \
    actor_rollout_ref.actor.clip_ratio_low=${ACTOR_CLIP_RATIO_LOW} \
    actor_rollout_ref.actor.clip_ratio_high=${ACTOR_CLIP_RATIO_HIGH} \
    actor_rollout_ref.actor.strategy=${ACTOR_STRATEGY} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=${ACTOR_MODEL_DTYPE} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${ACTOR_ULYSSES_SP_SIZE} \
    actor_rollout_ref.rollout.name=${ROLLOUT_NAME} \
    actor_rollout_ref.rollout.mode=${ROLLOUT_MODE} \
    actor_rollout_ref.rollout.calculate_log_probs=${ROLLOUT_CALC_LOG_PROBS} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${ROLLOUT_LOG_PROB_MB_PER_GPU} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TENSOR_MP_SIZE} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL} \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE} \
    actor_rollout_ref.rollout.top_p=${ROLLOUT_TOP_P} \
    actor_rollout_ref.rollout.top_k=${ROLLOUT_TOP_K} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${VAL_TOP_K} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=${VAL_DO_SAMPLE} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=64 \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    critic.optim.lr=1e-5 \
    critic.model.path=null \
    critic.ppo_micro_batch_size_per_gpu=4 \
    algorithm.adv_estimator=${ADV_ESTIMATOR} \
    algorithm.kl_ctrl.kl_coef=${KL_COEF} \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=${NORM_ADV_BY_STD_IN_GRPO} \
    reward_model.reward_manager=${REWARD_MANAGER} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${REWARD_OVERLONG_BUFFER_ENABLE} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${REWARD_OVERLONG_BUFFER_LEN} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${REWARD_OVERLONG_BUFFER_PENALTY_FACTOR} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=${REWARD_OVERLONG_BUFFER_LOG} \
    +reward_model.reward_kwargs.max_resp_len=${REWARD_MAX_RESP_LEN} \
    custom_reward_function.path=${CUSTOM_REWARD_FN_PATH} \
    custom_reward_function.name=${CUSTOM_REWARD_FN_NAME} \
    trainer.logger=[console,wandb] \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.val_before_train=${VAL_BEFORE_TRAIN} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.resume_mode=${RESUME_MODE} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.default_local_dir=${DEFAULT_LOCAL_DIR}
```
</details>

</details>



## Use GRRM for downstream machine translation GRPO training

We employ [verl with Generative Reward Models support](https://github.com/DoubleVII/verl/tree/gen_rm) for GRPO training.

![verl_GenRM_pipeline](static/verl_GenRM_pipeline.png)

### SFT Cold-Start
<details><summary>SFT Cold-start Pipeline</summary>

#### Data Preprocessing

We have prepared the translation CoT data based on TowerBlocks-MT. Run the following command to download it:
```bash
hf download double7/TowerBlocks-MT-CoT-ZhEn --repo-type dataset --local-dir parquet_data/TowerBlocks-MT-CoT-ZhEn
```
Run the command below to add translation instructions to the translation pairs for supervised fine-tuning:
```bash
python scripts/prepare_SFT_MT_training_data.py \
  --data_path  parquet_data/TowerBlocks-MT-CoT-ZhEn/train.parquet \
  --output_path sft_towerblocks_mt_cot.parquet \
  --response_key response
```
#### Training

We employ [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for supervised fine-tuning.

**Setup:**
Install the LLaMA-Factory runtime environment according to the [documentation](https://github.com/hiyouga/LLaMA-Factory?tab=readme-ov-file#getting-started).

**Dataset:**
- Place the SFT data in the `LLaMA-Factory/data` directory.
- Add the dataset information to `LLaMA-Factory/data/dataset_info.json`:
```json
{
    "sft_towerblocks_mt_cot": {
        "file_name": "sft_towerblocks_mt_cot.parquet",
        "formatting": "sharegpt",
        "columns": {
            "messages": "messages"
        },
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant"
        }
    }
}
``` 

**Training Config:**
Create a yaml file to configure training hyperparameters, e.g., `qwen7b_full_sft_ds3.yaml`:
```yaml
### model
model_name_or_path: Qwen2.5-7B
flash_attn: fa2

### method
stage: sft
do_train: true
finetuning_type: full
deepspeed: examples/deepspeed/ds_z3_config.json

### dataset
dataset: sft_towerblocks_mt_cot
template: chatml
cutoff_len: 12000
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: saves/qwen-7b/full/sft/cot_mt_sft
logging_steps: 1
save_steps: 2500
plot_loss: true
overwrite_output_dir: true
report_to: wandb  # choices: [none, wandb, tensorboard, swanlab, mlflow]

### train
per_device_train_batch_size: 4
gradient_accumulation_steps: 2
learning_rate: 1.0e-5
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
# warmup_steps: 300
bf16: true
ddp_timeout: 180000000
resume_from_checkpoint: null
```

> [!NOTE]  
> Depending on the number of computing devices, the total batch_size is calculated as `your_device_num * per_device_train_batch_size * gradient_accumulation_steps`.


**Run Training:**

Run the following command to start training:
```bash
FORCE_TORCHRUN=1 llamafactory-cli train path/to/qwen7b_full_sft_ds3.yaml 
```

> [!TIP]
> For Multiple Nodes training, refer to [Supervised Fine-Tuning on Multiple Nodes](https://github.com/hiyouga/LLaMA-Factory/tree/main/examples#supervised-fine-tuning-on-multiple-nodes).

</details>


### GRPO Training

<details><summary>GRPO Training Pipeline</summary>

#### Data Preprocessing

We have prepared the translation task data extracted from TowerBlocks. Run the following command to download it:
```bash
hf download double7/TowerBlocks-MT --repo-type dataset --local-dir parquet_data/TowerBlocks-MT
```

Run the command below to construct translation training data for GRPO learning:
```bash
python scripts/prepare_RL_MT_training_data.py construct_tower \
  --data_path  parquet_data/TowerBlocks-MT/data/train.parquet \
  --output_path rl_towerblocks_mt.parquet
```

Run the command below to construct translation training data with Cross-Lingual Augmentation (CLA):
```bash
python scripts/prepare_RL_MT_training_data.py construct_towerx \
  --data_path  parquet_data/TowerBlocks-MT/data/train.parquet \
  --output_path rl_towerblocks_mt_cla.parquet
```

The output includes both the original data and the CLA data, doubling the total size.

#### Training

**Setup:**

```bash
git clone https://github.com/DoubleVII/verl.git
git checkout gen_rm # switch to gen_rm branch is mandatory
```

Install the verl runtime environment according to the [documentation](https://verl.readthedocs.io/en/latest/start/install.html). **Both `vllm` and `sglang` are required.**
```bash
git checkout gen_rm
```

**Training Config:**

Here we list the most frequently used arguments. For reward settings, refer to the *Reward Model* and *Reward Control* sections.

<details><summary>GRPO Training Config</summary>

```bash
# data
TRAIN_FILES=path/to/rl_towerblocks_mt.parquet
VAL_FILES=path/to/your_test_data.parquet
TRAIN_BATCH_SIZE=512
VAL_BATCH_SIZE=256
MAX_PROMPT_LENGTH=1280
MAX_RESPONSE_LENGTH=4096

# Actor/Optimization
ACTOR_MODEL_PATH=path/to/cot_mt_sft
ACTOR_OPT_LR=1e-5
ACTOR_LR_SCHEDULER_TYPE=constant
ACTOR_MIN_LR_RATIO=null
ACTOR_WARMUP_STEPS=-1
ACTOR_PPO_MINI_BSZ=128
ACTOR_PPO_MICRO_BSZ_PER_GPU=32
ACTOR_LOSS_MODE=gspo
ACTOR_LOSS_AGG_MODE=seq-mean-token-mean
ACTOR_CLIP_RATIO_LOW=0.0003
ACTOR_CLIP_RATIO_HIGH=0.0004
ACTOR_STRATEGY=fsdp2
ACTOR_MODEL_DTYPE=bf16
ACTOR_ULYSSES_SP_SIZE=1
ACTOR_SFT_COEF=0.1 # Coefficient for SFT loss, see Appendix C. (Stabilizing Group Relative Policy Optimization) in the paper.

# Rollout
ROLLOUT_NAME=sglang # we use sglang for policy rollout and vllm for reward model rollout
ROLLOUT_MODE=sync
ROLLOUT_LOG_PROB_MB_PER_GPU=128
ROLLOUT_TENSOR_MP_SIZE=1
ROLLOUT_GPU_MEM_UTIL=0.6
ROLLOUT_N=4
ROLLOUT_TEMPERATURE=1.0
ROLLOUT_TOP_P=1.0
ROLLOUT_TOP_K=-1
VAL_TEMPERATURE=1.0
VAL_TOP_P=0.7
VAL_TOP_K=-1
VAL_DO_SAMPLE=true

# Algorithm
ADV_ESTIMATOR=grpo
KL_COEF=0.001
NORM_ADV_BY_STD_IN_GRPO=False
ACTOR_USE_KL_LOSS=False

# Reward Model
REWARD_MODEL_PATH=path/to/grrm_rlvr
REWARD_RESPONSE_LENGTH=8192
REWARD_PROMPT_LENGTH=2048
REWARD_MAX_NUM_BATCHED_TOKENS=12000
REWARD_ROLLOUT_NAME=vllm # we use sglang for policy rollout and vllm for reward model rollout
REWARD_ROLLOUT_MODE=sync
REWARD_GPU_MEM_UTIL=0.6
REWARD_TENSOR_MP_SIZE=1
REWARD_TEMPERATURE=0
REWARD_TOP_P=1.0
REWARD_TOP_K=-1
REWARD_SCORE_SCALE=0.01

# Reward Control
REWARD_MANAGER=naive
REWARD_ENABLE=True
REWARD_STRATEGY=GenRM
REWARD_FREE_CACHE_ENGINE=True
REWARD_CUSTOM_PROCESSOR_PATH=reward_utils/rm_lib.py
REWARD_CUSTOM_PROCESSOR_NAME=GroupRewardModelProcessor
CUSTOM_REWARD_FN_PATH=reward_utils/rm_lib.py
CUSTOM_REWARD_FN_NAME=score_reward_fn
REWARD_KEEP_GROUP=True # dispatch the rollout candidates by group, must be `True` for GRRM
RESPONSE_EXTRACTOR_TYPE=codeblock
OVERLONG_BUFFER_ENABLE=True
OVERLONG_BUFFER_LEN=2048
OVERLONG_BUFFER_PENALTY_FACTOR=0.04
DEFAULT_REWARD=-${OVERLONG_BUFFER_PENALTY_FACTOR} # default reward if we fail to get a valid reward

# Training/Logging
PROJECT_NAME=cot_mt_verl
EXPERIMENT_NAME=cot_mt_gspo.exp
N_GPUS_PER_NODE=8
VAL_BEFORE_TRAIN=False
NNODES=2
SAVE_FREQ=1000
TEST_FREQ=4000
RESUME_MODE=auto
TOTAL_EPOCHS=2
DEFAULT_LOCAL_DIR=path/to/cot_mt_gspo
```
</details>

**Run Training:**

Run the following command to start training:

<details><summary>verl Launch Command</summary>

```bash
ray job submit \
    --runtime-env=verl/trainer/runtime_env.yaml \
    --no-wait \
    -- \
    python3 -m \
    verl.trainer.main_ppo \
    data.train_files=${TRAIN_FILES} \
    data.val_files=${VAL_FILES} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.val_batch_size=${VAL_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.filter_overlong_prompts=True \
    data.shuffle=True \
    data.truncation=error \
    actor_rollout_ref.model.path=${ACTOR_MODEL_PATH} \
    actor_rollout_ref.actor.shuffle=True \
    actor_rollout_ref.actor.optim.lr=${ACTOR_OPT_LR} \
    actor_rollout_ref.actor.optim.lr_scheduler_type=${ACTOR_LR_SCHEDULER_TYPE} \
    actor_rollout_ref.actor.optim.min_lr_ratio=${ACTOR_MIN_LR_RATIO} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=${ACTOR_WARMUP_STEPS} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ACTOR_PPO_MINI_BSZ} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ACTOR_PPO_MICRO_BSZ_PER_GPU} \
    actor_rollout_ref.actor.policy_loss.loss_mode=${ACTOR_LOSS_MODE} \
    +actor_rollout_ref.actor.sft_coef=${ACTOR_SFT_COEF} \
    actor_rollout_ref.actor.loss_agg_mode=${ACTOR_LOSS_AGG_MODE} \
    actor_rollout_ref.actor.use_kl_loss=${ACTOR_USE_KL_LOSS} \
    actor_rollout_ref.actor.kl_loss_type=mse \
    actor_rollout_ref.actor.kl_loss_coef=${KL_COEF} \
    actor_rollout_ref.actor.clip_ratio_low=${ACTOR_CLIP_RATIO_LOW} \
    actor_rollout_ref.actor.clip_ratio_high=${ACTOR_CLIP_RATIO_HIGH} \
    actor_rollout_ref.actor.strategy=${ACTOR_STRATEGY} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=${ACTOR_MODEL_DTYPE} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${ACTOR_ULYSSES_SP_SIZE} \
    actor_rollout_ref.rollout.name=${ROLLOUT_NAME} \
    actor_rollout_ref.rollout.mode=${ROLLOUT_MODE} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${ROLLOUT_LOG_PROB_MB_PER_GPU} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TENSOR_MP_SIZE} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL} \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE} \
    actor_rollout_ref.rollout.top_p=${ROLLOUT_TOP_P} \
    actor_rollout_ref.rollout.top_k=${ROLLOUT_TOP_K} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${VAL_TOP_K} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=${VAL_DO_SAMPLE} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=64 \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    critic.optim.lr=1e-5 \
    critic.model.path=null \
    critic.ppo_micro_batch_size_per_gpu=4 \
    algorithm.adv_estimator=${ADV_ESTIMATOR} \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=${NORM_ADV_BY_STD_IN_GRPO} \
    reward_model.reward_manager=${REWARD_MANAGER} \
    reward_model.enable=${REWARD_ENABLE} \
    reward_model.strategy=${REWARD_STRATEGY} \
    +reward_model.keep_group=${REWARD_KEEP_GROUP} \
    +reward_model.rollout.free_cache_engine=${REWARD_FREE_CACHE_ENGINE} \
    +reward_model.rollout.name=${REWARD_ROLLOUT_NAME} \
    +reward_model.rollout.mode=${REWARD_ROLLOUT_MODE} \
    +reward_model.rollout.gpu_memory_utilization=${REWARD_GPU_MEM_UTIL} \
    +reward_model.rollout.tensor_model_parallel_size=${REWARD_TENSOR_MP_SIZE} \
    +reward_model.rollout.max_num_batched_tokens=${REWARD_MAX_NUM_BATCHED_TOKENS} \
    +reward_model.rollout.temperature=${REWARD_TEMPERATURE} \
    +reward_model.rollout.top_p=${REWARD_TOP_P} \
    +reward_model.rollout.top_k=${REWARD_TOP_K} \
    +reward_model.rollout.response_length=${REWARD_RESPONSE_LENGTH} \
    +reward_model.prompt_length=${REWARD_PROMPT_LENGTH} \
    +reward_model.score_scale_factor=${REWARD_SCORE_SCALE} \
    +reward_model.default_reward=${DEFAULT_REWARD} \
    reward_model.model.path=${REWARD_MODEL_PATH} \
    +reward_model.custom_processor.path=${REWARD_CUSTOM_PROCESSOR_PATH} \
    +reward_model.custom_processor.name=${REWARD_CUSTOM_PROCESSOR_NAME} \
    +reward_model.custom_processor.extractor_type=${RESPONSE_EXTRACTOR_TYPE} \
    +reward_model.custom_processor.overlong_buffer.enable=${OVERLONG_BUFFER_ENABLE} \
    +reward_model.custom_processor.overlong_buffer.max_resp_len=${MAX_RESPONSE_LENGTH} \
    +reward_model.custom_processor.overlong_buffer.len=${OVERLONG_BUFFER_LEN} \
    +reward_model.custom_processor.overlong_buffer.penalty_factor=${OVERLONG_BUFFER_PENALTY_FACTOR} \
    custom_reward_function.path=${CUSTOM_REWARD_FN_PATH} \
    custom_reward_function.name=${CUSTOM_REWARD_FN_NAME} \
    trainer.logger=[console,wandb] \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.val_before_train=${VAL_BEFORE_TRAIN} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.resume_mode=${RESUME_MODE} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.default_local_dir=${DEFAULT_LOCAL_DIR}
```

</details>
</details>



## Evaluation

Work in progress.

## Citation

```bibtex
@misc{yang2026grrmgrouprelativereward,
      title={GRRM: Group Relative Reward Modeling for Machine Translation}, 
      author={Sen Yang and Shanbo Cheng and Lu Xu and Jianbing Zhang and Shujian Huang},
      year={2026},
      eprint={2602.14028},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.14028},
}
```