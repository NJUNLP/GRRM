
# GRRM Training Pipeline

## SFT Cold-start

### Data Preprocessing

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
### Training

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
## model
model_name_or_path: Qwen/Qwen2.5-7B
flash_attn: fa2

## method
stage: sft
do_train: true
finetuning_type: full
deepspeed: examples/deepspeed/ds_z3_config.json

## dataset
dataset: sft_towerblocks_mt_ranking
template: chatml
cutoff_len: 12000
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4

## output
output_dir: saves/qwen-7b/full/grrm_sft
logging_steps: 1
save_steps: 500000
plot_loss: true
overwrite_output_dir: true
report_to: wandb  # choices: [none, wandb, tensorboard, swanlab, mlflow]

## train
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


## RLVR Training

We employ [verl with Generative Reward Models support](https://github.com/DoubleVII/verl/tree/gen_rm) for RLVR training. Our **Ranking Accuracy Reward** is defined in [this function](https://github.com/DoubleVII/verl/blob/b99cd9587c2f8aacae84df90e7fcbb9a1f33eb81/reward_utils/ranking_score_reward.py#L307).

### Data Preprocessing

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

### Training

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


