#!/bin/bash
set -x

NUM_EPISODES=3
n_samples_per_prompt=8
n_rollout_max=12
n_rollout_min=4

LR_ACTOR=5e-6
entropy_coeff=0


n_rollout_update=2
enable_temperature_scheduler=True
enable_annealing=True

TRAIN_DATADIR=./dataset/train_data_10k.parquet
VAL_DATADIR=./dataset/valid_data.parquet
MODELDIR=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B


PRETRAIN_DIR=$MODELDIR
SAVE_DIR=../checkpoint/e3_1_5b/
TENSORBOARD_PATH=$SAVE_DIR/tensorboard

export TENSORBOARD_DIR=$TENSORBOARD_PATH
export HYDRA_FULL_ERROR=1

python3 -m e3.main_e3 \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_DATADIR \
    data.val_files=$VAL_DATADIR \
    data.train_batch_size=64 \
    data.max_prompt_length=2048 \
    data.max_response_length=6144 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODELDIR \
    actor_rollout_ref.actor.optim.lr=$LR_ACTOR \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=512 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=$entropy_coeff \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=$n_samples_per_prompt \
    actor_rollout_ref.rollout.n_low=$n_rollout_min \
    actor_rollout_ref.rollout.n_high=$n_rollout_max \
    actor_rollout_ref.rollout.n_update=$n_rollout_update \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.rollout.enable_temperature_scheduler=$enable_temperature_scheduler \
    actor_rollout_ref.rollout.enable_annealing=$enable_annealing \
    actor_rollout_ref.rollout.max_steps=480 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name='GRPO' \
    trainer.experiment_name='Qwen' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.default_local_dir=$SAVE_DIR \
    trainer.save_freq=80 \
    trainer.test_freq=10 \
    trainer.total_epochs=$NUM_EPISODES $@
