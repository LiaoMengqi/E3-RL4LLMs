TEST_DATADIR=./dataset/test.json
MODELDIR=..checkpoint/grpo_1_5b/global_step_480/actor/huggingface
OUTPUT_FILE=./.results/dapo_7b_s16_1.json

python3 -m adre.main_generate \
    --model=$MODELDIR \
    --input_file=$TEST_DATADIR \
    --output_file=$OUTPUT_FILE \
    --tensor_parallel_size=1 \
    --gpu_memory_utilization=0.95 \
    --temperature=1 \
    --max_tokens=6144 \
    --n_samples=16
