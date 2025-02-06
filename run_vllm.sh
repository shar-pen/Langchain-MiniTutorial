export CUDA_VISIBLE_DEVICES=5

modelpath=../DataCollection/officials/Qwen2.5-7B-Instruct
modelname=Qwen2.5-7B-Instruct

nohup python -m vllm.entrypoints.openai.api_server \
    --model $modelpath \
    --served-model-name $modelname \
    --port 5551 \
    --gpu-memory-utilization 0.4 \
    --dtype=half \
    > run_vllm.log 2>&1 &