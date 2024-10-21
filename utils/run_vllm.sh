set -o errexit

# huggingface model id or path
MODEL_ID=meta-llama/Meta-Llama-3.1-70B-Instruct
# MODEL_ID=mistralai/Mixtral-8x7B-Instruct-v0.1
# number of GPUs needed
NUM_GPUS=4
set -o nounset

# get a random valid port
PORT=$(python -c "import socket;sock = socket.socket();sock.bind(('', 0));print(sock.getsockname()[1])")

# print the endpoint information to the log
echo "starting"
echo "http://${HOSTNAME}.pok.ibm.com:${PORT}/v1"

python -u -m vllm.entrypoints.openai.api_server \
    --distributed-executor-backend mp \
    --disable-custom-all-reduce \
    --host 0.0.0.0 \
    --port $PORT \
    --model $MODEL_ID \
    --tensor-parallel-size $NUM_GPUS \
    --load-format safetensors \
    --gpu-memory-utilization 0.9 --max-model-len 8196