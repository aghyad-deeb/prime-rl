uv run inference @ infer.toml \
    --api-key $INFERENCE_SERVER_API_KEY
    --data-parallel-size 12 \
    --tensor-parallel-size 2 \
    --data-parallel-size-local 4 \
    --data-parallel-address $DATA_PARALLEL_ADDRESS \
    --data-parallel-rpc-port $DATA_PARALLEL_RPC_PORT \
