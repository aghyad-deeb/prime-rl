uv run inference @ infer.toml \
	--model.name "Qwen/Qwen3-32b" \
	--data-parallel-size 6 \
	--tensor-parallel-size 4 \
	--data-parallel-size-local 2 \
	--data-parallel-address $DATA_PARALLEL_ADDRESS \
	--data-parallel-rpc-port $DATA_PARALLEL_RPC_PORT \
	--data-parallel-start-rank 4 \
	--api-key $INFERENCE_SERVER_API_KEY \
	--headless
