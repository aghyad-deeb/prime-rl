uv run orchestrator @ orch.toml \
    --model.name "Qwen/Qwen3-32b" \
    --log.level debug \
    --client.base-url http://$INFERENCE_SERVER_IP:8000/v1 \
    --client.api-key-var INFERENCE_SERVER_API_KEY \
    --output-dir $OUTPUT_DIR \

