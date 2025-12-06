uv run torchrun \
    --nproc-per-node 8 \
    --local-ranks-filter 0 \
    src/prime_rl/trainer/rl/train.py  @ trainer.toml \
        --output-dir $OUTPUT_DIR \
        --log.level debug \
