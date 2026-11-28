# syntax=docker/dockerfile:1.7
# Production Docker image for zstep on T-Head PPU-ZW810E (SAIL 1.7 / HGGC 12.3)
# Compatible with Nebula (`nebula-node --zstep-docker-image <image>`).
#
# Build:
#   docker build -f deploy/docker/Dockerfile.sail \
#     --build-arg BASE_IMAGE=<your-sail-1.7-image> \
#     --build-arg POLICY_FILE=docs/benchmarks/sail_zw810e_dsv41_approved_policy.json \
#     -t zstep:sail-zw810e .

ARG RUST_IMAGE=rust:1.85-bookworm
ARG BASE_IMAGE=ppu-sail:1.7-py312
ARG POLICY_FILE=docs/benchmarks/sail_zw810e_dsv41_approved_policy.json

FROM ${RUST_IMAGE} AS rust-builder
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
RUN cargo build --release -p zs-server && \
    strip target/release/zs-server

FROM ${BASE_IMAGE} AS runtime
ARG POLICY_FILE

COPY --from=rust-builder /src/target/release/zs-server /usr/local/bin/zs-server
COPY worker /opt/zstep/worker
COPY ${POLICY_FILE} /etc/zstep/sail_approved_policy.json

# Install zstep_worker into the container Python environment without replacing
# the vendor SAIL PyTorch / triton-for-sail / HGGC stack from BASE_IMAGE.
RUN pip install --no-cache-dir --no-deps --no-build-isolation /opt/zstep/worker

# Bake Evolith-approved SAIL policy as default; explicit ZSTEP_SAIL_* env vars
# passed at runtime by Nebula always take precedence over ZSTEP_SAIL_POLICY_PATH.
ENV ZSTEP_BACKEND=sail \
    ZSTEP_WORKER_ROOT=/opt/zstep/worker \
    ZSTEP_SAIL_POLICY_PATH=/etc/zstep/sail_approved_policy.json \
    HF_HOME=/model/.cache/huggingface \
    XDG_CACHE_HOME=/model/.cache \
    TRITON_CACHE_DIR=/model/.cache/triton \
    TORCH_EXTENSIONS_DIR=/model/.cache/torch_extensions

WORKDIR /opt/zstep
ENTRYPOINT ["zs-server"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
