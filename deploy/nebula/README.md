# Exporting zstep Docker Images to Nebula

`zstep` ships self-contained Docker images (`deploy/docker/`) that bundle `zs-server`,
the installed `zstep_worker` package (including SM120/SAIL C++/CUDA sources), and
optional Evolith-approved policy defaults so `nebula-node` does not depend on any
host source working directory (`--zstep-worker-root`).

## 1. Build a Target Image

### NVIDIA SM120 (RTX 5090 / RTX 6000D)

```bash
docker build -f deploy/docker/Dockerfile.sm120 -t zstep:sm120 .
```

### T-Head PPU-ZW810E (SAIL 1.7 / HGGC 12.3)

```bash
docker build -f deploy/docker/Dockerfile.sail \
  --build-arg BASE_IMAGE=<your-sail-1.7-image> \
  --build-arg POLICY_FILE=docs/benchmarks/sail_zw810e_dsv41_approved_policy.json \
  -t zstep:sail-zw810e-gen3 .
```

The SAIL image bakes `POLICY_FILE` into `/etc/zstep/sail_approved_policy.json` and
sets `ZSTEP_SAIL_POLICY_PATH`. Any explicit `ZSTEP_SAIL_*` environment variables
passed at container runtime by Nebula override the baked defaults.

## 2. Export a Validated Nebula Bundle

```bash
python3 scripts/export_nebula_bundle.py \
  --backend sail \
  --image zstep:sail-zw810e-gen3 \
  --model-uid deepseek-v4.1-flash \
  --model-dir /model/DeepSeek-V4.1-Flash \
  --policy docs/benchmarks/sail_zw810e_dsv41_approved_policy.json \
  --world-size 8 \
  --speculative-method dspark \
  --num-speculative-tokens 5 \
  --output dist/nebula-dsv41-sail
```

This validates the policy JSON against `zstep_worker.env_config.WorkerRuntimeConfig`
and emits:
- `policy.env` — `ZSTEP_*` overrides
- `nebula-deployment.json` — ready for `nebula-gateway` (`POST /models/load`)
- `docker-build.sh` — reproducible image build command

## 3. Run Under `nebula-node`

```bash
nebula-node \
  --zstep-docker-image zstep:sail-zw810e-gen3 \
  --zstep-model-dir /data/models
```

`nebula-node` mounts `--zstep-model-dir` to `/model`, configures `--ipc=host`,
persists kernel caches under `/model/.cache/*`, and starts `/usr/local/bin/zs-server serve`.
