# Running on a Tesla V100 (sm70)

The V100 works, but the repo's pinned stack does not run on it out of the box.
Five fixes are needed. Budget roughly 2.5x to 3.5x the wall-clock of an A100;
the gap is raw throughput and VRAM, not missing kernels.

## 1. PyTorch version: the pinned torch drops Volta

`uv.lock` pins `torch==2.11.0+cu128`. Those wheels are built for sm_75/80/86/90/100/120
with no sm_70, so on a V100 every CUDA op fails with
`no kernel image is available for execution on the device`.

The newest torch that still ships Volta kernels is 2.7.x. Because this diverges from
the lockfile, use a dedicated venv rather than `uv sync`:

```bash
uv venv .venv-v100 --python 3.12
uv pip install --python .venv-v100/bin/python -e ".[b601,vjepa,wandb]"
uv pip install --python .venv-v100/bin/python \
  "torch==2.7.1+cu126" "torchvision==0.22.1+cu126" "torchcodec<0.6" \
  --index-url https://download.pytorch.org/whl/cu126
```

Install torch in its own step with `--index-url`, and pin the `+cu126` local version.
With `--extra-index-url` PyPI stays a candidate and uv resolves `torch==2.7.1` to the
**cu128** wheel, whose arch list also starts at sm_75 -- the same failure as the pinned
2.11.0, from a wheel whose version looks correct. A bare `2.7.1` pin will not reinstall
over an already-present cu128 build either; the version strings match, so uv reports
"Checked 3 packages" and changes nothing. Verify rather than assume:

```bash
.venv-v100/bin/python -c "import torch; print(torch.cuda.get_arch_list())"
```

`sm_70` must appear in that list. The `vastai/pytorch:2.7.1-cu128-*` images ship a
Volta-incapable torch despite the 2.7.1 in their tag.

## 2. torchcodec native library

torchcodec 0.5's FFmpeg shims need `libnppicc.so.12` (NVIDIA Performance Primitives),
which the torch cu126 wheel does not bundle. The dataloader dies at startup with
`Could not load libtorchcodec ... libnppicc.so.12: cannot open shared object file`.

```bash
uv pip install --python .venv-v100/bin/python nvidia-npp-cu12 nvidia-nvjpeg-cu12
export LD_LIBRARY_PATH=$(ls -d .venv-v100/lib/python3.12/site-packages/nvidia/*/lib | tr '\n' ':')$LD_LIBRARY_PATH
```

Put the `export` in the run script or shell rc so every process sees it.

## 3. Precision: fp16, never bf16

The V100 has no bf16 tensor cores. Under `mixed_precision: bf16`:

- `F.scaled_dot_product_attention` falls back to the MATH backend, materializing the
  full `B*H*N*N` score matrix with no memory-efficient kernel.
- every ViT-B and DiT GEMM runs without tensor-core acceleration. Measured about 5x
  slower than fp16, and slower than plain fp32.

`fp16` is the repo default (`configs/experiment/base.yaml`), so nothing needs passing;
only an explicit `experiment.mixed_precision=bf16` reintroduces the slowdown. Accelerate
adds the grad scaler automatically. Watch the first few hundred steps for fp16 overflow while the loss
scale settles.

## 4. Gradient checkpointing must stay on

For unfrozen-backbone arms (e.g. a flow WAM with a trainable ViT-B), `experiment.gradient_checkpointing=false`
OOMs at micro-batch 2 on 16 GB. Such arms set it `true` in their run config; do not override it.
The `proj_*` encoders use `norm: batch`, so micro-batch 1 (the only size that fits
without checkpointing) is degenerate anyway.

## 5. C compiler for torch.compile

The Vast `pytorch` images ship no C compiler, so inductor fails with
`Failed to find C compiler`.

```bash
apt-get update && apt-get install -y build-essential
export CC=gcc CXX=g++
```

FlexAttention still will not run on sm70 even with a compiler: the Triton codegen
crashes (`LLVM ERROR: Unsupported rounding mode for conversion`, or
`Failed to compute parent layout for slice layout`). `_resolve_attn_backend` already
routes `auto` to `sdpa` below sm80, so leave `algorithm.attn_backend` at its default.

## Also

- Accelerate warns `Detected kernel version 5.4.0, which is below the recommended
  minimum of 5.5.0`. Harmless on the runs tested, but if the process hangs at startup
  this is the first suspect.

## Expected performance

A ViT-B flow WAM arm on b601 data, 16 GB V100, fp16, measured via `batch_size=auto`:

| | unfrozen ViT-B | frozen ViT-B |
|---|---|---|
| max micro-batch | ~114 (grad-ckpt on) | 256 |
| compute per 256-sample step | ~13.6 s | ~4.2 s |
| full 20k-step run | ~4 days | ~1 day |

## pi0.5 inference

Serving works on a 16 GB V100 once fix 1 is applied. `pi05` (4.14B params, the
`lerobot/pi05_base` backbone) at batch 1, one 256x256 frame per camera:

| | RTX 5090 | V100 16 GB |
|---|---|---|
| action chunk latency | 297 ms | 790 ms |
| peak VRAM | 9.5 GB | 9.5 GB |

2.7x slower, but it fits with 6 GB to spare.

## One-shot run command

```bash
cd /path/to/thesis
export LD_LIBRARY_PATH=$(ls -d .venv-v100/lib/python3.12/site-packages/nvidia/*/lib | tr '\n' ':')$LD_LIBRARY_PATH
export CC=gcc CXX=g++
.venv-v100/bin/python main.py run=<arm> \
  experiment.num_workers=8 \
  wandb.mode=online wandb.entity=<entity>
```

## OCBench (MJWarp)

`warp-lang` on PyPI is a CUDA 13 build, and CUDA 13 dropped Volta: the first MJWarp kernel
launch fails with `CUDA 13.x requires sm_75 or higher`. Install the `+cu12` wheel from the
Warp GitHub release over it, matching the version the `ocbench` extra resolved:

```bash
uv pip install --python .venv-v100/bin/python -e ".[ocbench]"
uv pip install --python .venv-v100/bin/python --reinstall-package warp-lang \
  "warp-lang @ https://github.com/NVIDIA/warp/releases/download/v1.18.0/warp_lang-1.18.0%2Bcu12-py3-none-manylinux_2_28_x86_64.whl"
```

The first rollout compiles Warp's kernels for sm_70, about 6 minutes; they are cached in
`~/.cache/warp` after that. On eryk-pc the V100 is `CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=1`.
