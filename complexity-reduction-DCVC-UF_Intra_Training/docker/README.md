# Training DCVC-UF-Intra in Docker

A self-contained environment for `train_image_allqp_OpenImage.sh` (and the other
`train_image*.sh` recipes). Nothing on the host is needed beyond Docker, the
NVIDIA Container Toolkit, and the datasets.

## Layout

The training scripts reach the datasets as `../datasets/...`, so both host
directories are bind-mounted as siblings under `/workspace`, mirroring how they
already sit next to each other in the home directory:

| host          | container              | holds |
| ------------- | ---------------------- | ----- |
| `~/DCVC-UF`   | `/workspace/DCVC-UF`   | this repo (working dir), `checkpoints/` |
| `~/datasets`  | `/workspace/datasets`  | `Open_Image/train`, `jpegai_validation_set` |

Everything else a run needs is inside the repo: `SAVE_DIR`
(`checkpoints/image_model_allqp_openimage`), the reference checkpoint
(`checkpoints/cvpr2026_image.pth.tar`) and the reference RD cache
(`checkpoints/ref_rd_cache/`). So `TRAIN_DATASET=../datasets/Open_Image/train`
and `REF_CKPT=checkpoints/cvpr2026_image.pth.tar` resolve inside the container
exactly as they do on the host — the shell scripts need no Docker-specific edits.

### The reference RD cache is environment-bound

`train_image.py` keys the reference RD cache — and the `ckpt_best_<sig>.pth.tar`
signature — on the **absolute** path, size and mtime of the checkpoint and the
validation list (`_file_identity`), so one cache file is only valid for one
execution environment.

`checkpoints/ref_rd_cache/ref_rd_points_jpegai10.json` has already been migrated
to this layout: the checkpoint it names is
`/workspace/DCVC-UF/checkpoints/cvpr2026_image.pth.tar`, the mtime of the repo
copy was restored with `touch -r` from the original, and the two files are
byte-identical, so the cached RD points describe exactly this checkpoint.
`compute_reference_rd()` returns `cached` for it — the official model is not
re-evaluated.

Running the same recipe **outside** the container (plain `python train_image.py`
on the host) sees `/home/<user>/DCVC-UF/...` instead, misses, re-evaluates the
official model once (10 images × 5 QPs) and rewrites the cache with host paths —
which then makes the container miss. Pick one environment per cache file, or
give the host runs their own `REF_CACHE` path.

The repository is **not** copied into the image, only mounted, so editing
`train_image.py` or a `.sh` recipe takes effect on the next run with no rebuild.

## Quick start

```bash
cp .env.example .env      # optional: WANDB_API_KEY, GPU index, uid/gid
docker compose build      # ~5 min, mostly the torch cu128 wheel
docker compose up train   # runs train_image_allqp_OpenImage.sh
```

Detached, with the log on disk:

```bash
docker compose up -d train
docker compose logs -f train
```

Stop with `docker compose down`. `train_image.py` resumes from the newest
checkpoint in `SAVE_DIR` on the next start, so an interrupted run continues
where it left off.

## `description.json` is generated automatically

`src/datasets/image_dataset.py` reads `<dataset root>/description.json`, a flat
JSON array of image paths relative to that root. The datasets are bind-mounted,
so the file cannot be baked into the image; `docker/entrypoint.sh` calls
`docker/prepare_datasets.sh` on **every container start** instead. It is
idempotent — a root that already has `description.json` is skipped.

Two kinds of index are generated.

**Full-root indexes** (`DATASET_DESCRIPTIONS`) — one `description.json` per
dataset root, covering every image in it:

| dataset root                                | ext    | entries |
| ------------------------------------------- | ------ | ------- |
| `/workspace/datasets/Open_Image/train`      | `.jpg` | 384795  |
| `/workspace/datasets/jpegai_validation_set` | `.png` | 350     |

This fork ships no test/evaluation entry point, so the test sets (`jpegai_test_set`,
`clic_test_set`, `kodak_test_set`) are deliberately not indexed — add them to
`DATASET_DESCRIPTIONS` if you bring your own evaluation code.

**Subset indexes** (`DATASET_LIST_JSON`) — a `description.json` describes a whole
root, so a subset needs its own file. Each listed `<name>.txt` is mirrored to
`<name>.json` in place, keeping the line order:

| source                              | generated                            | entries |
| ----------------------------------- | ------------------------------------ | ------- |
| `jpegai_validation_set_10.txt`      | `jpegai_validation_set_10.json`      | 10      |

So the validation set carries both: `description.json` with all 350 images, and
`jpegai_validation_set_10.json` with the 10 the training recipe validates on.
`VAL_LIST` in `train_image_allqp_OpenImage.sh` points at the 10-image one — swap
it for `description.json` to validate on the full 350 (much slower, and it
changes the reference-cache identity, see below).

`make_description.py` walks a root with `os.walk`, so Open Images'
`train_0/1/2` subdirectories are indexed without flattening anything. The first
scan of ~385k files over a bind mount takes about a minute. Its `--list` mode is
what produces the subset JSON:

```bash
python make_description.py ../datasets/jpegai_validation_set \
    --list ../datasets/jpegai_validation_set/jpegai_validation_set_10.txt
```

Rebuild the indexes after adding or removing images:

```bash
FORCE_DESCRIPTION=1 docker compose run --rm prepare
```

A different dataset (JPEG-AI crops, for example), or another subset list:

```bash
DATASET_DESCRIPTIONS=/workspace/datasets/jpegai_training_random_crop:.png \
  docker compose run --rm prepare

DATASET_LIST_JSON=/workspace/datasets/jpegai_validation_set/jpegai_validation_set.txt \
  docker compose run --rm prepare
```

## GPUs

`src/utils/common.py:start_train()` spawns **one DDP process per visible GPU**.
`CUDA_VISIBLE_DEVICES` is therefore pinned to a single index (`DCVC_GPUS`,
default `0`) — otherwise `train_image_allqp_OpenImage.sh` silently becomes a
multi-GPU run with `batch_size 16` *per rank*, which is not the documented
recipe.

```bash
DCVC_GPUS=2 docker compose up train     # train on physical GPU 2
```

The chosen card is `cuda:0` inside the container. Two concurrent single-GPU
experiments are two `docker compose run` invocations with different `DCVC_GPUS`
and different `SAVE_DIR`s.

## Other knobs

| variable | default | meaning |
| -------- | ------- | ------- |
| `TRAIN_SCRIPT` | `train_image_allqp_OpenImage.sh` | what `up train` runs |
| `DCVC_GPUS` | `0` | `CUDA_VISIBLE_DEVICES` |
| `WANDB_API_KEY` | empty | monitoring; see below |
| `DOCKER_UID` / `DOCKER_GID` | `1006` | uid/gid the container runs as |
| `OMP_NUM_THREADS` | `8` | per-process thread cap |
| `FORCE_DESCRIPTION` | `0` | `1` rebuilds `description.json` |
| `PREPARE_DATASETS` | `1` | `0` skips dataset indexing at start |

Put them in `.env` (gitignored) or prefix the command.

`train_image_allqp_OpenImage.sh` is currently the only recipe in the repo; point
`TRAIN_SCRIPT` at another one when you add it:

```bash
TRAIN_SCRIPT=train_image_allqp_jpegai.sh docker compose up train
```

## wandb

`train_image.py` calls `wandb.init(mode='online')` unless the script passes
`--wandb_mode disabled`, and an explicit `mode=` argument overrides `$WANDB_MODE`
— so the way to run unmonitored is the flag, not the environment variable.
Without `WANDB_API_KEY` the init simply fails, `train_image.py` logs a warning
and trains on. The `train` service runs without a TTY so that failure is
immediate rather than a login prompt waiting for input.

## The environment

`python 3.10 + torch 2.10.0+cu128 + triton 3.6.0` on
`nvidia/cuda:12.8.1-devel-ubuntu22.04` — the stack the throughput table in
`train_image_allqp_OpenImage.sh` was measured on. `--compile full
--compile_mode max-autotune` is the part that is sensitive to these versions;
bump the `TORCH_VERSION` / `PYTHON_VERSION` build args only deliberately.

The CUDA toolkit comes from the base image only for `nvcc` and the C++ compiler
inductor shells out to — the torch wheel ships its own cudart/cudnn/cublas, and
the driver is injected by the NVIDIA Container Toolkit.

`$HOME` (`/home/trainer`) is a named volume, so the triton/inductor autotune
caches survive restarts; only the first start pays the max-autotune compile
(twice: once for the 256×256 graph, again for 512×512 after epoch 90). To reset
them: `docker volume rm dcvc-uf_trainer-home`.

## Entropy-coding extensions

Not built, and not needed for training or validation: `MLCodec_extensions_cpp`
and `inference_extensions_cuda` are imported lazily and only on the
`compress()`/`decompress()` paths. Validation measures bitrate from the entropy
model, not from a real bitstream. If you need actual bitstreams (`test_image_rd.py`),
build them inside the container — the toolchain is there:

```bash
docker compose run --rm shell
git clone https://github.com/NVIDIA/cutlass third_party/cutlass
cd third_party/cutlass && git checkout v4.4.1 && cd ../..
cd src/cpp && bash install.sh
cd ../layers/extensions/inference && bash install.sh
```

Run as root (`docker compose run --rm --user root shell`) if the build wants to
install into site-packages.

## Troubleshooting

**DataLoader dies with "bus error"** — shared memory. `ipc: host` is already set
in `docker-compose.yml`; if you launch with plain `docker run`, add
`--ipc=host` or `--shm-size=16g`.

**Files owned by root on the host** — `DOCKER_UID`/`DOCKER_GID` do not match the
owner of `~/datasets` and `~/DCVC-UF`. Check with `id` and set them in `.env`;
rebuild so `$HOME` inside the image is chowned to match.

**`nvidia.com/gpu=all: unresolvable CDI device`** — the NVIDIA Container Toolkit
CDI spec is missing or stale. Regenerate it:
`sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`.
