<div align="center">

# DCVC-UF-Intra — Training Fork

**CVPR2026**

</div>

[![page](https://img.shields.io/badge/Project-Page-blue?logo=github)](https://github.com/microsoft/DCVC)
[![arXiv](https://img.shields.io/badge/arXiv-2606.04410-b31b1b.svg)](https://arxiv.org/abs/2606.04410)

A training-focused fork of the official [DCVC-UF](https://github.com/microsoft/DCVC) repository, narrowed
down to **one job: training the intra (image) model, DCVC-UF-Intra, inside Docker**.

Relative to upstream, the bitstream test harness, the BD-rate tooling, the video training entry point
and the DCVC-family documentation have been removed. What is left is `train_image.py`, the model and
dataset code it needs, and a container that runs it. If you want the full codec — real bitstreams,
video models, benchmark scripts — use the upstream release linked from the
[project page](https://github.com/microsoft/DCVC) instead.

---

## Contents

1. [Directory layout](#1-directory-layout)
2. [Build the Docker image](#2-build-the-docker-image)
3. [Prepare the data](#3-prepare-the-data)
4. [Start training](#4-start-training)
5. [What a run produces](#5-what-a-run-produces)

---

## 1. Directory layout

The training scripts reach their datasets with paths relative to the repository (`../datasets/...`),
so **the repository and the dataset folder must be siblings**:

```
/home/<user>/
├── DCVC-UF/                       # this repository
│   ├── train_image.py
│   ├── train_image_allqp_OpenImage.sh
│   ├── make_description.py
│   ├── docker/
│   ├── docker-compose.yml
│   ├── checkpoints/               # reference model, and everything a run writes
│   └── src/
└── datasets/
    ├── Open_Image/train/          # training set:   .jpg, in train_0/1/2 subfolders
    └── jpegai_validation_set/     # validation set: .png
```

Docker mirrors this one-to-one, so nothing in the shell recipes needs a container-specific edit:

| host                  | container             |
| --------------------- | --------------------- |
| `~/DCVC-UF`           | `/workspace/DCVC-UF`  |
| `~/datasets`          | `/workspace/datasets` |

Only these two directories are mounted. Checkpoints, the reference model and the reference RD cache
all live under this repo's own `checkpoints/`.

### Getting the datasets

**Training — Open Images.** Get it from the official
[CVDF distribution](https://github.com/cvdfoundation/open-images-dataset), which ships the train
split as 16 shards named `train_0` … `train_f`. This setup uses three of them, unpacked side by side
under `datasets/Open_Image/train/`:

| shard     |  images |
| --------- | ------: |
| `train_0` | 156,541 |
| `train_1` | 116,687 |
| `train_2` | 111,567 |
| **total** | **384,795** |

Nothing depends on that particular count — the shards are just directories, `make_description.py`
walks them recursively, and adding or dropping one only means re-running the indexer
(`FORCE_DESCRIPTION=1 docker compose run --rm prepare`). More data is generally better here; three
shards is simply what this recipe was tuned on.

**Validation — JPEG-AI validation set.** 350 PNGs under `datasets/jpegai_validation_set/`. The recipe
validates on the first 10 of them (see the subset index in
[section 3](#3-prepare-the-data)), which keeps a validation pass to well under a minute; point
`VAL_LIST` at `description.json` instead to use all 350.

### The reference checkpoint

Validation plots the official pretrained model's RD curve next to yours. Download
[`cvpr2026_image.pth.tar`](https://1drv.ms/f/c/2866592d5c55df8c/IgAalzb_985lQ79GkXyW2P5OASPpZHHcrcGWEVQxO-mQCVg?e=qyvMN6)
([backup](https://1drv.ms/f/c/2866592d5c55df8c/EozfVVwtWWYggCitBAAAAAABbT4z2Z10fMXISnan72UtSA?e=BID7DA))
into `checkpoints/`:

```
checkpoints/cvpr2026_image.pth.tar
```

This is optional — drop `--ref_ckpt` from the recipe and training runs without the reference curve.

---

## 2. Build the Docker image

Requires Docker with the NVIDIA Container Toolkit, and an NVIDIA GPU. Nothing else has to be
installed on the host: no conda, no CUDA toolkit, no PyTorch.

```bash
cd ~/DCVC-UF
cp .env.example .env      # optional: wandb key, GPU index, uid/gid
docker compose build      # ~5 min, mostly the torch cu128 wheel
```

The image pins `python 3.10 + torch 2.10.0+cu128 + triton 3.6.0` on
`nvidia/cuda:12.8.1-devel-ubuntu22.04` — the stack the throughput numbers in the training recipe were
measured on. `torch.compile --compile_mode max-autotune` is what is sensitive to these versions.

The repository itself is **not** copied into the image, only bind-mounted, so editing `train_image.py`
or a recipe takes effect on the next run with no rebuild. Details and troubleshooting:
[docker/README.md](docker/README.md).

---

## 3. Prepare the data

`src/datasets/image_dataset.py` reads a `description.json` from each dataset root — a flat JSON array
of image paths relative to that root:

```json
[
    "train_0/000002b66c9c498e.jpg",
    "train_0/000002b97e5471a0.jpg",
    "..."
]
```

**You do not have to create these by hand.** `docker/prepare_datasets.sh` runs on every container
start and generates whatever is missing (existing files are left alone):

| dataset root                     | ext    | generated                        |
| -------------------------------- | ------ | -------------------------------- |
| `datasets/Open_Image/train`      | `.jpg` | `description.json`               |
| `datasets/jpegai_validation_set` | `.png` | `description.json` (all images)  |

plus one **subset** index, because a `description.json` always covers a whole root:

| source                         | generated                        |
| ------------------------------ | -------------------------------- |
| `jpegai_validation_set_10.txt` | `jpegai_validation_set_10.json`  |

So the validation set carries both a 350-image index and the 10-image one the recipe validates on.
Indexing Open Images (~385k files) takes seconds on a warm page cache and up to a minute cold; every
later start skips it entirely.

Regenerating a list changes its mtime, and `train_image.py` keys the reference RD cache on that (see
[docker/README.md](docker/README.md)) — so the first run after a forced rebuild re-evaluates the
official reference model once (~20 s) and rewrites the cache. Every run after that hits it again.

Generate them without starting a training run, or force a rebuild after adding images:

```bash
docker compose run --rm prepare
FORCE_DESCRIPTION=1 docker compose run --rm prepare
```

Only the paths are indexed, so drop your own images anywhere under a root and re-run with
`FORCE_DESCRIPTION=1`. Subdirectories are walked recursively — Open Images' `train_0/1/2` split needs
no flattening.

---

## 4. Start training

```bash
docker compose up train            # foreground, logs to the terminal
docker compose up -d train         # detached
docker compose logs -f train       # follow a detached run
```

That runs [`train_image_allqp_OpenImage.sh`](train_image_allqp_OpenImage.sh), which is the official
variable-rate recipe: **all 64 QPs trained together** (each sample draws a random QP, its lambda
log-interpolated between the `LAMBDAS` endpoints), 105 epochs, batch 16, 256×256 patches switching to
512×512 at epoch 90, bf16 autocast with `torch.compile max-autotune`.

Edit the variables at the top of that file to change the run — dataset, lambdas, loss weights,
validation QPs, save directory. Point `TRAIN_SCRIPT` at another script to run a different recipe:

```bash
TRAIN_SCRIPT=my_recipe.sh docker compose up train
```

**Resuming is automatic.** `train_image.py` picks up the newest `status_epo*.pth.tar` in `SAVE_DIR`,
so an interrupted run continues where it stopped — which also means *a new dataset needs a new
`SAVE_DIR`*, or it will keep training the previous dataset's weights.

### One GPU or several

`src/utils/common.py:start_train()` spawns **one DDP process per visible GPU**, so the number of
cards is chosen entirely by `DCVC_GPUS` (which becomes `CUDA_VISIBLE_DEVICES`):

```bash
DCVC_GPUS=0   docker compose up train                    # single GPU (the default)
DCVC_GPUS=0,1 docker compose up train                    # two GPUs, DDP
DCVC_GPUS=0,1,2 BATCH_SIZE=18 docker compose up train    # three GPUs, DDP (see below)
```

`--batch_size` is the **global** batch: `get_dataloader()` gives each rank `batch_size //
world_size` and a `DistributedSampler` splits the data. So the recipe's batch 16 stays batch 16 on
any number of cards — the hyperparameters are unchanged, and **the step count per epoch is unchanged
too** (24,049 either way, because every step still consumes 16 images). What changes is only how much
of each step runs on each card.

`get_dataloader()` **asserts the batch divides by the world size**, so batch 16 works on 1, 2, 4, 8
or 16 cards but crashes on 3. Set `BATCH_SIZE` to a multiple of the card count (15 or 18 for three).

Note what a second card does *not* buy you at a fixed batch: splitting batch 16 into 8+8 leaves each
card with half the work per step and adds a NCCL gradient sync to every step, so the wall clock does
not simply halve. To put extra cards to work, scale the batch with them —
`DCVC_GPUS=0,1 BATCH_SIZE=32` keeps 16 images per card and halves the steps per epoch — but batch 32
under the official lr schedule is no longer the published recipe, so treat it as a different
experiment rather than a faster way to run the same one. Benchmark on idle cards before committing
either way.

Rank 0 owns validation, plotting and every checkpoint write, so the contents of `SAVE_DIR` are
identical either way. A resume can switch card count freely: the checkpoint holds no world size.

Two *independent* experiments are two invocations with different GPUs and different `SAVE_DIR`s:

```bash
DCVC_GPUS=0 SAVE_DIR=checkpoints/run_a EXP_NAME=run_a docker compose up -d train
DCVC_GPUS=1 SAVE_DIR=checkpoints/run_b EXP_NAME=run_b docker compose up -d train
```

An interactive shell in the same environment:

```bash
docker compose run --rm shell
```

### Training speed

Benchmarked on an RTX PRO 6000 Blackwell (sm_120, 97 GB) with torch 2.10.0+cu128 and triton 3.6.0,
UF model, batch 16, **ms per step** — lower is better:

| precision / compile        | patch 256 | patch 512 |
| -------------------------- | --------: | --------: |
| fp32 eager                 |    104.81 |         — |
| bf16 eager                 |     65.57 |    279.39 |
| bf16 `parts` / default     |     51.03 |    161.14 |
| bf16 `parts` / reduce-overhead | 43.47 |    149.73 |
| bf16 `parts` / max-autotune |    43.28 |         — |
| bf16 `full` / default      |     37.36 |    119.90 |
| bf16 `full` / reduce-overhead |  34.07 |    119.70 |
| **bf16 `full` / max-autotune** | **31.51** | **106.58** |

So `--amp --compile full --compile_mode max-autotune`, what the recipe uses, is **3.3× faster than
fp32 eager** at patch 256. The catch is compile time: max-autotune spends several minutes on the
first steps, and pays it twice — once for the 256×256 graph and again for 512×512 at epoch 90. The
container keeps the triton/inductor caches in a named volume, so only the first run of a given shape
pays it.

For a short probe run, trade that back:

```bash
EPOCHS=2 COMPILE_MODE=default SAVE_DIR=checkpoints/_probe docker compose up train
```

`COMPILE`, `COMPILE_MODE`, `EPOCHS`, `SAVE_DIR` and `EXP_NAME` all read from the environment, so a
one-off run needs no edit to the recipe.

These numbers assume the GPU is yours alone. Sharing it with other jobs costs a large and
unpredictable fraction of that, so re-measure on an idle card before drawing conclusions.

---

## 5. What a run produces

Everything lands in `SAVE_DIR` (`checkpoints/image_model_allqp_openimage` by default):

```
checkpoints/image_model_allqp_openimage/
├── status_epo<N>.pth.tar          # rolling resume points (weights + optimizer), --keep_status_num
├── ckpt_best_<sig>.pth.tar        # best validation mean-PSNR so far
├── ckpt.pth.tar                   # final weights, written when the last epoch completes
├── archives/ckpt_epo<NNN>.pth.tar # weights-only snapshots, --archive_interval
├── val_rd_history.json            # every validation epoch's RD points
├── rd_plots/                      # per-epoch and all-epochs RD curves, Y / Cb / Cr
└── training_monitor_state.json    # global step, best epoch, best checkpoint path
```

`<sig>` is a hash of the validation configuration, so changing the validation set or QPs starts a
separate best-checkpoint track instead of silently overwriting the old one. The path of the current
best is recorded in `training_monitor_state.json` under `best_checkpoint_path`.

Validation runs every `--val_interval` epochs (and on the final epoch): it measures bpp from the
entropy model and PSNR against the original, then redraws the RD plots with the official model's
curve overlaid in red.

**wandb** is optional. Put `WANDB_API_KEY` in `.env` to get it; without a key `wandb.init` fails, a
warning is logged and training carries on. `--wandb_mode disabled` turns it off explicitly.

---

## 6. Configuration reference

Compose variables — set them in `.env` or prefix the command:

| variable                    | default                            | meaning                                  |
| --------------------------- | ---------------------------------- | ---------------------------------------- |
| `TRAIN_SCRIPT`              | `train_image_allqp_OpenImage.sh`   | what `up train` runs                     |
| `DCVC_GPUS`                 | `0`                                | `CUDA_VISIBLE_DEVICES`; `0,1` runs DDP   |
| `WANDB_API_KEY`             | empty                              | monitoring; optional                     |
| `DOCKER_UID` / `DOCKER_GID` | `1006`                             | uid/gid the container runs as            |
| `OMP_NUM_THREADS`           | `8`                                | per-process thread cap                   |
| `FORCE_DESCRIPTION`         | `0`                                | `1` rebuilds the dataset indexes         |
| `PREPARE_DATASETS`          | `1`                                | `0` skips dataset indexing at start      |

Recipe variables — `train_image_allqp_OpenImage.sh` reads these from the environment, and
`docker-compose.yml` forwards them into the container, so a one-off run needs no edit anywhere:

| variable       | default                              | meaning                          |
| -------------- | ------------------------------------ | -------------------------------- |
| `EPOCHS`       | `105`                                | number of epochs                 |
| `BATCH_SIZE`   | `16`                                 | global batch; must divide by the GPU count |
| `SAVE_DIR`     | `checkpoints/image_model_allqp_openimage` | where the run writes        |
| `EXP_NAME`     | `image_openimage_allqp`              | wandb run name                   |
| `COMPILE`      | `full`                               | `off` / `parts` / `full`         |
| `COMPILE_MODE` | `max-autotune`                       | `default` / `reduce-overhead` / `max-autotune` |

Everything else in that file (dataset, lambdas, loss weights, validation) is edited in place.

The training flags themselves are documented in `python train_image.py --help`; the ones that matter
most are `--lambdas` (rate range), `--mse_yuv_mean` / `--mse_yuv_weights` / `--mse_rgb_weight`
(the distortion term), `--train_qp` (pin a single rate instead of training all 64), `--amp`,
`--compile` and `--seed`.

### `--amp` on a GPU without bf16

`--amp` runs the training forward/backward under **bf16** autocast (weights and optimizer state stay
fp32, validation always runs fp32, so checkpoints are interchangeable with fp32 runs). bf16 rather
than fp16 is deliberate: it keeps fp32's exponent range, so the entropy model's very small
probabilities survive the round trip and no `GradScaler` is needed.

bf16 tensor cores start at compute capability 8.0 (Ampere). On older cards `train_image.py` sorts
itself out at startup:

| GPU                                   | behaviour                                              |
| ------------------------------------- | ------------------------------------------------------ |
| sm_80+ (Ampere, Ada, Hopper, Blackwell) | native bf16, no message                              |
| pre-sm_80 where bf16 still works (V100, 2080 Ti) | **warns** that bf16 is emulated and may be slower than fp32; `--amp` stays on so you can measure it |
| bf16 unavailable entirely             | **warns and disables `--amp`**, training continues in fp32 |

The last row matters: `torch.autocast(dtype=torch.bfloat16)` raises
`RuntimeError: Current CUDA Device does not support bfloat16` on such a device, which would kill the
run on its first batch. The auto-disable is recorded in the wandb config (`amp: false`), so a
degraded run is never mistaken for a bf16 one.

If you want fp32 explicitly, drop `--amp` from the recipe — costing roughly the fp32-vs-bf16 gap in
the speed table above.

---
