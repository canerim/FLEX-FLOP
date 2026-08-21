#!/bin/bash
# Makes the container work correctly when started with an arbitrary
# `--user UID:GID` that has no matching /etc/passwd entry (e.g.
# `--user "$(id -u):$(id -g)"`). Without this, whoami/$HOME lookups and
# anything that resolves the login shell via getpwuid (tmux's default-shell,
# `su -`, ...) silently fall back to broken defaults.
#
# Requires /etc/passwd and /etc/group to be group/world-writable (set at image
# build time) since we are already running as a non-root, non-root-group user
# by the time this script executes.
set -euo pipefail

if ! id -un >/dev/null 2>&1; then
    USER_ID="$(id -u)"
    GROUP_ID="$(id -g)"
    USER_HOME="${HOME:-/tmp}"

    if [ -w /etc/group ] && ! getent group "${GROUP_ID}" >/dev/null 2>&1; then
        echo "trainer:x:${GROUP_ID}:" >> /etc/group
    fi

    if [ -w /etc/passwd ]; then
        echo "trainer:x:${USER_ID}:${GROUP_ID}:Container User:${USER_HOME}:/bin/bash" >> /etc/passwd
    fi
fi

export SHELL=/bin/bash
mkdir -p "${HOME:-/tmp}"

# PYTHONPATH is derived from the runtime working directory rather than baked
# into the image, so `docker run -w /workspace/<whatever>` works without
# touching the Dockerfile. `-e PYTHONPATH=...` still takes precedence.
export PYTHONPATH="${PYTHONPATH:-$PWD}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
# TRITON_CACHE_DIR / TORCHINDUCTOR_CACHE_DIR under $HOME keep the max-autotune
# results across container restarts, since compose mounts a volume on $HOME.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${HOME:-/tmp}/.triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${HOME:-/tmp}/.inductor}"
# Only caches are pre-created. train_image.py makes its own SAVE_DIR
# (init_train -> create_folder), and wandb writes into save_dir because
# wandb.init() is passed an explicit dir= -- which beats $WANDB_DIR, so setting
# that variable would only have created an empty wandb/ in the repo.
mkdir -p "${MPLCONFIGDIR}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"

# Generate the dataset description.json files (idempotent; PREPARE_DATASETS=0
# skips it entirely). Datasets are bind-mounted, so this is a start-time job,
# not a build-time one.
if [ "${PREPARE_DATASETS:-1}" = "1" ]; then
    REPO_DIR="${REPO_DIR:-$PWD}" /usr/local/bin/prepare_datasets.sh || true
fi

# wandb.init() is called with mode=online unless the training script passes
# --wandb_mode disabled, and an explicit mode argument overrides $WANDB_MODE.
# Without a key, init fails, train_image.py logs a warning and trains on
# regardless -- so this is a heads-up, not an error.
if [ -z "${WANDB_API_KEY:-}" ] && [ "${WANDB_MODE:-}" != "disabled" ]; then
    echo "[entrypoint] WANDB_API_KEY is not set: wandb monitoring will be skipped" \
         "(training continues). Put it in .env, or pass --wandb_mode disabled." >&2
fi

exec "$@"
