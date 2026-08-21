#!/bin/bash
# Build the description.json index files the ImageFolder datasets need.
#
# src/datasets/image_dataset.py opens <dataset root>/description.json -- a flat
# JSON array of image paths relative to that root -- and make_description.py is
# what writes it (os.walk, so nested subdirectories such as Open Images'
# train_0/1/2 are picked up without flattening anything).
#
# This cannot happen at image build time: the datasets are bind-mounted, so they
# do not exist until the container starts. The entrypoint therefore calls this
# on every start. It is idempotent -- a root that already has description.json
# is left alone unless FORCE_DESCRIPTION=1 -- and it never aborts the container:
# a missing or read-only dataset is a warning, so an interactive shell or a run
# that only touches one of the two datasets still comes up.
set -uo pipefail

REPO_DIR=${REPO_DIR:-/workspace/DCVC-UF}

# "<dataset root>:<image extension>" pairs, whitespace separated: every root
# gets a description.json covering all of its images. The defaults are the two
# datasets the training recipe uses. This fork ships no test/eval entry point,
# so the test sets are deliberately not indexed -- add them here if you bring
# your own evaluation code. Override with e.g.
#   DATASET_DESCRIPTIONS="/workspace/datasets/jpegai_training_random_crop:.png"
DATASET_DESCRIPTIONS=${DATASET_DESCRIPTIONS:-"\
/workspace/datasets/Open_Image/train:.jpg \
/workspace/datasets/jpegai_validation_set:.png"}

# Plain-text lists to mirror as JSON, whitespace separated. A description.json
# covers a whole root, but a *subset* (such as the 10-image validation list the
# training recipe uses) has no root of its own, so it is converted in place:
# <name>.txt -> <name>.json, keeping the line order. ImageValFolder reads both
# formats; producing the JSON twin is what lets validation be fed the same
# format the training set uses. The dataset root is the list's own directory.
DATASET_LIST_JSON=${DATASET_LIST_JSON:-"\
/workspace/datasets/jpegai_validation_set/jpegai_validation_set_10.txt"}

# 1 = rewrite the generated .json files even when they already exist (use after
# adding or removing images).
FORCE_DESCRIPTION=${FORCE_DESCRIPTION:-0}

if [ ! -f "${REPO_DIR}/make_description.py" ]; then
    echo "[prepare_datasets] ${REPO_DIR}/make_description.py not found; skipping" >&2
    exit 0
fi

for entry in ${DATASET_DESCRIPTIONS}; do
    root=${entry%:*}
    ext=${entry##*:}

    if [ ! -d "${root}" ]; then
        echo "[prepare_datasets] ${root} does not exist; skipping" >&2
        continue
    fi

    desc="${root}/description.json"
    if [ -f "${desc}" ] && [ "${FORCE_DESCRIPTION}" != "1" ]; then
        echo "[prepare_datasets] ${desc} already present; skipping" \
             "(FORCE_DESCRIPTION=1 to rebuild)"
        continue
    fi

    if [ ! -w "${root}" ]; then
        echo "[prepare_datasets] ${root} is not writable by uid $(id -u);" \
             "cannot write description.json" >&2
        continue
    fi

    echo "[prepare_datasets] indexing ${root} (*${ext}) -> ${desc}"
    # Scanning Open Images (~385k files) over a bind mount takes a minute or so;
    # this only happens on the first start.
    if ! python "${REPO_DIR}/make_description.py" "${root}" --ext "${ext}"; then
        echo "[prepare_datasets] make_description.py failed for ${root}" >&2
    fi
done

for list_txt in ${DATASET_LIST_JSON}; do
    if [ ! -f "${list_txt}" ]; then
        echo "[prepare_datasets] ${list_txt} does not exist; skipping" >&2
        continue
    fi

    list_json="${list_txt%.txt}.json"
    if [ -f "${list_json}" ] && [ "${FORCE_DESCRIPTION}" != "1" ]; then
        echo "[prepare_datasets] ${list_json} already present; skipping" \
             "(FORCE_DESCRIPTION=1 to rebuild)"
        continue
    fi

    list_root=$(dirname "${list_txt}")
    if [ ! -w "${list_root}" ]; then
        echo "[prepare_datasets] ${list_root} is not writable by uid $(id -u);" \
             "cannot write ${list_json}" >&2
        continue
    fi

    echo "[prepare_datasets] converting ${list_txt} -> ${list_json}"
    if ! python "${REPO_DIR}/make_description.py" "${list_root}" --list "${list_txt}"; then
        echo "[prepare_datasets] make_description.py --list failed for ${list_txt}" >&2
    fi
done
