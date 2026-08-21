# Helper: generate description.json for the image dataset (ImageFolder).
# Usage:
#   python make_description.py /path/to/image_dataset
#   python make_description.py /path/to/image_dataset --list subset.txt
# It scans the folder recursively for PNG images and writes description.json
# (a flat JSON array of paths relative to the dataset root).
#
# With --list it converts an existing plain-text list (one relative path per
# line) into the same JSON format instead of scanning, so a subset such as
# jpegai_validation_set_10.txt gets a .json twin and validation can be fed the
# same file format the training set uses. ImageValFolder accepts both, but
# reading JSON everywhere keeps train and val consistent.

import argparse
import json
import os
import sys


def scan_root(root, ext):
    rel_paths = []
    for dirpath, _, filenames in os.walk(root):
        for name in sorted(filenames):
            if name.lower().endswith(ext):
                full = os.path.join(dirpath, name)
                rel_paths.append(os.path.relpath(full, root))
    rel_paths.sort()
    return rel_paths


def read_list(list_path, root):
    """Read a plain-text list, keeping its order.

    The order is deliberately preserved rather than sorted: validation averages
    over the images in list order, and reordering would change the reference RD
    cache identity for no reason.
    """
    rel_paths = []
    with open(list_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rel_paths.append(line.split()[0])
    missing = [p for p in rel_paths if not os.path.exists(os.path.join(root, p))]
    if missing:
        raise SystemExit(
            f'{len(missing)} entries of {list_path} are not under {root}, '
            f'e.g. {missing[0]}'
        )
    return rel_paths


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=str, help='Image dataset root folder')
    parser.add_argument('--ext', type=str, default='.png',
                        help='Image extension to include (default: .png)')
    parser.add_argument('--list', type=str, default=None,
                        help='convert this plain-text list (one relative path per '
                             'line) instead of scanning the root; the output goes '
                             'next to it as <stem>.json')
    parser.add_argument('-o', '--out', type=str, default=None,
                        help='output path (default: <root>/description.json, or '
                             '<list stem>.json with --list)')
    args = parser.parse_args(argv)

    root = os.path.abspath(args.root)
    ext = args.ext.lower()

    if args.list:
        rel_paths = read_list(args.list, root)
        out_path = args.out or f'{os.path.splitext(args.list)[0]}.json'
        source = args.list
    else:
        rel_paths = scan_root(root, ext)
        out_path = args.out or os.path.join(root, 'description.json')
        source = f'"{ext}" images under {root}'

    with open(out_path, 'w') as f:
        json.dump(rel_paths, f, indent=0)

    print(f'found {len(rel_paths)} entries from {source}')
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main(sys.argv[1:])
