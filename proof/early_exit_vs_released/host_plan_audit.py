"""CPU-only exactness and dispatch-cost audit for the routed tile planner."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics
import sys
import time

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from flexuf.kernels.planned_decoder import make_tile_plan


def reference(mode_map: torch.Tensor):
    em = mode_map.clamp(2, 5)
    order = torch.argsort(em, descending=True, stable=True)
    sorted_em = em[order]
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(em.numel())
    bounds = tuple(int((sorted_em > depth).sum()) for depth in range(2, 6))
    return order, inverse, bounds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent /
                        'offline_data/host_plan_audit.json')
    args = parser.parse_args()
    torch.set_num_threads(1)
    rng = random.Random(20261003)
    sizes = (1, 2, 8, 15, 40, 64, 65, 80, 135)
    checked = 0
    for n in sizes:
        for _ in range(500):
            values = torch.tensor([rng.randint(-2, 8) for _ in range(n)])
            plan = make_tile_plan(values, n_tiles=n, split_depth=2,
                                  num_exits=6, device=torch.device('cpu'))
            old = reference(values)
            if (not torch.equal(plan.order, old[0]) or
                    not torch.equal(plan.inverse, old[1]) or
                    plan.bounds != old[2]):
                raise RuntimeError(f'Planner changed stable route order for {n} tiles')
            checked += 1
    medians = {}
    for n in (8, 15, 40, 80, 135):
        values = torch.tensor([rng.randrange(2, 6) for _ in range(n)])
        for _ in range(200):
            reference(values)
            make_tile_plan(values, n_tiles=n, split_depth=2,
                           num_exits=6, device=torch.device('cpu'))
        samples = {'reference': [], 'planned': []}
        for i in range(2000):
            order = ('reference', 'planned') if i % 2 else ('planned', 'reference')
            for arm in order:
                start = time.perf_counter_ns()
                if arm == 'reference':
                    reference(values)
                else:
                    make_tile_plan(values, n_tiles=n, split_depth=2,
                                   num_exits=6, device=torch.device('cpu'))
                samples[arm].append((time.perf_counter_ns() - start) / 1000)
        ref = statistics.median(samples['reference'])
        planned = statistics.median(samples['planned'])
        medians[str(n)] = {'reference_us': ref, 'planned_us': planned,
                           'host_plan_speedup': ref / planned}
    cpu = None
    for line in Path('/proc/cpuinfo').read_text().splitlines():
        if line.startswith('model name'):
            cpu = line.split(':', 1)[1].strip()
            break
    result = {'schema': 1, 'gpu_used': False, 'torch': torch.__version__,
              'cpu': cpu, 'exact_maps': checked, 'cases': medians,
              'scope': 'CPU route ordering/planning only; excludes H2D, GPU synthesis and full decode'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(args.out), 'exact_maps': checked,
                      'cases': medians}, allow_nan=False))


if __name__ == '__main__':
    main()
