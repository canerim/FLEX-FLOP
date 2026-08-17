"""Prove DDP actually synchronises gradients before trusting a 2-GPU run.

The failure this guards against is silent: without correct DDP wiring each rank
keeps its own gradients, the loss curve looks entirely normal, and the run
trains two different models on half the data each. Nothing in the logs says so.

The test is a direct one. Take a batch of 2N. Run it on ONE gpu in a single
step and record the gradients. Then run the same batch split across two ranks
under DDP and record theirs. DDP averages gradients across ranks, so if it is
wired correctly the two must agree to numerical precision. If it is not, each
rank's gradient is computed from its own half and the numbers diverge
immediately.
"""
import os, sys, tempfile
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path.home() / "DCVC"))
from flexuf.config import FlexUFConfig
from flexuf.model import FlexUFIntra, load_flexuf_state

CKPT = "runs/warmstart/ckpt_warmstart.pth.tar"
N = 2   # per rank


def build(dev):
    torch.manual_seed(0)
    cfg = FlexUFConfig(split_depth=2, latent_patch=16, seam_repair="grid",
                       adapter_kind="scaled")
    net = FlexUFIntra(cfg).to(dev)
    load_flexuf_state(net, torch.load(CKPT, map_location="cpu", weights_only=False))
    for n_, p in net.named_parameters():
        p.requires_grad = n_.startswith(("dec.", "router_head."))
    return net, cfg


def data(dev):
    g = torch.Generator().manual_seed(7)
    x = (torch.rand(2 * N, 3, 512, 512, generator=g) - 0.5) * 0.4
    qp = torch.randint(0, 64, (2 * N,), generator=g, dtype=torch.int32)
    return x.to(dev), qp.to(dev)


def grads(net):
    return {n_: p.grad.detach().clone() for n_, p in net.named_parameters()
            if p.grad is not None}


def worker(rank, path):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12399"
    dist.init_process_group("nccl", rank=rank, world_size=2)
    torch.cuda.set_device(rank)
    dev = f"cuda:{rank}"
    net, _ = build(dev)
    ddp = DDP(net, device_ids=[rank], find_unused_parameters=True)
    x, qp = data(dev)
    xs, qs = x[rank * N:(rank + 1) * N], qp[rank * N:(rank + 1) * N]
    out = ddp(xs, qs, mode="random_depth")
    # Mean over the global batch, which is what a single-GPU step computes.
    (out["mses"][0].mean() + 0.01 * out["bpp"].mean()).backward()
    if rank == 0:
        torch.save({k: v.cpu() for k, v in grads(net).items()}, path)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    tmp = tempfile.mktemp(suffix=".pt")
    mp.spawn(worker, nprocs=2, args=(tmp,), join=True)

    dev = "cuda:0"
    net, _ = build(dev)
    x, qp = data(dev)
    out = net(x, qp, mode="random_depth")
    (out["mses"][0].mean() + 0.01 * out["bpp"].mean()).backward()
    single = grads(net)
    dual = torch.load(tmp, map_location="cpu")

    shared = [k for k in single if k in dual]
    worst, name = 0.0, ""
    for k in shared:
        a, b = single[k].cpu(), dual[k]
        d = (a - b).abs().max().item() / max(a.abs().max().item(), 1e-12)
        if d > worst:
            worst, name = d, k
    print(f"\n  {len(shared)} gradyan tensoru karsilastirildi")
    print(f"  en buyuk BAGIL fark: {worst:.3e}  ({name})")
    if worst < 5e-2:
        print("  [ok ] DDP gradyanlari senkronize ediyor — 2 GPU tek kosu guvenli")
    else:
        print("  [HATA] gradyanlar UYUSMUYOR — 2 GPU kosusu sessizce yanlis olurdu")
        sys.exit(1)
