"""The beta bisection, in one place, for the test curve and for the calibration.

`scripts/router_curve.py` bisects beta against a quality budget on the CTC test
frames. `scripts/beta_calibration.py` has to run that same bisection on held-out
Open Images and then apply its answer, unchanged, to those same test frames. The
bisection lived inside `router_curve.main()` as three closures over its cache, so
a second caller could only copy it, and a copied bisection drifts from the one it
was copied from. This project has paid for that twice already, both times in a
selection rule: `make_paper_tables.pick()` was fixed for stale candidate ordering
and `check_paper.J()` was not, so the checker went on verifying a claim about a
different file from the one the tables used.

The closures therefore move here unchanged and both scripts call them. None of
the arithmetic is new.

What is new is `store`. The calibration set is 512 images rather than 53 frames,
and the cache holds a padded source frame and a latent per entry. Those sit on
whatever device the caller names, and only the small tables the bisection reads
every iteration -- the per-tile MSEs, the reference MSE, the log-probabilities --
stay on the card. `.to()` returns a tensor already on the requested device
unchanged, so a caller that stores on the card computes exactly what it computed
when this code lived in router_curve.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .eval import reference_frame_mse, tiled_exit_mses, true_frame_mse
from .measure import measured_saving_pct

# `exit_costs()` prices the decoder. It does NOT include the router, because in
# the signalled configuration the decoder never runs one. In configuration B it
# does, so its arithmetic is part of what the decode costs and has to come off
# the saving -- leaving it out would be the same class of error as the
# denominator in DECISIONS 58: a real cost sitting outside the figure it belongs
# in. Measured per RGB pixel against the decode's 217,113 MAC/px (mac_audit.py).
DECODE_MACPX = 453540e6 / (1920 * 1088)

#: The bisection bracket. -50 did not reach the all-deepest allocation: a
#: confident head has log-probability gaps of hundreds, and the masked exits sit
#: at -1e4, so the tilt has to be able to outweigh those.
LO, HI = -2.0e4, 2.0e4


def router_share(head, args, pixels: int) -> float:
    """Fraction of one decode that this router head costs, measured not assumed."""
    macs = {}

    def hook(m, i, o):
        if isinstance(m, torch.nn.Conv2d):
            macs[id(m)] = (m.in_channels * m.out_channels * m.kernel_size[0]
                           * m.kernel_size[1] * o.shape[-1] * o.shape[-2]
                           / m.groups)
        elif isinstance(m, torch.nn.Linear):
            n = 1
            for d in o.shape[:-1]:
                n *= d
            macs[id(m)] = m.in_features * m.out_features * n

    hs = [m.register_forward_hook(hook) for m in head.modules()]
    with torch.no_grad():
        head(*args)
    for h in hs:
        h.remove()
    return sum(macs.values()) / pixels / DECODE_MACPX


@torch.no_grad()
def build_cache(net, ref, head2, cfg, images, qp_v: int, dev, *,
                per_frame_scale: bool = False, store=None, rshare: float = 0.0,
                on_rshare=None):
    """One cache entry per image: (M, R, lp, y, q, xp), and the router's share.

    `rshare` is threaded through rather than recomputed per rate: it is a
    property of the head and the frame size, and router_curve measures it once,
    on the first frame it sees, then charges it against every saving.
    """
    store = dev if store is None else store
    cache = []
    for x in images:
        x = x.to(dev)
        _, _, H, W = x.shape
        P = cfg.rgb_patch
        ph, pw = (-H) % P, (-W) % P
        xp = F.pad(x, (0, pw, 0, ph), mode="replicate") if (ph or pw) else x
        qp = torch.full((1,), qp_v, dtype=torch.int32, device=dev)
        y, q, aux = net._encode_to_latent(xp, qp)
        # DEPLOYED path, one tiled decode per exit. The full-frame form runs the
        # trunk over the whole frame, so the tiling penalty cancels against the
        # full-frame reference and never appears in the reported dB
        # (flexuf/eval.py).
        M = tiled_exit_mses(net.dec, y, q, xp, cfg)
        R = reference_frame_mse(ref.dec, y, q, xp)
        # What the decoder can see: the stem it has already computed, and (for
        # V2) the latent and the entropy model's scales, both of which it has
        # decoded before the trunk runs. Still zero added bits -- nothing here
        # comes from the source frame.
        stem = net.dec.upsample(y)
        for g in range(cfg.split_depth):
            stem = net.dec.groups[g](stem)
        if head2 is not None:
            lg = head2(stem, y, aux["scales_hat"], qp,
                       cfg.feature_patch, cfg.latent_patch)
        else:
            lg = net.router_head(stem, qp, cfg.feature_patch)
        # The head masks exits below the split depth by ASSIGNING -1e4, which
        # stops being a mask the moment the head's own logits reach that scale
        # -- and this one's have: its raw outputs sit near -10000, so the "mask"
        # is the LARGEST entry in every row and log_softmax puts almost all the
        # mass on the two exits that do not exist. argmax then picks one and
        # clamp(min=j) turns it into the cheapest real exit, for reasons that
        # have nothing to do with the tile.
        #
        # Fixed at the decision site rather than in head2.py, because the head
        # files are imported by live training runs and a crash-restart would
        # pick up the edit mid-experiment. Every rule below reads lg[:, j:] only.
        lg = lg[:, cfg.split_depth:]
        lp = F.log_softmax(lg, 1)
        if per_frame_scale:
            # The tilt trades log-probability against cost, so its meaning
            # depends on how large the log-probabilities are, and a 144 K head is
            # far more confident on some frames than others. Normalising by the
            # frame's own mean magnitude removes that, and costs nothing: it is a
            # statistic of the decoder's own logits.
            m = (-lp).mean().clamp_min(1e-6)
            lp = lp / m
        if rshare == 0.0:
            px = xp.shape[-1] * xp.shape[-2]
            rshare = (router_share(head2, (stem, y, aux["scales_hat"], qp,
                                           cfg.feature_patch, cfg.latent_patch),
                                   px)
                      if head2 is not None else
                      router_share(net.router_head,
                                   (stem, qp, cfg.feature_patch), px))
            if on_rshare is not None:
                on_rshare(rshare)
        cache.append((M, R, lp, y.to(store), q.to(store), xp.to(store)))
    return cache, rshare


def exit_map(lp, cost, j: int, beta: float):
    """The tile assignment equation (2) makes at this tilt."""
    return (lp - beta * cost[None, j:]).argmax(1) + j


def at_beta(cache, cost, j: int, rshare: float, beta: float):
    """Saving and dB from the cheap table: (vs deepest, dB, vs release)."""
    SV = SVR = DB = n = 0.0
    for M, R, lp, _y, _q, _xp in cache:
        k = exit_map(lp, cost, j, beta)
        # rshare is added to the cost, i.e. subtracted from the saving: the
        # decoder pays for the router here.
        SV += (1 - (cost[k].mean() + rshare) / cost[-1]).item()
        SVR += (1 - cost[k].mean() - rshare).item()
        DB += (10 * torch.log10(
            M.gather(1, k[:, None]).squeeze(1).mean() / R)).item()
        n += 1
    return 100 * SV / n, DB / n, 100 * SVR / n


def true_db(dec, cache, cost, j: int, beta: float, dev):
    """What a decode of the router's actual map delivers.

    The table measures each tile with its neighbours at the same exit; a routed
    frame is mixed. One real decode per frame settles it, which is affordable
    outside the bisection but not inside.
    """
    tot = 0.0
    for M, R, lp, y_, q_, xp_ in cache:
        k = exit_map(lp, cost, j, beta)
        tot += (10 * torch.log10(
            true_frame_mse(dec, y_.to(dev), q_.to(dev), xp_.to(dev), k)
            / R)).item()
    return tot / len(cache)


def measured_saving(dec, ref_dec, cache, cost, j: int, rshare: float,
                    beta: float, dev) -> float:
    """The saving counted off the decode with hooks, router charged against it.

    The arithmetic model under-bills the shallow exits by a constant 0.008 of a
    released decode (scripts/ceiling_measured.py), and every table in the paper
    reports the hook count, so a caller has to carry this or configuration B is
    on a different definition from configuration A in the table that compares
    them.
    """
    m = 0.0
    for _M, _R, lp, y_, q_, xp_ in cache:
        k = exit_map(lp, cost, j, beta)
        m += measured_saving_pct(dec, ref_dec, y_.to(dev), q_.to(dev), k)
    return m / len(cache) - 100 * rshare


def bisect_to(cache, cost, j: int, rshare: float, t: float,
              lo: float = LO, hi: float = HI, iters: int = 60) -> float:
    """Bisect beta on the cheap table to a table-dB of `t`.

    Large beta = cheap exits = worse dB, so dB is increasing in beta and the
    invariant is the same as the lambda bisection's.
    """
    if at_beta(cache, cost, j, rshare, hi)[1] <= t:
        return hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if at_beta(cache, cost, j, rshare, mid)[1] <= t:
            lo = mid
        else:
            hi = mid
    return lo


def bisect_beta(dec, cache, cost, j: int, rshare: float, target: float, dev,
                rounds: int = 6, tol: float = 5e-4):
    """Bisect on the cheap table, then correct against a real decode.

    Same two-level scheme as signalled_curve.py, and for the same reason: the
    table is what an argmin can be run against, and the decode is what is
    delivered. Returns (beta, delivered dB).
    """
    inner, beta, td = target, None, None
    for _ in range(rounds):
        beta = bisect_to(cache, cost, j, rshare, inner)
        td = true_db(dec, cache, cost, j, beta, dev)
        if abs(td - target) < tol:
            break
        inner = inner + (target - td)
    return beta, td


def floor_db(dec, cache, cost, j: int, dev) -> float:
    """The dB at the deepest allocation the tilt can reach: the budget's floor."""
    return true_db(dec, cache, cost, j, LO, dev)
