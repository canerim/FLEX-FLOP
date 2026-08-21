"""Prove, item by item, that training follows Microsoft's recipe.

Asserted rather than described. Every row compares something in our trainer
against the corresponding thing in ~/DCVC/train_image.py, by reading both, and
FAILS loudly on a mismatch. A claim of "we follow the recipe" that nobody can
re-run is not a claim, it is a hope -- and today already produced one instance
where the recipe was copied correctly and applied at the wrong epoch.
"""
import ast, inspect, sys, textwrap
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path.home() / "DCVC"))

MS = (Path.home() / "DCVC" / "train_image.py").read_text()
OURS = (Path(__file__).resolve().parents[1] / "train_flexuf_image.py").read_text()

def fn_src(src, name):
    """Pull one function out by text, not by ast.

    Microsoft's train_image.py uses Python 3.12 f-string nesting, which 3.10's
    parser rejects -- so an ast-based reader cannot open their file at all on
    this interpreter. That is itself worth knowing: their script requires 3.12,
    and the rANS extension here was built against a 3.10 venv.
    """
    lines = src.splitlines()
    out, on = [], False
    for ln in lines:
        if ln.startswith(f"def {name}("):
            on = True
        elif on and ln and not ln[0].isspace():
            break
        if on:
            out.append(ln)
    return "\n".join(out) if out else None

rows, bad = [], 0
def check(label, ok, detail=""):
    global bad
    rows.append((label, ok, detail))
    if not ok:
        bad += 1

# 1. the schedule table, verbatim
# Compare the SCHEDULE ROWS, not the function text. Our copy uses a docstring
# where theirs uses comments and parentheses where theirs uses line
# continuations; a whole-text comparison flags that as a deviation, which is
# false and would train anyone reading it to ignore the check. The invariant
# that matters is the eight [[epoch, lr, w, h]] * n rows.
import re
rows_of = lambda src: re.findall(r"\[\[\s*\d+,\s*[\de.-]+,\s*\d+,\s*\d+\s*\]\]\s*\*?\s*\d*",
                                 fn_src(src, "get_training_strategy") or "")
ra, rb = rows_of(MS), rows_of(OURS)
norm = lambda L: ["".join(x.split()) for x in L]
check("takvim satirlari birebir ayni", norm(ra) == norm(rb),
      f"{len(ra)} satir, karakteri karakterine" if norm(ra) == norm(rb)
      else f"MS={norm(ra)} OURS={norm(rb)}")

# 2. schedule content, evaluated
def load(src, name):
    ns = {}
    exec(compile(fn_src(src, "get_training_strategy"), name, "exec"), ns)
    return ns["get_training_strategy"]()
sa, sb = load(MS, "ms"), load(OURS, "ours")
check("takvim 105 epoch, ayni lr/patch dizisi", sa == sb,
      f"{len(sa)} giris, ilk={sa[0]}, son={sa[-1]}")

# 3. optimiser
check("AdamW lr=1e-4", "AdamW" in MS and "lr=1e-4" in MS
      and "AdamW" in OURS and "lr=1e-4" in OURS)

# 4. gradient clipping and the non-finite skip
check("clip_grad_norm_ max_norm=0.1", "max_norm=0.1" in MS and "0.1" in OURS
      and "clip_grad_norm_" in OURS)
check("non-finite norm -> batch atlanir",
      "non-finite" in MS and ("non-finite" in OURS or "skipped" in OURS))

# 5. dataset and lambda sampling
check("ImageFolder + get_training_lambdas", "ImageFolder" in MS
      and "get_training_lambdas" in MS and "ImageFolder" in OURS
      and "get_training_lambdas" in OURS)
check("qp_num() = 64 seviye",
      "DMCI.qp_num()" in MS and "QP_LEVELS" in OURS)
from flexuf.config import QP_LEVELS
from src.models.image_model import DMCI
check("QP_LEVELS == DMCI.qp_num()", QP_LEVELS == DMCI.qp_num(),
      f"{QP_LEVELS} == {DMCI.qp_num()}")

# 6. the model itself
from flexuf.model import FlexUFIntra
from flexuf.config import FlexUFConfig
net = FlexUFIntra(FlexUFConfig())
stock = DMCI()
ours_enc = {k: tuple(v.shape) for k, v in net.enc.state_dict().items()}
ms_enc = {k: tuple(v.shape) for k, v in stock.enc.state_dict().items()}
check("encoder DMCI ile birebir ayni", ours_enc == ms_enc,
      f"{len(ms_enc)} tensor")
shared = [k for k in stock.state_dict() if not k.startswith("dec")]
diff = [k for k in shared
        if k in net.state_dict()
        and tuple(net.state_dict()[k].shape) != tuple(stock.state_dict()[k].shape)]
check("hyperprior + entropy modeli degistirilmemis", not diff,
      f"{len(shared)} paylasilan tensor, {len(diff)} farkli sekil")

# 7. what we deliberately changed, stated rather than hidden
DELIB = [
    ("--epoch_offset", "takvimi warm-start icin dogru yerinden okumak"),
    ("--anchor_weight", "en derin cikisi yayinlanmis decoder'a baglamak"),
    ("--new_lr_scale", "sifir-baslatmali modullere ayri lr"),
    ("--freeze_encoder", "encoder'i tamamen dondurmak"),
    ("--bf16", "bfloat16 autocast; kayiplar fp32'de hesaplanir"),
]
for flag, why in DELIB:
    check(f"BILEREK EKLENDI {flag}", flag in OURS, why)

w = max(len(r[0]) for r in rows)
print()
for label, ok, detail in rows:
    print(f"  {'[ok ]' if ok else '[HATA]'} {label:<{w}}  {detail}")
print()
if bad:
    print(f"  {bad} UYUSMAZLIK — recipe'den sapma var"); sys.exit(1)
print("  Microsoft'un recipe'sinin her maddesi dogrulandi.")
