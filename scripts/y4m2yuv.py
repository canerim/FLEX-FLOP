"""Strip a .y4m to raw planar YUV420 8-bit — the form DCVC's test harness reads.

A y4m is exactly the raw planes with a text header and a 'FRAME\\n' marker before
each one, so this is a byte-level strip, not a transcode: no resampling, no
requantisation, nothing that could alter a single sample. That matters because
the file is the reference against which PSNR is computed.
"""
import sys

src, dst = sys.argv[1], sys.argv[2]
with open(src, "rb") as f, open(dst, "wb") as g:
    hdr = b""
    while not hdr.endswith(b"\n"):
        hdr += f.read(1)
    tags = hdr.decode().split()
    w = h = None
    for t in tags:
        if t[0] == "W": w = int(t[1:])
        if t[0] == "H": h = int(t[1:])
        if t[0] == "C" and not t.startswith("C420"):
            raise SystemExit(f"{src}: not 4:2:0 ({t})")
    fsz = w * h * 3 // 2
    n = 0
    while True:
        m = b""
        while not m.endswith(b"\n"):
            b_ = f.read(1)
            if not b_:
                print(f"{dst}: {w}x{h} {n} frames"); raise SystemExit
            m += b_
        if not m.startswith(b"FRAME"):
            raise SystemExit(f"{src}: expected FRAME, got {m[:20]!r}")
        data = f.read(fsz)
        if len(data) < fsz:
            print(f"{dst}: {w}x{h} {n} frames (truncated tail dropped)"); raise SystemExit
        g.write(data); n += 1
