# Released-output D2 distillation

This experiment begins **after** the independent D6 run reaches `state=complete`,
writes its final checkpoint, and GPU 6 is idle. It does not modify the D6 run.

The student starts from the completed D2 epoch-105 checkpoint. The teacher is
Microsoft's released DCVC-UF-Intra D12 image checkpoint, loaded strictly into
the unchanged upstream `DMCI`, frozen, and run in evaluation mode on the **same
augmented YCbCr input and QP** as the student. The teacher target is its decoded
reconstruction, not the source image or a student's own latent decoded with
teacher weights. Every optimizer update uses the official RD objective plus:

`0.25 × mean_i[lambda(QP_i) × MSE_YCbCr(student_recon_i, teacher_recon_i)]`.

The Microsoft image-model loader, `train_0/1/2` data scope (all 384,795 images),
batch size 16, 64-QP sampling, `[10,2048]` lambda range, AdamW defaults,
gradient clip 0.1, FP32, and full 105-epoch learning-rate/patch-size schedule
remain in place. **Warm-starting from D2 and adding the teacher loss are
intentional departures from Microsoft's from-scratch recipe.** Distillation
strength 0.25 is a preregistered starting choice, not a tuned result. The
student's original RD and teacher components are logged separately. The teacher
is evaluated in microbatches of 16 at 256px and 4 at 512px to control memory;
this does not change the student batch size.

The supervisor runs one-step compiled smoke checks at both patch sizes after
GPU 6 is free, then starts the full training. If either smoke check fails, the
full job does not start. Training uses atomic resume checkpoints and bounded
retries, with the data/checkpoint/source hashes in the manifest. At completion,
the supervisor runs the same full-resolution 24-image Kodak diagnostic as the
depth baselines. The diagnostic uses entropy **estimates**, not actual bitstream
rates. A matched-QP BD-rate comparison with both original D2 and released D12
is required before claiming a gain.

Runtime files and logs: `/data10/shareddata/can_karsal/dcvcuf_depth_20260927/runs/d2_distilled_released/`.
Source snapshot: `/data10/shareddata/can_karsal/dcvcuf_depth_20260927/distill_d2_snapshot_20261003/`.
