#!/bin/bash
# Keep the long runs alive, and keep the path they save to from vanishing.
#
# Written after a 12-hour run died at its first epoch boundary because the
# `runs` symlink at the repo root disappeared: the trainers hold ABSOLUTE save
# paths, so a missing link is invisible until torch.save raises, and the process
# is gone. The checkpoint from the previous epoch survives, so a resume costs one
# epoch -- but only if something notices.
#
# Checks every two minutes: the symlink, and each expected run. Restarts from
# status_latest.pth.tar, which the trainer already resumes from on its own.
cd /home/can_karsal/FLEX-UF
LOG=logs/keepalive.log
mkdir -p logs
say () { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG; }

MSR45_ARGS="--train_dataset /data10/shareddata/openimages/dcvc_train \
--save_dir /home/can_karsal/FLEX-UF/runs/MSR45 \
--pretrain /home/can_karsal/FLEX-UF/runs/warmstart/ckpt_warmstart.pth.tar \
--freeze_encoder --train_patched --epoch_offset 0 --compress_schedule -e 45 \
--steps_per_epoch 350 --batch_size 8 --grad_accum 2 -n 8 --min_crop 512 \
--new_lr_scale 20 --anchor_weight 10.0 --seam_repair grid --adapter_kind scaled \
--distill_weight 1.0 --distill_teacher adjacent --lambdas 10 2048 --num_exits 6 \
--split_depth 2 --latent_patch 16 --latent_halo 2 --aux_weight 1.0 \
--aux_schedule constant --tile_pad replicate --device 1 --log_every 50 \
--ckpt_every 340 --tag MSR45"

while true; do
    if [ ! -e runs ]; then
        ln -s scripts/runs runs && say "RESTORED the runs symlink"
    fi
    if ! pgrep -f -- "--tag MSR45" > /dev/null; then
        say "MSR45 is not running; restarting (it resumes from status_latest)"
        nohup ./.venv/bin/python -u train_flexuf_image.py $MSR45_ARGS \
            >> runs/MSR45/train.log 2>&1 &
        sleep 60
    fi
    for t in RECIPE512 BEST FINE12; do
        pgrep -f -- "--tag $t" > /dev/null || say "$t is NOT running (not auto-restarted: it is the user's main experiment and its argv is not reproduced here)"
    done
    sleep 120
done
