# Fixed epoch-30 CPU validation milestone

This separate code snapshot schedules the same full100-image, five-QP actual-byte protocol used at epoch20. It does not modify the original codec/evaluator, change official training, select a checkpoint by quality, or use a GPU. D2/D4/D6 are evaluated at exactly epoch30/105; releasedD12 remains the separate practical anchor.

The queue waits for all three epoch30 weight snapshots **and their completed end-of-epoch validation records**. A visible half-written checkpoint is not sufficient. It then verifies36 fresh-process encode/decode cases, including irregular geometry and symbol/container guards, before evaluating all2,000 model-image-QP cases on the fixed100 DIV2K centre512 crops. A decoder receives only stream metadata and bytes; no encoder tensor cache. FUFREF1 CPUFP32 is a research format, not native CUDA.

Two CPU threads and nice10 are used. One lock prevents duplicate queues. Waiting expires after48hours; preflight and analysis each have30-minute limits and evaluation has a12-hour limit. Failures stop the queue without parameter changes or automatic retries. Only the queue's own evaluation child may be stopped on its failure. Official GPU training processes are never signalled.

Before the queue is launched, this cloned codec pipeline is tested with the existing epoch20 snapshots. That36-case pipeline test is engineering evidence, not an epoch30 result. The actual milestone preflight is repeated after epoch30 is ready.

Outputs:
- Server queue status: `/data10/shareddata/can_karsal/dcvcuf_depth_20260927/research/epoch030_validation_queue/progress.json`
- Full streams/cases: `.../research/div2k100_reference_epoch030`
- Analysis after completion: `docs/research/2026-09-27-six-hour/div2k100_epoch030`

The queue never edits or pushes the paper automatically. Its results require inspection before publication.
