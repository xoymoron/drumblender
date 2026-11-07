# Clean epoch135 control protocol

Code is developed locally and deployed with commit/push, then pull on `DSP_sjh`.
No reconstruction weights, model code, training configuration, or existing runs
are changed. No cache, training, or audition is launched by installation.

## Frozen inputs and split

`scripts/control_epoch135.sh` names the epoch135 checkpoint found on DSP_sjh,
the existing 64-mode cache, processed audio and noise bank. Environment variables
can override these paths. The cache records checkpoint, bank, manifest and modal
configuration identities; the split and seed come from the reconstruction run.
An epoch85 h cache or SAE cannot initialize this experiment.

## Training

1. Extract new train/validation h and train-only channel mean/std.
2. Train a randomly initialized temporal SAE using `L_h + L_AuxK/32` only.
   Keep the best validation checkpoint. Dictionary: 512 features, TopK 8,
   nine-frame kernels; Adam 3e-4. Every feature index is eligible.
3. Clone this warm-up into main and comparison with identical weights, separate
   storage and fresh Adam states (1e-4). Both receive the same latent batches
   and update counts. Main additionally receives `0.1 L_int` through frozen
   S, N and S+N; readers use Adam 3e-4. There is no modal-only warm-up.
4. Recalculate the train-supported cohort (at most 64 features), per-feature
   edit floors and acoustic calibration from this checkpoint. Validation selects
   the warm-up checkpoint and candidates; paired models retain their final matched
   update. Test does not participate in training or selection.
5. Keep the signed exponential knob and beta=0.25. Strike code stays neutral;
   T may follow edited S, but T is not an optimization target.

Warm-up defaults to 25 minutes with a 30-minute cap. The paired command's
120-minute target and 180-minute cap **include the entire warm-up invocation**,
including warm-up validation, not just the selected checkpoint's elapsed time.
Paired timing includes both SAE updates and validation. Cache extraction,
calibration, screening and rendering are measured separately. Checks between
updates reserve time for validation/checkpoint writes; they are not an OS timeout
for a stalled CUDA operation. No training-time or sound-quality result is claimed.
Resuming checks the original protocol and conservatively charges interruption
time using the saved UTC clock, including across server reboots.

## Explicit commands (DSP_sjh only)

```bash
cd ~/percussives/drumblender
bash scripts/control_epoch135.sh          # Help only; no training

# Run these only when the experiment is authorized:
bash scripts/control_epoch135.sh cache
bash scripts/control_epoch135.sh warmup
bash scripts/control_epoch135.sh train
bash scripts/control_epoch135.sh audition
```

The default root is `runs/control_epoch135_clean`. Existing cache/warm-up/run
folders are not overwritten. Training alone does not render or evaluate test.
The audition is `paired/audition/index.html`, with branch solos and a comparison
page. Candidate sets are ranked independently for main and comparison from
fixed validation examples; the user chooses two main controls. WAVs are FLOAT.

```bash
# After choosing two NEW validation atom IDs:
bash scripts/control_epoch135.sh test ID1 ID2
bash scripts/control_epoch135.sh analyze
```

The IDs are required inputs and become part of the locked test protocol.
No atom ID is excluded or preselected. Reports, figures and the two-knob grid
read the selected IDs from that protocol. Old modal-only/intervention scripts
remain legacy utilities, not stages of this clean protocol. Previous results
and manuscript claims are not relabeled as epoch135 results.

## Implementation checks

- Common fork: equal values, independent parameter storage; no atom exclusion.
- Reject old intervention initialization and mismatched reconstruction/cache.
- Subtract total warm-up duration from both continuation budgets.
- Require two distinct, in-range, audition-selected IDs before test.
- Existing tests cover frozen synthesis gradients, neutral edits and noise-only
  branch invariants. These are small synthetic checks, not reconstruction runs.
