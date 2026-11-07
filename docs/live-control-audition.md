# Live validation audition

The original fixed three examples per atom stay unchanged. The live server
adds a random fourth slot; reloading the page restores the last rendered draw.
Each request draws uniformly from the validation split, excluding the fixed
three examples and earlier draws for that model/atom. The pool resets only after
exhaustion. There is no activation, pack, instrument, duration or effect-size
filter. Inactive features and render failures are retained in the draw log.

The server holds the frozen reconstruction model and main/comparison SAEs in
eval mode. It reuses cached validation h, keeps neutral strike code and noise
offsets fixed, and renders 0, +/-4 and +/-16. Playback copies are RMS-matched;
raw and playback WAVs both use Float32. Rendering is on demand, not audio callback
streaming. Existing static HTML and training artifacts are not overwritten.

Start on DSP_sjh using the existing epoch135 paths:

```bash
cd ~/percussives/drumblender
bash scripts/control_epoch135.sh serve
```

Forward from local PowerShell (keep the terminal running):

```powershell
ssh -N -o ExitOnForwardFailure=yes -L 8766:127.0.0.1:8766 DSP_sjh
```

Open http://127.0.0.1:8766/index.html. Each atom has a new random-sample button.
The main, modal, noise and comparison pages support the same live controls.
The server binds only to loopback and serializes GPU requests; the existing
static server on 8765 can continue independently.

## Choosing and reporting controls

- Listen to the fixed examples and several unfiltered draws for each candidate.
  Inactive draws count as limited coverage rather than being silently replaced.
  Listen at neutral and +/-4 first, then +/-16 when useful. Extreme +/-64 is not
  required in this view. Check both usefulness and overlap between two knobs.
- Send the two main atom IDs and the useful strength range after listening.
  Checkboxes persist locally per SAE/model; they do not lock or launch a test.
- Freeze these choices/ranges before held-out evaluation. The original test
  split remains untouched by this interface; no model is trained here.
- Fixed activation-selected examples and uniform random validation draws serve
  different purposes and must be described as such in the paper. Artist choice
  is exploratory selection, not a perceptual user study or proof of disentanglement.
- Report chosen-controls' held-out activation coverage, modal/noise/mix response
  versus the paired comparison at declared strengths, and joint/noise-only path
  measurements. Larger change is not a quality score. Claim generality only to
  the extent supported by those distributions. T is not an intervention target.
- Training sampled the full signed exponential range up to +/-64. Restricting
  listening/evaluation to +/-4 or +/-16 does not change that training distribution;
  document the distinction wherever training and evaluation strengths are given.

`paired/audition/live/protocol.json` locks the SAE/checkpoint/cache hashes and
sampling policy. `live/draws.jsonl` records every draw, identity, activation,
elapsed render time, outcome and file links, including errors. This is an audit
trail of exploration, not test evidence. No manuscript text is edited by serving.
