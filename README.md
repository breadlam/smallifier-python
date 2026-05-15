# Smallifier

> Pin a video to an exact size. Maximize quality within that envelope.

A specialized AV1 encoder for size-budgeted video distribution — Discord's 10 MB limit, an email attachment cap, a chat platform's upload ceiling. Optimized for fast-motion content with on-screen text, where conventional encoders compromise text legibility first.

The naive way to hit a size target is to compute `target_bits / duration` and feed it as a bitrate to a two-pass encoder. The result is uniform mediocrity: easy scenes get more bits than they need, hard scenes get fewer than they need, and aggressive size targets produce uniformly poor output across the whole video. This tool does something different.

## How it works

**Per-scene convex-hull search.** [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) splits the source at shot boundaries. For each scene independently, the encoder samples a grid of (resolution × CRF) operating points and measures the (size, VMAF) curve for that scene. The Pareto frontier of those points captures the actual rate-quality tradeoff for *this specific content* — not an average over all scenes, not a guess from heuristics, not a learned model. The grid is the probe.

**Adaptive resolution ladder.** Per scene, the encoder walks up the resolution ladder (360p → 432p → 540p → 720p → 1080p), stopping as soon as the next rung either contributes nothing to the Pareto frontier (the scene doesn't benefit from more pixels at this bitrate) or produces encodes larger than the scene could plausibly be allocated. Easy scenes stop climbing early; visually complex scenes climb further. The search shape is content-driven, not pre-configured.

**Slope-equalization allocator.** Given each scene's Pareto frontier, a greedy knapsack repeatedly applies the upgrade (across all scenes) with the highest ΔVMAF × duration / Δbytes ratio. This is the Lagrangian-multiplier solution to constrained rate-distortion optimization — bits flow to where they buy the most quality per byte, which is the principle behind Netflix's [Dynamic Optimizer](https://netflixtechblog.com/dynamic-optimizer-a-perceptual-video-encoding-optimization-framework-e19f1e3a277f). Complex scenes naturally get more than their proportional share; simple scenes accept lower bitrates because the perceptual loss is small.

**Output resolution emerges from budget.** Rather than picking output resolution as a configuration choice, the encoder computes it from the bit-budget feasibility floor: the largest standard resolution where a generous chunk allocation could plausibly clear a perceptual quality threshold (~0.05 bpp for AV1 on fast-motion content). Aggressive 10 MB targets land at 432p–720p; loose 100 MB targets land at 1080p. No more 1080p-output-with-doomed-bitrate.

**Adaptive audio.** Audio bitrate scales with total budget — 5% of total, clamped to [24, 96] kbps Opus. Tight budgets reserve more for video; loose budgets get near-transparent audio.

**Text-friendly encoder configuration.** SVT-AV1 at preset 2 with:
- `enable-tf=0` — disables the temporal filter, which softens text on motion
- `enable-qm=1` with full quality-matrix range — preserves high-frequency detail
- `aq-mode=2` — variance-based adaptive quantization spends bits where the eye notices
- `tune=0` — psy-visual mode optimized for perceptual quality
- 10-bit output by default; more compression-efficient than 8-bit even for 8-bit source, and supported wherever AV1 plays at all

**Honest quality measurement.** Quality is measured with VMAF v0.6.1neg (the "no enhancement gain" model) rather than standard VMAF. The default model rewards sharpening filters that don't actually improve perceived quality; NEG corrects for this, giving the allocator an honest signal during optimization.

**Emergency CRF probing.** If the cheapest combination of trial encodes still exceeds the budget after the main exploration phases, the encoder automatically probes higher CRFs (60, 63) at the smallest resolution to extend the Pareto frontier toward smaller sizes. Rarely needed; available when it is.

**Parallelism.** Trial encodes are dispatched to a multiprocessing pool sized to your core count. A 100-second 1080p source produces 200–400 trial encodes on the adaptive grid; on a 256-core machine the full run completes in 5–10 minutes.

## Requirements

- `ffmpeg` built with `libsvtav1`, `libopus`, and `libvmaf` (verify with `ffmpeg -encoders | grep -iE 'svtav1|opus'` and `ffmpeg -filters | grep libvmaf`)
- `libvmaf` version 2.0+ for the NEG model (`vmaf_v0.6.1neg`)
- Python 3.9+
- `pip install scenedetect`

## Usage

```bash
# Default: 10 MB target
./smallifier.py input.mp4 output.mp4

# Custom size target (the only required tunable)
./smallifier.py input.mp4 output.mp4 --target-mb 25

# Slower preset, better compression — typical use is on a long batch overnight
./smallifier.py input.mp4 output.mp4 --preset 1

# 8-bit output for legacy decoder compatibility (default is 10-bit)
./smallifier.py input.mp4 output.mp4 --bit-depth 8

# Override the adaptive audio bitrate
./smallifier.py input.mp4 output.mp4 --audio-bps 64000

# Persistent workdir — survives interruption, resumable
./smallifier.py input.mp4 output.mp4 --workdir /var/tmp/enc-run
```

## What it tells you

```
Source: 1920x1080 @ 59.94fps, 100.1s, pix=yuv420p, audio=True
Budget: 10.0 MB total → video ~9.62 MB (avg 805 kbps)
Output: 1280x720 @ 30.0fps, 10-bit (0.029 avg bpp)
Workdir: /tmp/smart_av1_xxxxxx (fresh)
Pool: 32 × 8 threads
Encoding audio at 40 kbps Opus [adaptive]...
  audio: 488.3 KiB (4.8% of 10.0 MB target); video budget now 9.62 MB
Detecting scenes...
  24 chunks, median 2.46s
Ladder: 640x360 → 768x432 → 960x540 → 1280x720

Phase 1/4: 640x360 — 24 chunks × 4 CRFs
Phase 2/4: 768x432 — 24 chunks × 4 CRFs
  3 chunk(s) stopped expanding after this rung
Phase 3/4: 960x540 — 21 chunks × 4 CRFs
  8 chunk(s) stopped expanding after this rung
Phase 4/4: 1280x720 — 13 chunks × 4 CRFs

Exploration done. Avg 9.7 Pareto points/chunk; max rung explored: 1280x720

Allocating to 9.62 MB video budget...
Total: 9.51 MB (video 9.02 + audio 0.49) / 10.0 MB target
VMAF: duration-weighted mean 82.43, range [76.21, 91.05]
Chunk choices:
  640x360: 4 chunk(s)
  768x432: 11 chunk(s)
  960x540: 7 chunk(s)
  1280x720: 2 chunk(s)
  CRFs: {30: 2, 35: 5, 40: 9, 45: 6, 50: 2}

Done → output.mp4 (9.55 MB, under target by 0.45 MB / -4.5%)
```

The per-chunk summary at the end shows which resolutions and CRFs the allocator selected — useful for sanity-checking the algorithm against your content. If you see every chunk picking the same resolution and CRF, the trial grid is too coarse for your input; if you see wide variation, the algorithm is doing real work.

## Resume

Pass `--workdir` and the next run will reuse on-disk work that matches the parameters from the previous run:

```bash
./smallifier.py input.mp4 output.mp4 --workdir /var/tmp/enc-run
# ... Ctrl-C, machine reboot, anything ...
./smallifier.py input.mp4 output.mp4 --workdir /var/tmp/enc-run
# Picks up exactly where it left off.
```

The startup sweep ffprobe-validates every cached trial in parallel and re-encodes any that were corrupted by an abnormal exit. Changing encoding parameters (`--bit-depth`, `--preset`, etc.) invalidates the workdir by design — the script refuses to mix encodes with inconsistent parameters and explains which parameter changed.

## Caveats

This is an offline batch tool. Trial-encoding 200–400 candidate operating points takes real time — 5–10 minutes on a many-core server, 30–60 minutes on a laptop. Not for interactive use.

The size target is a hard ceiling, not a goal. Outputs typically land 2–5% under target. If your use case demands hitting a specific size precisely (rather than under it), you'll want a different tool.

VMAF correlates well with perceptual quality for most content but is not a guarantee. Particular failure modes — banding on smooth gradients, certain motion artifacts, encoder-specific blocking patterns — can be underweighted. The output reports the VMAF range; if the spread between worst and best chunks is wide, eyeball the result before shipping it.

Non-16:9 source aspect ratios are currently handled with forced scaling and no aspect preservation; non-16:9 inputs come out distorted. Plumbing aspect-aware scaling is straightforward but not done.

The tool is single-machine. There's no built-in distributed mode; the pool is bounded by `cpu_count()`. For larger workloads, look at [Av1an](https://github.com/master-of-zen/Av1an), which is a mature production-grade encoder orchestrator and shares much of the same design philosophy.

## License

MIT.
