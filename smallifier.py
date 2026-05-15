#!/usr/bin/env python3
"""
Smallifier AV1 encoder — adaptive per-chunk convex-hull search with slope-equalization
allocation. Maximizes duration-weighted mean VMAF subject to a size budget.

Algorithm:
  1. Pick output resolution from budget feasibility (largest ladder rung where
     a generous chunk allocation could plausibly hit 0.05+ bpp).
  2. For each chunk, walk up the resolution ladder. At each rung, sample
     several CRFs and merge results into the chunk's accumulated Pareto
     frontier. Stop expanding when this rung adds nothing to the frontier
     OR its smallest-size encode exceeds the chunk's max plausible allocation.
  3. Globally allocate via greedy slope-equalization: repeatedly apply the
     upgrade (across all chunks) with the best ΔVMAF × duration / Δbytes
     ratio, until no further upgrade fits the budget.
  4. Concatenate winners with `-c copy`, mux Opus audio.

No VMAF threshold — trust VMAF, accept whatever quality the budget buys.

Resume: pass --workdir PATH. Trial encodes are cached by deterministic
filename; matching params on rerun → cached work reused.

Requires: ffmpeg with libsvtav1, libopus, libvmaf compiled in;
Python: `pip install scenedetect`.
"""

import os
import sys
import json
import shutil
import tempfile
import argparse
import subprocess
from dataclasses import dataclass, field
from typing import Optional
from multiprocessing import Pool, cpu_count

from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector


# ---------------- DEFAULTS ---------------- #
DEFAULT_TARGET_MB     = 10.0
DEFAULT_BIT_DEPTH     = 10
DEFAULT_SVT_PRESET    = 2

# Audio bitrate selection. None of these are user-tunable from CLI; the
# user instead passes --audio-bps to override entirely.
# Default behavior: scale audio with total budget, clamped to a
# perceptually meaningful Opus range.
AUDIO_FRACTION = 0.05      # 5% of total budget
AUDIO_BPS_MIN  = 24_000    # Opus intelligibility floor for mixed content
AUDIO_BPS_MAX  = 96_000    # near-transparent; above this is diminishing returns

# Resolution ladder. Encoder walks up from smallest; capped at output res.
# All exactly 16:9, mod-16 width, all mod-4+ height.
RESOLUTION_LADDER = [(640, 360), (768, 432), (960, 540), (1280, 720), (1920, 1080)]

# Per-resolution CRF samples. Spans the useful AV1 range. CRF 56 covers
# the tight-budget tail at the smallest resolution; without it, tight
# budgets can end up with all-chunks-at-cheapest-option still exceeding
# the size target (no smaller fallback exists).
CRFS_PER_RESOLUTION = [30, 40, 50, 56]

# Emergency probes — run only at the smallest ladder rung, only if the
# cheapest Pareto combination still exceeds the size budget after the
# main exploration phases. CRF 63 is AV1's hard maximum.
EMERGENCY_CRFS = [60, 63]

# Output resolution feasibility threshold (bits per pixel).
# 0.05 bpp is a rough AV1 floor for fast-motion content to land VMAF ~75+.
OUTPUT_BPP_FLOOR     = 0.05
OUTPUT_BPP_HEADROOM  = 3.0   # a chunk can get up to 3× fair share

# A chunk's max plausible allocation = (fair share) × this factor.
CHUNK_BUDGET_HEADROOM = 5.0

CONTAINER_OVERHEAD   = 0.97
VALID_MP4_MIN_BYTES  = 1024


# ---------------- SHELL / PROBE HELPERS ---------------- #

def run(cmd):
    subprocess.run(cmd, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def ffprobe_duration(path):
    out = subprocess.check_output([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", path,
    ])
    return float(out.strip())

def ffprobe_video(path):
    out = subprocess.check_output([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,pix_fmt",
        "-of", "json", path,
    ])
    s = json.loads(out)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    fps = float(num) / float(den) if float(den) else 30.0
    return s["width"], s["height"], fps, s["pix_fmt"]

def has_audio(path):
    out = subprocess.check_output([
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_type",
        "-of", "default=noprint_wrappers=1:nokey=1", path,
    ]).strip()
    return out == b"audio"

def filesize(path):
    return os.path.getsize(path)

def ffprobe_valid(path):
    try:
        subprocess.run(["ffprobe", "-v", "error", path],
                       check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=15)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return False


# ---------------- OUTPUT RESOLUTION SELECTION ---------------- #

def choose_output_resolution(video_budget_bytes, total_duration, fps,
                             src_w, src_h):
    """Largest ladder rung where a generous chunk allocation could plausibly
    hit OUTPUT_BPP_FLOOR. Caps at source resolution. Returns smallest rung
    if budget is too tight for any rung — caller can warn."""
    avg_bps = video_budget_bytes * 8 / max(total_duration, 0.001)
    max_chunk_bps = avg_bps * OUTPUT_BPP_HEADROOM

    feasible = []
    for (w, h) in RESOLUTION_LADDER:
        if w > src_w or h > src_h:
            continue
        bpp = max_chunk_bps / (w * h * fps)
        if bpp >= OUTPUT_BPP_FLOOR:
            feasible.append((w, h))

    if not feasible:
        # Pick the smallest source-fitting rung anyway.
        for (w, h) in RESOLUTION_LADDER:
            if w <= src_w and h <= src_h:
                return (w, h), False
        return RESOLUTION_LADDER[0], False
    return max(feasible, key=lambda wh: wh[0] * wh[1]), True


def choose_audio_bps(target_mb, duration):
    """Default audio bitrate from total budget. AUDIO_FRACTION of total,
    clamped to [AUDIO_BPS_MIN, AUDIO_BPS_MAX]. User --audio-bps overrides."""
    total_bps = target_mb * 1024 * 1024 * 8 / max(duration, 0.001)
    return int(max(AUDIO_BPS_MIN, min(AUDIO_BPS_MAX, total_bps * AUDIO_FRACTION)))


# ---------------- SCENE DETECTION ---------------- #

def detect_scenes(input_file, min_scene_sec=1.0):
    video = open_video(input_file)
    sm = SceneManager()
    sm.add_detector(ContentDetector(
        threshold=27.0,
        min_scene_len=max(1, int(min_scene_sec * video.frame_rate)),
    ))
    sm.detect_scenes(video, show_progress=False)
    scenes = sm.get_scene_list()
    if not scenes:
        return [(0.0, ffprobe_duration(input_file))]
    return [(s.seconds, e.seconds) for s, e in scenes]


# ---------------- ENCODE / VMAF ---------------- #

def svt_params(threads):
    return (
        "tune=0"
        ":enable-tf=0"
        ":enable-qm=1"
        ":qm-min=0:qm-max=15"
        ":aq-mode=2"
        ":scd=1"
        ":enable-overlays=1"
        ":film-grain=0"
        f":lp={threads}"
    )

def _scaling_vf(pre_w, pre_h, out_w, out_h, out_fps):
    vf = []
    if (pre_w, pre_h) != (out_w, out_h):
        vf.append(f"scale={pre_w}:{pre_h}:flags=lanczos")
        vf.append(f"scale={out_w}:{out_h}:flags=lanczos")
    else:
        vf.append(f"scale={out_w}:{out_h}:flags=lanczos")
    vf.append(f"fps={out_fps}")
    vf.append("atadenoise=0.02:0.02:0.02")
    return ",".join(vf)

def svt_cmd(src, start, dur, out_path, pre_w, pre_h, out_w, out_h, out_fps,
            crf, preset, threads, bit_depth):
    pix_fmt = "yuv420p10le" if bit_depth == 10 else "yuv420p"
    return [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start:.6f}",
        "-i", src,
        "-t", f"{dur:.6f}",
        "-vf", _scaling_vf(pre_w, pre_h, out_w, out_h, out_fps),
        "-c:v", "libsvtav1",
        "-preset", str(preset),
        "-svtav1-params", svt_params(threads),
        "-pix_fmt", pix_fmt,
        "-crf", str(crf),
        "-b:v", "0",
        "-g", "240",
        "-an",
        out_path,
    ]

def _parse_vmaf_json(path):
    with open(path) as f:
        data = json.load(f)
    return float(data["pooled_metrics"]["vmaf"]["mean"])

def compute_vmaf(distorted, reference, start, dur, out_w, out_h, out_fps, threads):
    log_path = distorted + ".vmaf.json"
    if os.path.exists(log_path):
        try:
            return _parse_vmaf_json(log_path)
        except (json.JSONDecodeError, KeyError, OSError):
            try: os.unlink(log_path)
            except OSError: pass
    filt = (
        f"[0:v]scale={out_w}:{out_h}:flags=lanczos,fps={out_fps},"
        f"setpts=PTS-STARTPTS[d];"
        f"[1:v]scale={out_w}:{out_h}:flags=lanczos,fps={out_fps},"
        f"setpts=PTS-STARTPTS[r];"
        f"[d][r]libvmaf=model=version=vmaf_v0.6.1neg:n_threads={threads}"
        f":log_fmt=json:log_path={log_path}"
    )
    subprocess.run([
        "ffmpeg", "-loglevel", "error",
        "-i", distorted,
        "-ss", f"{start:.6f}",
        "-i", reference,
        "-t", f"{dur:.6f}",
        "-lavfi", filt,
        "-f", "null", "-",
    ], check=True)
    return _parse_vmaf_json(log_path)


# ---------------- DATA TYPES ---------------- #

@dataclass(order=True)
class OpPoint:
    size: int
    vmaf: float = field(compare=False)
    pre_w: int = field(compare=False)
    pre_h: int = field(compare=False)
    crf: int = field(compare=False)
    path: str = field(compare=False)


@dataclass
class Chunk:
    id: int
    start: float
    end: float
    pareto: list = field(default_factory=list)
    choice_idx: int = 0

    @property
    def duration(self):
        return self.end - self.start

    @property
    def choice(self) -> OpPoint:
        return self.pareto[self.choice_idx]


def pareto_filter(points):
    """Size-ascending, VMAF strictly increasing."""
    pts = sorted(points, key=lambda p: (p.size, -p.vmaf))
    out = []
    best_vmaf = -1.0
    for p in pts:
        if p.vmaf > best_vmaf:
            out.append(p)
            best_vmaf = p.vmaf
    return out


# ---------------- TRIAL WORKER ---------------- #

def run_trial(job):
    """Encode + VMAF for one (chunk, prefilter_res, CRF) trial. Idempotent."""
    out_path = job["out"]
    try:
        if (not os.path.exists(out_path)
                or os.path.getsize(out_path) < VALID_MP4_MIN_BYTES):
            cmd = svt_cmd(
                job["src"], job["start"], job["dur"], out_path,
                job["pre_w"], job["pre_h"],
                job["out_w"], job["out_h"], job["out_fps"],
                job["crf"], job["preset"], job["threads"], job["bit_depth"],
            )
            run(cmd)
        job["size"] = filesize(out_path)
        job["vmaf"] = compute_vmaf(
            out_path, job["src"], job["start"], job["dur"],
            job["out_w"], job["out_h"], job["out_fps"],
            threads=max(2, job["threads"] // 2),
        )
        job["ok"] = True
    except (subprocess.CalledProcessError, OSError, KeyError, json.JSONDecodeError):
        job["ok"] = False
    return job


# ---------------- WORKDIR / RESUME ---------------- #

def build_params(args, input_file, out_w, out_h):
    st = os.stat(input_file)
    return {
        "input_file": os.path.abspath(input_file),
        "input_size": st.st_size,
        "input_mtime": int(st.st_mtime),
        "bit_depth": args.bit_depth,
        "preset": args.preset,
        "out_w": out_w,
        "out_h": out_h,
        "ladder": [list(r) for r in RESOLUTION_LADDER],
        "crfs": list(CRFS_PER_RESOLUTION),
        "emergency_crfs": list(EMERGENCY_CRFS),
        "audio_bps": args.audio_bps,
    }

def setup_workdir(args, params):
    if args.workdir:
        wd = args.workdir
        os.makedirs(wd, exist_ok=True)
        pp = os.path.join(wd, "params.json")
        if os.path.exists(pp):
            with open(pp) as f:
                existing = json.load(f)
            if existing != params:
                diffs = []
                for k in set(existing) | set(params):
                    if existing.get(k) != params.get(k):
                        diffs.append(f"  {k}: {existing.get(k)!r} → {params.get(k)!r}")
                sys.exit(f"Workdir {wd} has different params:\n"
                         + "\n".join(diffs) +
                         f"\nUse a different --workdir or delete {wd}.")
            return wd, False, True
        with open(pp, "w") as f:
            json.dump(params, f, indent=2)
        return wd, False, False
    wd = tempfile.mkdtemp(prefix="smart_av1_", dir="/tmp")
    with open(os.path.join(wd, "params.json"), "w") as f:
        json.dump(params, f, indent=2)
    return wd, True, False

def _validate_one(path):
    if os.path.getsize(path) < VALID_MP4_MIN_BYTES:
        return path
    return None if ffprobe_valid(path) else path

def sweep_invalid(workdir, pool_size):
    paths = [os.path.join(workdir, f) for f in os.listdir(workdir)
             if f.endswith(".mp4") and f.startswith("c")]
    if not paths:
        return 0
    with Pool(pool_size) as p:
        results = p.map(_validate_one, paths)
    bad = [r for r in results if r]
    for b in bad:
        try: os.unlink(b)
        except OSError: pass
        v = b + ".vmaf.json"
        if os.path.exists(v):
            try: os.unlink(v)
            except OSError: pass
    return len(bad)


# ---------------- ALLOCATOR (slope equalization) ---------------- #

def allocate(chunks, video_budget):
    """Greedy slope-equalization. Starts every chunk at its smallest Pareto
    point. Repeatedly applies the (chunk, upgrade) pair with the highest
    ΔVMAF × duration / Δsize, subject to fitting in the remaining budget.
    Terminates when no candidate upgrade fits."""
    for c in chunks:
        c.choice_idx = 0
    while True:
        total = sum(c.choice.size for c in chunks)
        remaining = video_budget - total
        if remaining <= 0:
            break
        best = None
        best_value = -1.0
        for c in chunks:
            if c.choice_idx >= len(c.pareto) - 1:
                continue
            cur = c.pareto[c.choice_idx]
            nxt = c.pareto[c.choice_idx + 1]
            dsize = nxt.size - cur.size
            if dsize <= 0 or dsize > remaining:
                continue
            dvmaf = nxt.vmaf - cur.vmaf
            if dvmaf <= 0:
                continue
            value = dvmaf * c.duration / dsize
            if value > best_value:
                best_value = value
                best = c
        if best is None:
            break
        best.choice_idx += 1


# ---------------- MAIN PIPELINE ---------------- #

def parse_args():
    p = argparse.ArgumentParser(
        description="Smallifier AV1 encoder. Maximizes weighted-mean VMAF at a size target."
    )
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("-s", "--target-mb", type=float, default=DEFAULT_TARGET_MB,
                   help=f"Target output size in MB (default {DEFAULT_TARGET_MB}).")
    p.add_argument("-a", "--audio-bps", type=int, default=None,
                   help=f"Opus audio bitrate in bps. Default: adaptive — "
                        f"{int(AUDIO_FRACTION*100)} percent of total budget, "
                        f"clamped to [{AUDIO_BPS_MIN}, {AUDIO_BPS_MAX}].")
    p.add_argument("-b", "--bit-depth", type=int, choices=[8, 10],
                   default=DEFAULT_BIT_DEPTH,
                   help=f"Output bit depth (default {DEFAULT_BIT_DEPTH}).")
    p.add_argument("--preset", type=int, default=DEFAULT_SVT_PRESET,
                   help=f"SVT-AV1 preset 0-13 (default {DEFAULT_SVT_PRESET}).")
    p.add_argument("--workdir", type=str, default=None,
                   help="Persistent workdir for resume. Default: /tmp/smart_av1_*.")
    p.add_argument("--workers", type=int, default=0,
                   help="Worker count (0 = auto).")
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.input):
        sys.exit(f"Input not found: {args.input}")

    # ---- probe ----
    duration = ffprobe_duration(args.input)
    src_w, src_h, src_fps, src_pix = ffprobe_video(args.input)
    input_has_audio = has_audio(args.input)
    out_fps = min(30.0, src_fps)
    print(f"Source: {src_w}x{src_h} @ {src_fps:.2f}fps, "
          f"{duration:.1f}s, pix={src_pix}, audio={input_has_audio}")

    # ---- budget / output resolution decision ----
    budget = args.target_mb * 1024 * 1024 * CONTAINER_OVERHEAD

    # Resolve audio bitrate: user override wins, else adaptive scaling.
    audio_explicit = args.audio_bps is not None
    if not audio_explicit:
        args.audio_bps = choose_audio_bps(args.target_mb, duration)

    # Rough audio reservation; refined after encoding.
    audio_reserve = (args.audio_bps * duration / 8) if input_has_audio else 0
    video_budget = max(0, budget - audio_reserve)

    (out_w, out_h), feasible = choose_output_resolution(
        video_budget, duration, out_fps, src_w, src_h
    )
    avg_bps = video_budget * 8 / duration
    avg_bpp = avg_bps / (out_w * out_h * out_fps)
    print(f"Budget: {args.target_mb} MB total → video ~{video_budget/1024/1024:.2f} MB "
          f"(avg {avg_bps/1000:.0f} kbps)")
    print(f"Output: {out_w}x{out_h} @ {out_fps}fps, {args.bit_depth}-bit "
          f"({avg_bpp:.3f} avg bpp)")
    if not feasible:
        print(f"  ! budget gives <{OUTPUT_BPP_FLOOR} bpp even at smallest "
              f"resolution; quality will be poor regardless of CRF.")

    # ---- workdir ----
    params = build_params(args, args.input, out_w, out_h)
    workdir, cleanup_on_success, resuming = setup_workdir(args, params)
    print(f"Workdir: {workdir} ({'resuming' if resuming else 'fresh'})")

    try:
        # ---- pool sizing ----
        cores = args.workers if args.workers else cpu_count()
        threads_per_job = max(2, min(8, cores // 16))
        pool_size = max(1, cores // threads_per_job)
        print(f"Pool: {pool_size} × {threads_per_job} threads")

        if resuming:
            bad = sweep_invalid(workdir, pool_size)
            if bad:
                print(f"  removed {bad} corrupt cached file(s)")

        # ---- audio ----
        audio_path = None
        audio_size = 0
        if input_has_audio:
            audio_path = os.path.join(workdir, "audio.opus")
            if not (os.path.exists(audio_path) and ffprobe_valid(audio_path)):
                print(f"Encoding audio at {args.audio_bps/1000:.0f} kbps Opus"
                      + (" [user override]" if audio_explicit else " [adaptive]")
                      + "...")
                run([
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-i", args.input,
                    "-vn", "-c:a", "libopus",
                    "-b:a", str(args.audio_bps), "-vbr", "on",
                    audio_path,
                ])
            audio_size = filesize(audio_path)
            video_budget = max(0, budget - audio_size)
            audio_pct = audio_size / (args.target_mb * 1024 * 1024) * 100
            print(f"  audio: {audio_size/1024:.1f} KiB "
                  f"({audio_pct:.1f}% of {args.target_mb} MB target)"
                  f"; video budget now {video_budget/1024/1024:.2f} MB")

        # ---- scene detection ----
        print("Detecting scenes...")
        scene_ranges = detect_scenes(args.input)
        chunks = [Chunk(id=i, start=s, end=e)
                  for i, (s, e) in enumerate(scene_ranges) if e - s >= 0.2]
        if not chunks:
            sys.exit("No usable scenes.")
        med = sorted(c.duration for c in chunks)[len(chunks)//2]
        print(f"  {len(chunks)} chunks, median {med:.2f}s")

        # ---- ladder bounded by output resolution and source ----
        active_ladder = [(w, h) for (w, h) in RESOLUTION_LADDER
                         if w <= out_w and h <= out_h and w <= src_w and h <= src_h]
        if not active_ladder:
            active_ladder = [(out_w, out_h)]
        print(f"Ladder: {' → '.join(f'{w}x{h}' for w,h in active_ladder)}")

        # ---- adaptive per-chunk expansion across phases ----
        # Each phase encodes one resolution for all currently active chunks.
        # Chunks drop out of the active set when their next-rung exploration
        # would be (a) dominated by accumulated frontier OR (b) too expensive
        # for their max plausible allocation.
        active = list(chunks)
        for phase_idx, (pre_w, pre_h) in enumerate(active_ladder):
            if not active:
                break
            jobs = []
            for c in active:
                for crf in CRFS_PER_RESOLUTION:
                    out = os.path.join(
                        workdir, f"c{c.id:04d}_{pre_w}x{pre_h}_crf{crf}.mp4"
                    )
                    jobs.append({
                        "chunk_id": c.id, "out": out,
                        "src": args.input, "start": c.start, "dur": c.duration,
                        "pre_w": pre_w, "pre_h": pre_h,
                        "out_w": out_w, "out_h": out_h, "out_fps": out_fps,
                        "crf": crf, "preset": args.preset,
                        "threads": threads_per_job, "bit_depth": args.bit_depth,
                    })
            cached = sum(
                1 for j in jobs
                if os.path.exists(j["out"])
                and os.path.getsize(j["out"]) >= VALID_MP4_MIN_BYTES
                and os.path.exists(j["out"] + ".vmaf.json")
            )
            print(f"\nPhase {phase_idx+1}/{len(active_ladder)}: {pre_w}x{pre_h} "
                  f"— {len(active)} chunks × {len(CRFS_PER_RESOLUTION)} CRFs"
                  + (f" ({cached} cached)" if cached else ""))
            with Pool(pool_size) as p:
                results = p.map(run_trial, jobs)
            results = [r for r in results if r.get("ok")]

            # Group by chunk; update frontiers; decide who continues.
            by_chunk = {}
            for r in results:
                by_chunk.setdefault(r["chunk_id"], []).append(r)

            next_active = []
            for c in active:
                rs = by_chunk.get(c.id, [])
                if not rs:
                    continue  # all failed; stop expanding
                new_points = [OpPoint(
                    size=r["size"], vmaf=r["vmaf"],
                    pre_w=r["pre_w"], pre_h=r["pre_h"], crf=r["crf"],
                    path=r["out"],
                ) for r in rs]
                new_pareto = pareto_filter(c.pareto + new_points)
                # Did this rung contribute anything?
                added = [p for p in new_pareto if (p.pre_w, p.pre_h) == (pre_w, pre_h)]
                c.pareto = new_pareto
                if not added:
                    continue  # dominated; larger rungs will be too
                # Can this chunk plausibly afford the next rung?
                max_chunk_bytes = (video_budget
                                   * (c.duration / duration)
                                   * CHUNK_BUDGET_HEADROOM)
                min_size_here = min(r["size"] for r in rs)
                if min_size_here > max_chunk_bytes:
                    continue  # can't even afford this rung's cheapest
                next_active.append(c)
            dropped = len(active) - len(next_active)
            if dropped:
                print(f"  {dropped} chunk(s) stopped expanding after this rung")
            active = next_active

        # ---- summary of exploration ----
        avg_pts  = sum(len(c.pareto) for c in chunks) / len(chunks)
        max_res  = max((max((p.pre_w, p.pre_h) for p in c.pareto) for c in chunks),
                       default=(0,0))
        print(f"\nExploration done. Avg {avg_pts:.1f} Pareto points/chunk; "
              f"max rung explored: {max_res[0]}x{max_res[1]}")

        # ---- emergency probe at high CRFs if cheapest-feasible exceeds budget ----
        cheapest_total = sum(c.pareto[0].size for c in chunks)
        if cheapest_total > video_budget and EMERGENCY_CRFS:
            over_mb = (cheapest_total - video_budget) / 1024 / 1024
            print(f"\n! Cheapest combination ({cheapest_total/1024/1024:.2f} MB) "
                  f"exceeds video budget by {over_mb:.2f} MB.")
            print(f"  Emergency probe: CRFs {EMERGENCY_CRFS} at "
                  f"{active_ladder[0][0]}x{active_ladder[0][1]}...")
            smallest_w, smallest_h = active_ladder[0]
            ej = []
            for c in chunks:
                for crf in EMERGENCY_CRFS:
                    out = os.path.join(
                        workdir, f"c{c.id:04d}_{smallest_w}x{smallest_h}_crf{crf}.mp4"
                    )
                    ej.append({
                        "chunk_id": c.id, "out": out,
                        "src": args.input, "start": c.start, "dur": c.duration,
                        "pre_w": smallest_w, "pre_h": smallest_h,
                        "out_w": out_w, "out_h": out_h, "out_fps": out_fps,
                        "crf": crf, "preset": args.preset,
                        "threads": threads_per_job, "bit_depth": args.bit_depth,
                    })
            with Pool(pool_size) as p:
                er = p.map(run_trial, ej)
            er = [r for r in er if r.get("ok")]
            by_chunk = {}
            for r in er:
                by_chunk.setdefault(r["chunk_id"], []).append(r)
            for c in chunks:
                rs = by_chunk.get(c.id, [])
                if rs:
                    new_points = [OpPoint(
                        size=r["size"], vmaf=r["vmaf"],
                        pre_w=r["pre_w"], pre_h=r["pre_h"], crf=r["crf"],
                        path=r["out"],
                    ) for r in rs]
                    c.pareto = pareto_filter(c.pareto + new_points)
            new_cheapest = sum(c.pareto[0].size for c in chunks)
            if new_cheapest <= video_budget:
                print(f"  resolved: cheapest now "
                      f"{new_cheapest/1024/1024:.2f} MB (fits)")
            else:
                still_over = (new_cheapest - video_budget) / 1024 / 1024
                print(f"  ! still over by {still_over:.2f} MB even at CRF 63 "
                      f"on smallest rung; output will exceed target")

        # ---- allocate ----
        print(f"\nAllocating to {video_budget/1024/1024:.2f} MB video budget...")
        allocate(chunks, video_budget)

        total_v = sum(c.choice.size for c in chunks)
        total = total_v + audio_size
        mean_vmaf = (sum(c.choice.vmaf * c.duration for c in chunks)
                     / sum(c.duration for c in chunks))
        min_vmaf = min(c.choice.vmaf for c in chunks)
        max_vmaf = max(c.choice.vmaf for c in chunks)
        print(f"Total: {total/1024/1024:.2f} MB "
              f"(video {total_v/1024/1024:.2f} + audio {audio_size/1024/1024:.2f}) "
              f"/ {args.target_mb} MB")
        print(f"VMAF: duration-weighted mean {mean_vmaf:.2f}, "
              f"range [{min_vmaf:.2f}, {max_vmaf:.2f}]")

        # ---- per-chunk diagnostic ----
        from collections import Counter
        res_counts = Counter((c.choice.pre_w, c.choice.pre_h) for c in chunks)
        crf_counts = Counter(c.choice.crf for c in chunks)
        print("Chunk choices:")
        for (w, h), n in sorted(res_counts.items()):
            print(f"  {w}x{h}: {n} chunk(s)")
        print(f"  CRFs: {dict(sorted(crf_counts.items()))}")

        # ---- concat + mux ----
        listfile = os.path.join(workdir, "concat.txt")
        with open(listfile, "w") as f:
            for c in sorted(chunks, key=lambda c: c.id):
                f.write(f"file '{os.path.abspath(c.choice.path)}'\n")
        video_only = os.path.join(workdir, "video_only.mp4")
        run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0",
            "-i", listfile,
            "-c", "copy", video_only,
        ])
        mux_cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", video_only]
        if audio_path:
            mux_cmd += ["-i", audio_path, "-map", "0:v:0", "-map", "1:a:0"]
        mux_cmd += ["-c", "copy", "-movflags", "+faststart", args.output]
        run(mux_cmd)

        final_mb = filesize(args.output) / (1024 * 1024)
        delta_mb = final_mb - args.target_mb
        delta_pct = delta_mb / args.target_mb * 100
        if delta_mb > 0.01:
            print(f"\nDone → {args.output} ({final_mb:.2f} MB, "
                  f"OVER target by {delta_mb:.2f} MB / {delta_pct:+.1f}%)")
            print("  ! Size constraint violated. Budget too tight even at maximum compression.")
        elif delta_mb < -0.01:
            print(f"\nDone → {args.output} ({final_mb:.2f} MB, "
                  f"under target by {-delta_mb:.2f} MB / {delta_pct:+.1f}%)")
        else:
            print(f"\nDone → {args.output} ({final_mb:.2f} MB, on target)")

        if cleanup_on_success:
            shutil.rmtree(workdir, ignore_errors=True)
        else:
            print(f"Workdir kept at {workdir}")
    except BaseException:
        if not cleanup_on_success:
            print(f"\nWorkdir preserved at {workdir}")
        raise


if __name__ == "__main__":
    main()
