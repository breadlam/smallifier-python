#!/usr/bin/env python3
"""
Smart AV1 encoder — convex-hull per-chunk optimization with VMAF gating,
optional ensemble encoding (SVT-AV1 + aomenc), and global size-budget allocation.

Pipeline:
  1. PySceneDetect ContentDetector finds scene boundaries.
  2. For each chunk: do a (resolution × CRF) grid search with SVT-AV1
     (and aomenc if --aomenc). Measure size and VMAF (NEG model) for each.
  3. Compute Pareto frontier per chunk over (size, VMAF). The grid IS the
     complexity probe — no separate analysis stage.
  4. Global allocator: greedily pick operating points to hit size budget while
     keeping every chunk above VMAF floor.
  5. Concatenate winning encodes with `-c copy` (uniform output spec across
     chunks), mux Opus audio, write final file.

All time-range operations use accurate seek (`-ss` after `-i`), so chunk
boundaries are frame-accurate without intermediate lossless files.

Resume: pass --workdir PATH. If PATH exists with matching params, work that's
already on disk (trial mp4s + their vmaf json sidecars) is skipped. Without
--workdir a temp dir is used and cleaned on success.

Requires: ffmpeg with libsvtav1, libaom-av1, libopus, libvmaf compiled in;
Python: `pip install scenedetect`.
"""

import os
import sys
import json
import shutil
import hashlib
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
DEFAULT_VMAF          = 90.0
DEFAULT_AUDIO_BPS     = 48_000
DEFAULT_BIT_DEPTH     = 10
DEFAULT_SVT_PRESET    = 2
DEFAULT_AOM_CPU_USED  = 2

# Default grid: 3 resolutions × 5 CRFs. CRFs centered on the aggressive end
# where ~10 MB / ~90 s targets actually land. Resolutions chosen for clean
# encoder properties rather than display convention: all exactly 16:9,
# mod-16 where possible, with clean Lanczos downscale ratios against the
# 720p/1080p output target (e.g. 540p = exactly 1/2 of 1080p). 1080p is
# omitted from the fast grid — at aggressive budgets it doesn't make the
# convex hull, so trialing it just wastes compute.
GRID_RESOLUTIONS_FAST = [(768, 432), (960, 540), (1280, 720)]
GRID_CRFS_SVT_FAST    = [28, 33, 38, 43, 48]
GRID_CRFS_AOM_FAST    = [28, 34, 40, 46, 52]

# Dense grid: 5 resolutions × 6 CRFs, for when --dense-grid is set. Adds
# 360p below and 1080p above for budgets where they might land on the hull.
GRID_RESOLUTIONS_DENSE = [(640, 360), (768, 432), (960, 540), (1280, 720), (1920, 1080)]
GRID_CRFS_SVT_DENSE    = [22, 27, 32, 37, 42, 47]
GRID_CRFS_AOM_DENSE    = [22, 28, 34, 40, 46, 52]

CONTAINER_OVERHEAD = 0.97
SIZE_TOLERANCE     = 0.03
VMAF_HEADROOM      = 0.5

VALID_MP4_MIN_BYTES = 1024


# ---------------- SHELL / PROBE HELPERS ---------------- #

def run(cmd, quiet=True):
    subprocess.run(
        cmd, check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL if quiet else None,
    )

def ffprobe_duration(path):
    out = subprocess.check_output([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
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
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]).strip()
    return out == b"audio"

def filesize(path):
    return os.path.getsize(path)

def ffprobe_valid(path):
    """Quick validity check: file parses without errors."""
    try:
        subprocess.run(
            ["ffprobe", "-v", "error", path],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=15,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return False


# ---------------- WORKDIR / RESUME ---------------- #

def file_signature(path):
    """Fast file identity — size + mtime. Good enough; hashing is too slow
    for the multi-GB sources this script targets."""
    st = os.stat(path)
    return {"size": st.st_size, "mtime": int(st.st_mtime)}

def build_params(args, input_file):
    """Parameters that, if changed, invalidate cached trial encodes.

    target_mb and vmaf threshold are deliberately excluded — they only affect
    allocation, so changing them lets you re-allocate without re-encoding."""
    sig = file_signature(input_file)
    return {
        "input_file": os.path.abspath(input_file),
        "input_size": sig["size"],
        "input_mtime": sig["mtime"],
        "bit_depth": args.bit_depth,
        "preset": args.preset,
        "aom_cpu_used": args.aom_cpu_used,
        "use_aomenc": args.use_aomenc,
        "resolutions": [list(r) for r in args.resolutions],
        "crfs_svt": list(args.crfs_svt),
        "crfs_aom": list(args.crfs_aom) if args.use_aomenc else [],
        "audio_bps": args.audio_bps,
    }

def setup_workdir(args):
    """Decide on workdir path; check param compatibility for resume.
    Returns (workdir, will_be_cleaned_on_success, resuming)."""
    params = build_params(args, args.input)

    if args.workdir:
        wd = args.workdir
        cleanup = False
        os.makedirs(wd, exist_ok=True)
        params_path = os.path.join(wd, "params.json")
        if os.path.exists(params_path):
            with open(params_path) as f:
                existing = json.load(f)
            if existing != params:
                # Pinpoint what's different for a useful error.
                diffs = []
                for k in set(existing) | set(params):
                    if existing.get(k) != params.get(k):
                        diffs.append(f"  {k}: {existing.get(k)!r} → {params.get(k)!r}")
                sys.exit(
                    f"Workdir {wd} was created with different parameters. "
                    f"Refusing to mix encodes.\n"
                    + "\n".join(diffs) +
                    f"\n\nEither delete {wd}, or use a different --workdir."
                )
            return wd, False, True
        else:
            with open(params_path, "w") as f:
                json.dump(params, f, indent=2)
            return wd, False, False
    else:
        wd = tempfile.mkdtemp(prefix="smart_av1_", dir="/tmp")
        with open(os.path.join(wd, "params.json"), "w") as f:
            json.dump(params, f, indent=2)
        return wd, True, False

def _validate_one(path):
    """Worker for parallel resume-sweep validation. Returns path if invalid."""
    if os.path.getsize(path) < VALID_MP4_MIN_BYTES:
        return path
    return None if ffprobe_valid(path) else path

def sweep_invalid(workdir, pool_size):
    """Find and delete corrupt .mp4 trial files (and their .vmaf.json
    sidecars). Catches encodes killed mid-write on the prior run."""
    paths = [os.path.join(workdir, f) for f in os.listdir(workdir)
             if f.endswith(".mp4") and f.startswith("c")]
    if not paths:
        return 0
    with Pool(pool_size) as p:
        results = p.map(_validate_one, paths)
    bad = [r for r in results if r]
    for b in bad:
        try:
            os.unlink(b)
        except OSError:
            pass
        v = b + ".vmaf.json"
        if os.path.exists(v):
            try:
                os.unlink(v)
            except OSError:
                pass
    return len(bad)


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
    return [(s.get_seconds(), e.get_seconds()) for s, e in scenes]


# ---------------- ENCODE PARAM BUILDERS ---------------- #

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

def aom_cmd(src, start, dur, out_path, pre_w, pre_h, out_w, out_h, out_fps,
            crf, cpu_used, threads, bit_depth):
    pix_fmt = "yuv420p10le" if bit_depth == 10 else "yuv420p"
    return [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start:.6f}",
        "-i", src,
        "-t", f"{dur:.6f}",
        "-vf", _scaling_vf(pre_w, pre_h, out_w, out_h, out_fps),
        "-c:v", "libaom-av1",
        "-cpu-used", str(cpu_used),
        "-crf", str(crf),
        "-b:v", "0",
        "-aq-mode", "2",
        "-arnr-strength", "0",
        "-tune-content", "default",
        "-row-mt", "1",
        "-tile-columns", "1",
        "-pix_fmt", pix_fmt,
        "-g", "240",
        "-threads", str(threads),
        "-an",
        out_path,
    ]


# ---------------- VMAF (NEG model) ---------------- #

def _parse_vmaf_json(path):
    with open(path) as f:
        data = json.load(f)
    return float(data["pooled_metrics"]["vmaf"]["mean"])

def compute_vmaf(distorted, reference, start, dur, out_w, out_h, out_fps, threads):
    log_path = distorted + ".vmaf.json"
    # Reuse cached result if present and parseable.
    if os.path.exists(log_path):
        try:
            return _parse_vmaf_json(log_path)
        except (json.JSONDecodeError, KeyError, OSError):
            try:
                os.unlink(log_path)
            except OSError:
                pass

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
    encoder: str = field(compare=False)
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


# ---------------- CONVEX HULL ---------------- #

def pareto_filter(points):
    pts = sorted(points, key=lambda p: p.size)
    out = []
    best_vmaf = -1.0
    for p in pts:
        if p.vmaf > best_vmaf:
            out.append(p)
            best_vmaf = p.vmaf
    return out

def trial_jobs_for_chunk(src, chunk, workdir, out_w, out_h, out_fps,
                         source_w, source_h, args, threads):
    jobs = []
    resolutions = [(w, h) for (w, h) in args.resolutions
                   if w <= source_w and h <= source_h]
    if (out_w, out_h) not in resolutions:
        resolutions.append((out_w, out_h))

    for (pre_w, pre_h) in resolutions:
        for crf in args.crfs_svt:
            out = os.path.join(
                workdir, f"c{chunk.id:04d}_svt_{pre_w}x{pre_h}_crf{crf}.mp4"
            )
            jobs.append({
                "chunk_id": chunk.id, "encoder": "svt",
                "pre_w": pre_w, "pre_h": pre_h, "crf": crf,
                "out": out, "src": src, "start": chunk.start, "dur": chunk.duration,
                "out_w": out_w, "out_h": out_h, "out_fps": out_fps,
                "preset": args.preset, "threads": threads,
                "bit_depth": args.bit_depth,
            })
        if args.use_aomenc:
            for crf in args.crfs_aom:
                out = os.path.join(
                    workdir, f"c{chunk.id:04d}_aom_{pre_w}x{pre_h}_crf{crf}.mp4"
                )
                jobs.append({
                    "chunk_id": chunk.id, "encoder": "aom",
                    "pre_w": pre_w, "pre_h": pre_h, "crf": crf,
                    "out": out, "src": src, "start": chunk.start, "dur": chunk.duration,
                    "out_w": out_w, "out_h": out_h, "out_fps": out_fps,
                    "cpu_used": args.aom_cpu_used, "threads": threads,
                    "bit_depth": args.bit_depth,
                })
    return jobs

def run_trial(job):
    """Encode + measure VMAF. Idempotent: skips work already on disk.

    Combined into one worker call so a worker handling a "resumed" trial
    can do both the encode-skip and the VMAF-fetch in one round-trip,
    minimizing IPC overhead when most work is cached."""
    out_path = job["out"]
    encode_needed = (
        not os.path.exists(out_path)
        or os.path.getsize(out_path) < VALID_MP4_MIN_BYTES
    )
    try:
        if encode_needed:
            if job["encoder"] == "svt":
                cmd = svt_cmd(
                    job["src"], job["start"], job["dur"], out_path,
                    job["pre_w"], job["pre_h"],
                    job["out_w"], job["out_h"], job["out_fps"],
                    job["crf"], job["preset"], job["threads"], job["bit_depth"],
                )
            else:
                cmd = aom_cmd(
                    job["src"], job["start"], job["dur"], out_path,
                    job["pre_w"], job["pre_h"],
                    job["out_w"], job["out_h"], job["out_fps"],
                    job["crf"], job["cpu_used"], job["threads"], job["bit_depth"],
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


# ---------------- GLOBAL ALLOCATOR ---------------- #

def initial_selection(chunks, vmaf_target):
    for c in chunks:
        passing = [(i, p) for i, p in enumerate(c.pareto) if p.vmaf >= vmaf_target]
        if passing:
            c.choice_idx = min(passing, key=lambda ip: ip[1].size)[0]
        else:
            c.choice_idx = max(range(len(c.pareto)),
                               key=lambda i: c.pareto[i].vmaf)

def allocate_to_budget(chunks, budget_bytes, vmaf_target, max_steps=2000):
    for _ in range(max_steps):
        total = sum(c.choice.size for c in chunks)

        if total > budget_bytes:
            cands = [c for c in chunks
                     if c.choice_idx > 0
                     and c.choice.vmaf >= vmaf_target + VMAF_HEADROOM]
            if not cands:
                cands = [c for c in chunks if c.choice_idx > 0]
                if not cands:
                    break
            pick = max(cands, key=lambda c: c.choice.vmaf - vmaf_target)
            pick.choice_idx -= 1

        elif total < budget_bytes * (1.0 - SIZE_TOLERANCE):
            below = [c for c in chunks
                     if c.choice.vmaf < vmaf_target
                     and c.choice_idx < len(c.pareto) - 1]
            if below:
                pick = min(below, key=lambda c: c.choice.vmaf)
            else:
                cands = [c for c in chunks if c.choice_idx < len(c.pareto) - 1]
                if not cands:
                    break
                pick = min(cands, key=lambda c: c.choice.vmaf)
            next_pt = pick.pareto[pick.choice_idx + 1]
            if total + (next_pt.size - pick.choice.size) > budget_bytes:
                break
            pick.choice_idx += 1
        else:
            break


# ---------------- MAIN PIPELINE ---------------- #

def parse_args():
    p = argparse.ArgumentParser(
        description="Smart AV1 encoder with convex-hull per-chunk optimization."
    )
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("-s", "--target-mb", type=float, default=DEFAULT_TARGET_MB,
                   help=f"Target output size in MB (default {DEFAULT_TARGET_MB}).")
    p.add_argument("-v", "--vmaf", type=float, default=DEFAULT_VMAF,
                   help=f"Per-chunk VMAF floor (default {DEFAULT_VMAF}).")
    p.add_argument("-a", "--audio-bps", type=int, default=DEFAULT_AUDIO_BPS,
                   help=f"Opus audio bitrate (default {DEFAULT_AUDIO_BPS}).")
    p.add_argument("-b", "--bit-depth", type=int, choices=[8, 10],
                   default=DEFAULT_BIT_DEPTH,
                   help=f"Output bit depth (default {DEFAULT_BIT_DEPTH}).")
    p.add_argument("--preset", type=int, default=DEFAULT_SVT_PRESET,
                   help=f"SVT-AV1 preset 0-13 (default {DEFAULT_SVT_PRESET}).")
    p.add_argument("--aom-cpu-used", type=int, default=DEFAULT_AOM_CPU_USED,
                   help=f"aomenc cpu-used 0-8 (default {DEFAULT_AOM_CPU_USED}).")
    p.add_argument("--aomenc", action="store_true",
                   help="Add aomenc to the trial ensemble (roughly doubles "
                        "compute; chunk-by-chunk the better encoder is kept).")
    p.add_argument("--dense-grid", action="store_true",
                   help="Use a 5-resolution × 6-CRF grid instead of the "
                        "default 3 × 5 (2x more trials).")
    p.add_argument("--workdir", type=str, default=None,
                   help="Explicit work directory (persistent across runs). "
                        "Required for resume; with matching params, existing "
                        "trial encodes and VMAF scores are reused.")
    p.add_argument("--workers", type=int, default=0,
                   help="Worker process count (0 = auto).")
    return p.parse_args()


def main():
    args = parse_args()
    args.use_aomenc = args.aomenc

    # bind grid
    if args.dense_grid:
        args.resolutions = GRID_RESOLUTIONS_DENSE
        args.crfs_svt    = GRID_CRFS_SVT_DENSE
        args.crfs_aom    = GRID_CRFS_AOM_DENSE
    else:
        args.resolutions = GRID_RESOLUTIONS_FAST
        args.crfs_svt    = GRID_CRFS_SVT_FAST
        args.crfs_aom    = GRID_CRFS_AOM_FAST

    if not os.path.exists(args.input):
        sys.exit(f"Input not found: {args.input}")

    workdir, cleanup_on_success, resuming = setup_workdir(args)
    print(f"Workdir: {workdir}  ({'resuming' if resuming else 'fresh'})")

    try:
        # ---- probe ----
        duration = ffprobe_duration(args.input)
        src_w, src_h, src_fps, src_pix = ffprobe_video(args.input)
        input_has_audio = has_audio(args.input)
        print(f"Source: {src_w}x{src_h} @ {src_fps:.2f}fps, "
              f"{duration:.1f}s, pix={src_pix}, audio={input_has_audio}")

        # ---- pool sizing ----
        cores = args.workers if args.workers else cpu_count()
        threads_per_job = max(2, min(8, cores // 16))
        pool_size = max(1, cores // threads_per_job)
        print(f"Pool: {pool_size} workers × {threads_per_job} threads "
              f"({pool_size * threads_per_job} threads total)")

        # ---- resume sweep: nuke partial/corrupt trial files ----
        if resuming:
            print("Validating existing trial files...")
            bad = sweep_invalid(workdir, pool_size)
            if bad:
                print(f"  removed {bad} corrupt file(s); they'll be re-encoded")

        # ---- audio ----
        audio_path = None
        audio_size = 0
        if input_has_audio:
            audio_path = os.path.join(workdir, "audio.opus")
            if not (os.path.exists(audio_path)
                    and os.path.getsize(audio_path) > 0
                    and ffprobe_valid(audio_path)):
                print("Encoding audio...")
                run([
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-i", args.input,
                    "-vn", "-c:a", "libopus",
                    "-b:a", str(args.audio_bps),
                    "-vbr", "on",
                    audio_path,
                ])
            audio_size = filesize(audio_path)
            print(f"  audio: {audio_size/1024:.1f} KiB")

        # ---- scene detection ----
        # Always re-run; PySceneDetect is deterministic given the same input.
        print("Detecting scenes (PySceneDetect ContentDetector)...")
        scene_ranges = detect_scenes(args.input)
        chunks = [Chunk(id=i, start=s, end=e)
                  for i, (s, e) in enumerate(scene_ranges) if e - s >= 0.2]
        if not chunks:
            sys.exit("No usable scenes detected.")
        med = sorted(c.duration for c in chunks)[len(chunks)//2]
        print(f"  {len(chunks)} chunks (median {med:.2f}s)")

        # ---- output spec ----
        available = [(w, h) for (w, h) in args.resolutions
                     if w <= src_w and h <= src_h]
        if not available:
            available = [(src_w, src_h)]
        out_w, out_h = max(available, key=lambda wh: wh[0] * wh[1])
        out_w -= out_w % 2
        out_h -= out_h % 2
        out_fps = min(30.0, src_fps)
        encoder_label = "SVT-AV1+aomenc" if args.use_aomenc else "SVT-AV1"
        grid_label = "dense" if args.dense_grid else "fast"
        print(f"Output: {out_w}x{out_h} @ {out_fps}fps, "
              f"{args.bit_depth}-bit, encoders={encoder_label}, grid={grid_label}")

        # ---- build all trial jobs ----
        all_jobs = []
        for c in chunks:
            all_jobs.extend(trial_jobs_for_chunk(
                args.input, c, workdir, out_w, out_h, out_fps,
                src_w, src_h, args, threads_per_job,
            ))
        # Count what's already done.
        cached = sum(
            1 for j in all_jobs
            if os.path.exists(j["out"])
            and os.path.getsize(j["out"]) >= VALID_MP4_MIN_BYTES
            and os.path.exists(j["out"] + ".vmaf.json")
        )
        if resuming and cached:
            print(f"  {cached}/{len(all_jobs)} trials already complete on disk")
        print(f"\nRunning {len(all_jobs)} trials "
              f"({len(all_jobs) // len(chunks)} per chunk)...")

        # ---- run trials (idempotent — skips cached work per-job) ----
        with Pool(pool_size) as p:
            scored_jobs = p.map(run_trial, all_jobs)
        scored_jobs = [j for j in scored_jobs if j.get("ok")]
        failed = len(all_jobs) - len(scored_jobs)
        if failed:
            print(f"  ! {failed} trial(s) failed")
        print(f"  {len(scored_jobs)} trials succeeded")

        # ---- build per-chunk Pareto frontiers ----
        chunk_by_id = {c.id: c for c in chunks}
        raw_per_chunk = {c.id: [] for c in chunks}
        for j in scored_jobs:
            raw_per_chunk[j["chunk_id"]].append(OpPoint(
                size=j["size"], vmaf=j["vmaf"], encoder=j["encoder"],
                pre_w=j["pre_w"], pre_h=j["pre_h"], crf=j["crf"], path=j["out"],
            ))
        for cid, points in raw_per_chunk.items():
            if not points:
                sys.exit(f"Chunk {cid} has no successful trials.")
            chunk_by_id[cid].pareto = pareto_filter(points)
        avg_pts = sum(len(c.pareto) for c in chunks) / len(chunks)
        print(f"  avg {avg_pts:.1f} Pareto points per chunk")

        # ---- allocate ----
        budget = args.target_mb * 1024 * 1024 * CONTAINER_OVERHEAD
        video_budget = max(0, budget - audio_size)
        print(f"\nAllocating to budget ({video_budget/1024/1024:.2f} MB video)...")
        initial_selection(chunks, args.vmaf)
        allocate_to_budget(chunks, video_budget, args.vmaf)

        total_video = sum(c.choice.size for c in chunks)
        total = total_video + audio_size
        below = [c for c in chunks if c.choice.vmaf < args.vmaf]
        mean_vmaf = sum(c.choice.vmaf for c in chunks) / len(chunks)
        min_vmaf = min(c.choice.vmaf for c in chunks)
        print(f"Total: {total/1024/1024:.2f} MB "
              f"(video {total_video/1024/1024:.2f} + "
              f"audio {audio_size/1024/1024:.2f}) "
              f"/ {args.target_mb} MB target")
        print(f"VMAF: mean {mean_vmaf:.2f}, min {min_vmaf:.2f}, "
              f"{len(below)}/{len(chunks)} below {args.vmaf}")
        if below:
            print(f"  ! {len(below)} chunk(s) unable to meet VMAF at this budget")
        if args.use_aomenc:
            svt_picked = sum(1 for c in chunks if c.choice.encoder == "svt")
            aom_picked = sum(1 for c in chunks if c.choice.encoder == "aom")
            print(f"Encoder picks: SVT-AV1 {svt_picked}, aomenc {aom_picked}")

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
            "-c", "copy",
            video_only,
        ])
        print("Muxing final output...")
        mux_cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", video_only]
        if audio_path:
            mux_cmd += ["-i", audio_path, "-map", "0:v:0", "-map", "1:a:0"]
        mux_cmd += ["-c", "copy", "-movflags", "+faststart", args.output]
        run(mux_cmd)

        final_mb = filesize(args.output) / (1024 * 1024)
        print(f"\nDone → {args.output} ({final_mb:.2f} MB)")
        size_err = abs(final_mb - args.target_mb) / args.target_mb * 100
        print(f"Size error: {size_err:.1f}% of target")

        if cleanup_on_success:
            shutil.rmtree(workdir, ignore_errors=True)
        else:
            print(f"Workdir kept at {workdir} "
                  f"(rerun with same --workdir to reuse)")
    except BaseException:
        # On any failure (including KeyboardInterrupt), preserve workdir so
        # the run can be resumed.
        if not cleanup_on_success:
            print(f"\nWorkdir preserved at {workdir} for resume")
        raise


if __name__ == "__main__":
    main()
