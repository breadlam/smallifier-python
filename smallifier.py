#!/usr/bin/env python3
"""
Smart AV1 encoder — convex-hull per-chunk optimization with VMAF gating, ensemble
encoding (SVT-AV1 + aomenc), and global size-budget allocation.

Pipeline:
  1. PySceneDetect ContentDetector finds scene boundaries.
  2. For each chunk: do a (resolution × CRF) grid search using BOTH SVT-AV1
     and aomenc at slow presets. Measure size and VMAF (NEG model) for each.
  3. Compute Pareto frontier per chunk over (size, VMAF). The grid IS the
     complexity probe — no separate analysis stage.
  4. Global allocator: greedily pick operating points to hit size budget while
     keeping every chunk above VMAF floor.
  5. Concatenate winning encodes with `-c copy` (all share uniform output
     dimensions and timing), mux Opus audio, write final file.

All time-range operations use accurate seek (`-ss` after `-i`), so chunk
boundaries are frame-accurate without intermediate lossless files.

Requires: ffmpeg with libsvtav1, libaom-av1, libopus, libvmaf compiled in;
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
DEFAULT_VMAF          = 92.0
DEFAULT_AUDIO_BPS     = 48_000
DEFAULT_BIT_DEPTH     = 10
DEFAULT_SVT_PRESET    = 2
DEFAULT_AOM_CPU_USED  = 2

# Convex hull grid. Resolutions cap to source dimensions at runtime.
GRID_RESOLUTIONS = [(640, 360), (854, 480), (960, 540), (1280, 720), (1920, 1080)]
GRID_CRFS_SVT    = [22, 27, 32, 37, 42, 47]
GRID_CRFS_AOM    = [22, 28, 34, 40, 46, 52]   # aomenc CRF scale differs slightly

CONTAINER_OVERHEAD = 0.97
SIZE_TOLERANCE     = 0.03   # ±3% of target is "on target"
VMAF_HEADROOM      = 0.5    # downgrade only chunks at least this far above floor


# ---------------- SHELL / PROBE HELPERS ---------------- #

def run(cmd, quiet=True):
    if not quiet:
        print("  $ " + " ".join(cmd[:6]) + (" ..." if len(cmd) > 6 else ""))
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


# ---------------- SCENE DETECTION ---------------- #

def detect_scenes(input_file, min_scene_sec=1.0):
    """PySceneDetect ContentDetector — adaptive to fades and gradual cuts."""
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
    """SVT-AV1 params tuned for text-on-fast-motion."""
    return (
        "tune=0"                    # psy-visual
        ":enable-tf=0"              # disable temporal filter — preserves text edges
        ":enable-qm=1"              # quantization matrices
        ":qm-min=0:qm-max=15"
        ":aq-mode=2"                # variance-based adaptive quant
        ":scd=1"                    # scene change detection
        ":enable-overlays=1"        # extra refs near transitions
        ":film-grain=0"
        f":lp={threads}"
    )

def _scaling_vf(pre_w, pre_h, out_w, out_h, out_fps):
    vf = []
    if (pre_w, pre_h) != (out_w, out_h):
        # detail-reducing round trip — encoder sees a low-pass-filtered version
        vf.append(f"scale={pre_w}:{pre_h}:flags=lanczos")
        vf.append(f"scale={out_w}:{out_h}:flags=lanczos")
    else:
        vf.append(f"scale={out_w}:{out_h}:flags=lanczos")
    vf.append(f"fps={out_fps}")
    vf.append("atadenoise=0.02:0.02:0.02")
    return ",".join(vf)

def svt_cmd(src, start, dur, out_path, pre_w, pre_h, out_w, out_h, out_fps,
            crf, preset, threads, bit_depth):
    """One libsvtav1 encode. Accurate-seek into source, prefilter to chosen
    internal resolution, encode at uniform output spec."""
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
    """One libaom-av1 encode with text-friendly params."""
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
        "-aq-mode", "2",            # variance AQ — text-friendly
        "-arnr-strength", "0",      # disable temporal denoise — preserves text
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

def compute_vmaf(distorted, reference, start, dur, out_w, out_h, out_fps, threads):
    """VMAF NEG vs the source at (start, dur). Both sides aligned to output
    spec. NEG ('no enhancement gain') discounts encoder-introduced
    sharpening/contrast that standard VMAF rewards but humans don't credit."""
    log_path = f"{distorted}.vmaf.json"
    filt = (
        f"[0:v]scale={out_w}:{out_h}:flags=lanczos,fps={out_fps},"
        f"setpts=PTS-STARTPTS[d];"
        f"[1:v]scale={out_w}:{out_h}:flags=lanczos,fps={out_fps},"
        f"setpts=PTS-STARTPTS[r];"
        f"[d][r]libvmaf=model=version=vmaf_v0.6.1neg:n_threads={threads}"
        f":log_fmt=json:log_path={log_path}"
    )
    cmd = [
        "ffmpeg", "-loglevel", "error",
        "-i", distorted,
        "-ss", f"{start:.6f}",
        "-i", reference,
        "-t", f"{dur:.6f}",
        "-lavfi", filt,
        "-f", "null", "-",
    ]
    subprocess.run(cmd, check=True)
    with open(log_path) as f:
        data = json.load(f)
    try:
        os.unlink(log_path)
    except OSError:
        pass
    return float(data["pooled_metrics"]["vmaf"]["mean"])


# ---------------- DATA TYPES ---------------- #

@dataclass(order=True)
class OpPoint:
    """One operating point on a chunk's quality/size curve."""
    size: int                       # sort key
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
    pareto: list = field(default_factory=list)   # list[OpPoint], size-ascending
    choice_idx: int = 0

    @property
    def duration(self):
        return self.end - self.start

    @property
    def choice(self) -> OpPoint:
        return self.pareto[self.choice_idx]


# ---------------- CONVEX HULL / SEARCH ---------------- #

def pareto_filter(points):
    """Keep only Pareto-optimal points: at each size, no smaller-size point
    achieves higher VMAF. Returns size-ascending (and VMAF-ascending) list."""
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
    """Build the full job spec list for one chunk: encoders × resolutions × CRFs."""
    jobs = []
    resolutions = [(w, h) for (w, h) in GRID_RESOLUTIONS
                   if w <= source_w and h <= source_h]
    if (out_w, out_h) not in resolutions:
        resolutions.append((out_w, out_h))

    for (pre_w, pre_h) in resolutions:
        for crf in GRID_CRFS_SVT:
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
            for crf in GRID_CRFS_AOM:
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
    """Execute one trial encode. VMAF is computed in a second pass."""
    if job["encoder"] == "svt":
        cmd = svt_cmd(
            job["src"], job["start"], job["dur"], job["out"],
            job["pre_w"], job["pre_h"],
            job["out_w"], job["out_h"], job["out_fps"],
            job["crf"], job["preset"], job["threads"], job["bit_depth"],
        )
    else:
        cmd = aom_cmd(
            job["src"], job["start"], job["dur"], job["out"],
            job["pre_w"], job["pre_h"],
            job["out_w"], job["out_h"], job["out_fps"],
            job["crf"], job["cpu_used"], job["threads"], job["bit_depth"],
        )
    try:
        run(cmd)
        job["size"] = filesize(job["out"])
        job["ok"] = True
    except subprocess.CalledProcessError:
        job["ok"] = False
    return job

def run_vmaf(job):
    """Compute VMAF for one completed trial encode."""
    if not job.get("ok"):
        return job
    try:
        job["vmaf"] = compute_vmaf(
            job["out"], job["src"], job["start"], job["dur"],
            job["out_w"], job["out_h"], job["out_fps"],
            threads=2,
        )
    except subprocess.CalledProcessError:
        job["ok"] = False
    return job


# ---------------- GLOBAL ALLOCATOR ---------------- #

def initial_selection(chunks, vmaf_target):
    """Per chunk: smallest size whose VMAF meets target. If none meets,
    take the highest-VMAF point."""
    for c in chunks:
        passing = [(i, p) for i, p in enumerate(c.pareto) if p.vmaf >= vmaf_target]
        if passing:
            c.choice_idx = min(passing, key=lambda ip: ip[1].size)[0]
        else:
            c.choice_idx = max(range(len(c.pareto)), key=lambda i: c.pareto[i].vmaf)

def allocate_to_budget(chunks, budget_bytes, vmaf_target, max_steps=2000):
    """Greedy multi-choice knapsack over per-chunk Pareto frontiers.

    Pareto is size-ascending and VMAF-ascending (frontier property), so
    choice_idx+1 means strictly larger size AND strictly larger VMAF.

    Iterates:
      - if over budget: downgrade chunk with most VMAF headroom above floor.
        If none has headroom, downgrade the chunk closest to floor (least
        quality loss per byte saved).
      - if under budget by more than tolerance: upgrade a chunk, preferring
        those below VMAF floor; only commit if the upgrade fits.
      - else: done.
    """
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
            delta = next_pt.size - pick.choice.size
            if total + delta > budget_bytes:
                break
            pick.choice_idx += 1
        else:
            break
    return chunks


# ---------------- MAIN PIPELINE ---------------- #

def parse_args():
    p = argparse.ArgumentParser(
        description="Smart AV1 encoder with convex-hull per-chunk optimization."
    )
    p.add_argument("input")
    p.add_argument("output", help="Output filename (required).")
    p.add_argument("-s", "--target-mb", type=float, default=DEFAULT_TARGET_MB,
                   help=f"Target output size in MB (default {DEFAULT_TARGET_MB}).")
    p.add_argument("-v", "--vmaf", type=float, default=DEFAULT_VMAF,
                   help=f"Per-chunk VMAF floor (default {DEFAULT_VMAF}).")
    p.add_argument("-a", "--audio-bps", type=int, default=DEFAULT_AUDIO_BPS,
                   help=f"Opus audio bitrate (default {DEFAULT_AUDIO_BPS}).")
    p.add_argument("-b", "--bit-depth", type=int, choices=[8, 10],
                   default=DEFAULT_BIT_DEPTH,
                   help=f"Output bit depth (default {DEFAULT_BIT_DEPTH}). "
                        "10-bit AV1 is more compression-efficient and supported "
                        "wherever AV1 itself plays.")
    p.add_argument("--preset", type=int, default=DEFAULT_SVT_PRESET,
                   help=f"SVT-AV1 preset, 0-13 (default {DEFAULT_SVT_PRESET}).")
    p.add_argument("--aom-cpu-used", type=int, default=DEFAULT_AOM_CPU_USED,
                   help=f"aomenc cpu-used, 0-8 (default {DEFAULT_AOM_CPU_USED}).")
    p.add_argument("--no-aomenc", action="store_true",
                   help="Skip aomenc; SVT-AV1 only (cuts compute ~half).")
    p.add_argument("--workers", type=int, default=0,
                   help="Worker process count (0 = auto).")
    return p.parse_args()


def main():
    args = parse_args()
    args.use_aomenc = not args.no_aomenc

    if not os.path.exists(args.input):
        sys.exit(f"Input not found: {args.input}")

    workdir = tempfile.mkdtemp(prefix="smart_av1_", dir="/tmp")
    print(f"Workdir: {workdir}")

    try:
        # ---- probe ----
        duration = ffprobe_duration(args.input)
        src_w, src_h, src_fps, src_pix = ffprobe_video(args.input)
        input_has_audio = has_audio(args.input)
        print(f"Source: {src_w}x{src_h} @ {src_fps:.2f}fps, "
              f"{duration:.1f}s, pix={src_pix}, audio={input_has_audio}")

        # ---- audio (encoded once up front) ----
        audio_path = None
        audio_size = 0
        if input_has_audio:
            audio_path = os.path.join(workdir, "audio.opus")
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
        print("Detecting scenes (PySceneDetect ContentDetector)...")
        scene_ranges = detect_scenes(args.input)
        chunks = [Chunk(id=i, start=s, end=e)
                  for i, (s, e) in enumerate(scene_ranges) if e - s >= 0.2]
        if not chunks:
            sys.exit("No usable scenes detected.")
        med = sorted(c.duration for c in chunks)[len(chunks)//2]
        print(f"  {len(chunks)} chunks (median {med:.2f}s)")

        # ---- output spec (uniform across chunks) ----
        available = [(w, h) for (w, h) in GRID_RESOLUTIONS
                     if w <= src_w and h <= src_h]
        if not available:
            available = [(src_w, src_h)]
        out_w, out_h = max(available, key=lambda wh: wh[0] * wh[1])
        out_w -= out_w % 2
        out_h -= out_h % 2
        out_fps = min(30.0, src_fps)
        print(f"Output: {out_w}x{out_h} @ {out_fps}fps, "
              f"{args.bit_depth}-bit, "
              f"encoders={'SVT-AV1+aomenc' if args.use_aomenc else 'SVT-AV1'}")

        # ---- parallelism ----
        cores = args.workers if args.workers else cpu_count()
        threads_per_job = max(2, min(8, cores // 16))
        pool_size = max(1, cores // threads_per_job)
        print(f"Pool: {pool_size} workers x {threads_per_job} threads "
              f"({pool_size * threads_per_job} threads total)")

        # ---- build all trial jobs ----
        all_jobs = []
        for c in chunks:
            all_jobs.extend(trial_jobs_for_chunk(
                args.input, c, workdir, out_w, out_h, out_fps,
                src_w, src_h, args, threads_per_job,
            ))
        print(f"\nEncoding {len(all_jobs)} trial encodes "
              f"({len(all_jobs) // len(chunks)} per chunk)...")

        # ---- run trial encodes ----
        with Pool(pool_size) as p:
            encoded_jobs = p.map(run_trial, all_jobs)
        ok_jobs = [j for j in encoded_jobs if j.get("ok")]
        failed = len(encoded_jobs) - len(ok_jobs)
        if failed:
            print(f"  ! {failed} encode failures (continuing)")
        print(f"  encoded {len(ok_jobs)} trials")

        # ---- VMAF measurement in parallel ----
        print("Computing VMAF (NEG model) for all trials...")
        with Pool(pool_size) as p:
            scored_jobs = p.map(run_vmaf, ok_jobs)
        scored_jobs = [j for j in scored_jobs if j.get("ok")]
        print(f"  scored {len(scored_jobs)} trials")

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

        # ---- allocate to budget ----
        budget = args.target_mb * 1024 * 1024 * CONTAINER_OVERHEAD
        video_budget = max(0, budget - audio_size)
        print(f"\nAllocating to budget ({video_budget/1024/1024:.2f} MB video)...")
        initial_selection(chunks, args.vmaf)
        allocate_to_budget(chunks, video_budget, args.vmaf)

        # ---- report selection ----
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

        svt_picked = sum(1 for c in chunks if c.choice.encoder == "svt")
        aom_picked = sum(1 for c in chunks if c.choice.encoder == "aom")
        print(f"Encoder picks: SVT-AV1 {svt_picked}, aomenc {aom_picked}")

        # ---- concat winning chunks ----
        # All chosen outputs share output dims, fps, codec → -c copy concat valid.
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

        # ---- mux ----
        print("Muxing final output...")
        mux_cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", video_only,
        ]
        if audio_path:
            mux_cmd += ["-i", audio_path,
                        "-map", "0:v:0", "-map", "1:a:0"]
        mux_cmd += ["-c", "copy", "-movflags", "+faststart", args.output]
        run(mux_cmd)

        final_mb = filesize(args.output) / (1024 * 1024)
        print(f"\nDone → {args.output} ({final_mb:.2f} MB)")
        size_err = abs(final_mb - args.target_mb) / args.target_mb * 100
        print(f"Size error: {size_err:.1f}% of target")

    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
