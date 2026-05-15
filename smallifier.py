#!/usr/bin/env python3
"""
Smart AV1 encoder with scene-aware bitrate allocation, VMAF gating, and audio.

Designed for aggressive size targets on fast-motion content where text legibility
matters. The pipeline: detect scenes, chunk + analyze (motion + edges), pick
per-scene encode params, encode in parallel, measure size + VMAF, adjust, repeat.
"""

import os
import sys
import json
import shutil
import tempfile
import argparse
import subprocess
from multiprocessing import Pool, cpu_count

# ---------------- CONFIG ---------------- #
DEFAULT_TARGET_MB     = 10.0
DEFAULT_VMAF          = 92.0
DEFAULT_AUDIO_BPS     = 48_000
DEFAULT_MAX_ITER      = 6
SCENE_THRESHOLD       = 0.4
MIN_BITRATE_BPS       = 80_000   # floor per scene to avoid pathological encodes
CONTAINER_OVERHEAD    = 0.97     # leave headroom for MP4 container/index
SIZE_TOLERANCE        = 0.05     # ±5% of target counts as size-OK

# ---------------- SHELL HELPERS ---------------- #

def run(cmd, quiet=False):
    if not quiet:
        preview = " ".join(cmd[:6]) + (" ..." if len(cmd) > 6 else "")
        print(f"  $ {preview}")
    subprocess.run(
        cmd, check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL if quiet else None,
    )

def ffprobe_duration(path):
    out = subprocess.check_output([
        "ffprobe","-v","error",
        "-show_entries","format=duration",
        "-of","default=noprint_wrappers=1:nokey=1",
        path
    ])
    return float(out.strip())

def ffprobe_video(path):
    out = subprocess.check_output([
        "ffprobe","-v","error",
        "-select_streams","v:0",
        "-show_entries","stream=width,height,r_frame_rate",
        "-of","json",
        path
    ])
    s = json.loads(out)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    fps = float(num) / float(den) if float(den) else 30.0
    return s["width"], s["height"], fps

def has_audio(path):
    out = subprocess.check_output([
        "ffprobe","-v","error",
        "-select_streams","a:0",
        "-show_entries","stream=codec_type",
        "-of","default=noprint_wrappers=1:nokey=1",
        path
    ]).strip()
    return out == b"audio"

def filesize(path):
    return os.path.getsize(path)

# ---------------- SCENE DETECTION ---------------- #

def detect_scenes(input_file, threshold=SCENE_THRESHOLD):
    cmd = [
        "ffmpeg","-i",input_file,
        "-filter:v",f"select='gt(scene,{threshold})',showinfo",
        "-f","null","-"
    ]
    proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True)
    times = []
    for line in proc.stderr:
        if "pts_time:" in line:
            for part in line.split():
                if part.startswith("pts_time:"):
                    try:
                        times.append(float(part.split(":")[1]))
                    except ValueError:
                        pass
    proc.wait()
    return sorted(times)

# ---------------- ANALYSIS ---------------- #

def analyze_scene(input_file, start, duration):
    # motion proxy: average per-frame interframe diff luma
    motion_cmd = [
        "ffmpeg","-ss",str(start),"-t",str(duration),
        "-i",input_file,
        "-vf","tblend=all_mode=difference,signalstats",
        "-f","null","-"
    ]
    proc = subprocess.Popen(motion_cmd, stderr=subprocess.PIPE, text=True)
    motion_vals = []
    for line in proc.stderr:
        if "YAVG" in line:
            try:
                motion_vals.append(float(line.split("YAVG:")[1].split()[0]))
            except (ValueError, IndexError):
                pass
    proc.wait()
    motion = (sum(motion_vals) / len(motion_vals)) if motion_vals else 0.1

    # edge proxy: noisy but consistent — counts log lines from edgedetect
    edge_cmd = [
        "ffmpeg","-ss",str(start),"-t",str(duration),
        "-i",input_file,
        "-vf","edgedetect",
        "-f","null","-"
    ]
    proc = subprocess.Popen(edge_cmd, stderr=subprocess.PIPE, text=True)
    edge_count = sum(1 for _ in proc.stderr)
    proc.wait()
    edges = min(edge_count / 5000, 1.0)

    return motion, edges

# ---------------- DECISION ENGINE ---------------- #

def decide(motion, edges):
    # Resolution: text-heavy (high edges) keeps more pixels; calm scenes lose them.
    if edges > 0.25:
        res = (960, 540)
    elif motion > 0.5:
        res = (768, 432)
    else:
        res = (640, 360)

    if motion < 0.15:
        fps = 18
    elif motion < 0.3:
        fps = 24
    else:
        fps = 30

    mode = "vbr" if motion > 0.5 else "crf"
    crf  = 32 if edges > 0.2 else 36  # slightly lower than original — protects text
    return res, fps, mode, crf

# ---------------- ENCODING ---------------- #

def encode_scene(args):
    scene, out_w, out_h, out_fps, threads = args
    infile  = scene["file"]
    outfile = scene["out"]

    vf = (f"fps={out_fps}"
          f",scale={out_w}:{out_h}:flags=lanczos"
          ",atadenoise=0.02:0.02:0.02")
    if scene["edges"] > 0.25:
        vf += ",unsharp=5:5:0.8:3:3:0.4"

    svt_params = f"tune=0:film-grain=0:lp={threads}"

    base = [
        "ffmpeg","-y","-loglevel","error",
        "-i", infile,
        "-vf", vf,
        "-c:v","libsvtav1",
        "-preset","5",
        "-svtav1-params", svt_params,
        "-pix_fmt","yuv420p",
        "-g","240",
        "-an",
    ]
    if scene["mode"] == "crf":
        cmd = base + ["-crf", str(scene["crf"]), "-b:v","0", outfile]
    else:
        cmd = base + ["-b:v", str(scene["bitrate"]), outfile]
    run(cmd, quiet=True)
    return outfile

# ---------------- VMAF ---------------- #

def compute_vmaf(distorted, reference, out_w, out_h, out_fps, threads):
    """
    Distorted has been scaled/fps-changed during encode; reference is the
    untouched source chunk. We force both through identical scale + fps so
    VMAF compares like-for-like.
    """
    log_path = f"{distorted}.vmaf.json"
    filt = (
        f"[0:v]scale={out_w}:{out_h}:flags=lanczos,fps={out_fps},setpts=PTS-STARTPTS[d];"
        f"[1:v]scale={out_w}:{out_h}:flags=lanczos,fps={out_fps},setpts=PTS-STARTPTS[r];"
        f"[d][r]libvmaf=n_threads={threads}:log_fmt=json:log_path={log_path}"
    )
    cmd = [
        "ffmpeg","-loglevel","error",
        "-i", distorted, "-i", reference,
        "-lavfi", filt,
        "-f","null","-"
    ]
    subprocess.run(cmd, check=True)
    with open(log_path) as f:
        data = json.load(f)
    return float(data["pooled_metrics"]["vmaf"]["mean"])

def vmaf_worker(args):
    scene, out_w, out_h, out_fps, threads = args
    return scene["id"], compute_vmaf(
        scene["out"], scene["file"], out_w, out_h, out_fps, threads
    )

# ---------------- CLI ---------------- #

def parse_args():
    p = argparse.ArgumentParser(
        description="Smart AV1 encoder: scene-aware bitrate allocation + VMAF gating + audio.",
    )
    p.add_argument("input", help="Input video file.")
    p.add_argument("output", help="Output filename (required, no default).")
    p.add_argument("-s","--target-mb", type=float, default=DEFAULT_TARGET_MB,
                   help=f"Target output size in megabytes (default {DEFAULT_TARGET_MB}).")
    p.add_argument("-v","--vmaf", type=float, default=DEFAULT_VMAF,
                   help=f"Minimum acceptable per-scene VMAF (default {DEFAULT_VMAF}). "
                        "92 is a balanced default for aggressive compression of "
                        "text-bearing fast-motion content.")
    p.add_argument("-a","--audio-bps", type=int, default=DEFAULT_AUDIO_BPS,
                   help=f"Audio bitrate in bps (default {DEFAULT_AUDIO_BPS}).")
    p.add_argument("-n","--max-iter", type=int, default=DEFAULT_MAX_ITER,
                   help=f"Maximum optimization iterations (default {DEFAULT_MAX_ITER}).")
    return p.parse_args()

# ---------------- PIPELINE ---------------- #

def main():
    args = parse_args()

    if not os.path.exists(args.input):
        sys.exit(f"Input file not found: {args.input}")

    workdir = tempfile.mkdtemp(prefix="smart_av1_", dir="/tmp")
    print(f"Workdir: {workdir}")

    try:
        # ---- probe ----
        duration = ffprobe_duration(args.input)
        in_w, in_h, in_fps = ffprobe_video(args.input)
        input_has_audio = has_audio(args.input)
        print(f"Source: {in_w}x{in_h} @ {in_fps:.2f}fps, "
              f"{duration:.1f}s, audio={input_has_audio}")

        # ---- audio (encoded once, muxed at end) ----
        audio_path = None
        audio_size = 0
        if input_has_audio:
            audio_path = os.path.join(workdir, "audio.opus")
            print("Encoding audio...")
            run([
                "ffmpeg","-y","-loglevel","error",
                "-i", args.input,
                "-vn",
                "-c:a","libopus",
                "-b:a", str(args.audio_bps),
                "-vbr","on",
                audio_path
            ])
            audio_size = filesize(audio_path)
            print(f"  audio: {audio_size/1024:.1f} KiB")

        # ---- scene detection + chunking + analysis ----
        print("Detecting scenes...")
        cuts = detect_scenes(args.input)
        timestamps = [0.0] + cuts + [duration]

        scenes = []
        for i in range(len(timestamps) - 1):
            start = timestamps[i]
            end   = timestamps[i + 1]
            dur   = end - start
            if dur < 0.2:
                continue
            chunk = os.path.join(workdir, f"chunk_{i:05d}.mp4")
            run([
                "ffmpeg","-y","-loglevel","error",
                "-ss", str(start),
                "-i",  args.input,
                "-t",  str(dur),
                "-c:v","copy","-an",
                chunk
            ], quiet=True)
            motion, edges = analyze_scene(args.input, start, dur)
            res, fps, mode, crf = decide(motion, edges)
            scenes.append({
                "id":       i,
                "file":     chunk,
                "out":      os.path.join(workdir, f"chunk_{i:05d}_av1.mp4"),
                "motion":   motion,
                "edges":    edges,
                "duration": dur,
                "want_w":   res[0],
                "want_h":   res[1],
                "want_fps": fps,
                "mode":     mode,
                "crf":      crf,
                "bitrate":  0,
                "vmaf":     None,
                "last_size": 0,
            })
        if not scenes:
            sys.exit("No usable scenes detected.")
        print(f"  {len(scenes)} scene chunks")

        # ---- uniform output spec across all chunks so concat-copy is valid ----
        out_w   = min(in_w,   max(s["want_w"]   for s in scenes))
        out_h   = min(in_h,   max(s["want_h"]   for s in scenes))
        out_fps = min(in_fps, max(s["want_fps"] for s in scenes))
        out_w -= out_w % 2   # yuv420p needs even dims
        out_h -= out_h % 2
        print(f"Output: {out_w}x{out_h} @ {out_fps}fps")

        # ---- initial bitrate allocation (motion-weighted) ----
        budget_bytes  = args.target_mb * 1024 * 1024 * CONTAINER_OVERHEAD
        video_budget  = max(0, budget_bytes - audio_size)
        video_bits    = video_budget * 8
        total_motion  = sum(s["motion"] for s in scenes) or 1.0
        for s in scenes:
            weight = s["motion"] / total_motion
            s["bitrate"] = max(MIN_BITRATE_BPS,
                               int(video_bits * weight / s["duration"]))

        # ---- parallelism: pool x threads ≈ cores ----
        pool_size       = max(1, min(cpu_count(), len(scenes)))
        threads_per_job = max(1, cpu_count() // pool_size)
        print(f"Pool: {pool_size} workers × {threads_per_job} threads")

        # ---- iterative optimization: size AND VMAF ----
        outputs = []
        target_bytes = args.target_mb * 1024 * 1024

        for it in range(args.max_iter):
            print(f"\n=== Iteration {it+1}/{args.max_iter} ===")

            # encode
            with Pool(pool_size) as p:
                outputs = p.map(encode_scene, [
                    (s, out_w, out_h, out_fps, threads_per_job) for s in scenes
                ])
            for s, o in zip(scenes, outputs):
                s["last_size"] = filesize(o)
            video_size = sum(s["last_size"] for s in scenes)
            total_size = video_size + audio_size
            print(f"  size: {total_size/1024/1024:.2f} MB "
                  f"(video {video_size/1024/1024:.2f} + "
                  f"audio {audio_size/1024/1024:.2f}) "
                  f"/ {args.target_mb} MB")

            # measure VMAF
            print("  Computing VMAF...")
            with Pool(pool_size) as p:
                vmaf_results = p.map(vmaf_worker, [
                    (s, out_w, out_h, out_fps, threads_per_job) for s in scenes
                ])
            vmaf_map = dict(vmaf_results)
            for s in scenes:
                s["vmaf"] = vmaf_map[s["id"]]
            mean_v = sum(s["vmaf"] for s in scenes) / len(scenes)
            min_v  = min(s["vmaf"] for s in scenes)
            failing = [s for s in scenes if s["vmaf"] < args.vmaf]
            print(f"  VMAF: mean {mean_v:.2f} | min {min_v:.2f} | "
                  f"{len(failing)}/{len(scenes)} below {args.vmaf}")

            size_err   = abs(total_size - target_bytes) / target_bytes
            size_ok    = size_err < SIZE_TOLERANCE
            quality_ok = not failing
            if size_ok and quality_ok:
                print("  ✓ Both targets met.")
                break

            # per-scene boost based on VMAF gap, then global rescale for size
            mults = {}
            for s in scenes:
                if s["vmaf"] < args.vmaf:
                    gap = args.vmaf - s["vmaf"]
                    mults[s["id"]] = 1.0 + min(0.6, gap * 0.08)
                else:
                    mults[s["id"]] = 1.0
            predicted_video = sum(s["last_size"] * mults[s["id"]] for s in scenes)
            scale = (video_budget / predicted_video) if predicted_video > 0 else 1.0

            for s in scenes:
                # stubborn low-VMAF scenes leave CRF and go to explicit VBR so
                # the bitrate multiplier actually does something on the next pass
                if s["vmaf"] < args.vmaf and s["mode"] == "crf":
                    s["mode"] = "vbr"
                    s["bitrate"] = max(s["bitrate"],
                                       int(s["last_size"] * 8 / s["duration"]))
                s["bitrate"] = max(MIN_BITRATE_BPS,
                                   int(s["bitrate"] * mults[s["id"]] * scale))
        else:
            print("\n! Reached max_iter without meeting both targets; "
                  "using last encode.")

        # ---- concat video (uniform spec → -c copy is safe) ----
        listfile = os.path.join(workdir, "concat.txt")
        with open(listfile, "w") as f:
            for o in outputs:
                f.write(f"file '{os.path.abspath(o)}'\n")
        video_only = os.path.join(workdir, "video_only.mp4")
        run([
            "ffmpeg","-y","-loglevel","error",
            "-f","concat","-safe","0",
            "-i", listfile,
            "-c","copy",
            video_only
        ])

        # ---- mux ----
        if audio_path:
            run([
                "ffmpeg","-y","-loglevel","error",
                "-i", video_only,
                "-i", audio_path,
                "-map","0:v:0","-map","1:a:0",
                "-c","copy",
                "-movflags","+faststart",
                args.output
            ])
        else:
            run([
                "ffmpeg","-y","-loglevel","error",
                "-i", video_only,
                "-c","copy",
                "-movflags","+faststart",
                args.output
            ])

        final_mb = filesize(args.output) / (1024 * 1024)
        print(f"\nDone → {args.output} ({final_mb:.2f} MB)")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
