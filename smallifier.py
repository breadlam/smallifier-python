#!/usr/bin/env python3

import os
import sys
import math
import json
import shutil
import subprocess
from multiprocessing import Pool, cpu_count

# ---------------- CONFIG ---------------- #

DEFAULT_AUDIO_BITRATE = 48_000
MAX_ITER = 6

# ---------------------------------------- #

def run(cmd):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

def ffprobe_duration(path):
    out = subprocess.check_output([
        "ffprobe","-v","error",
        "-show_entries","format=duration",
        "-of","default=noprint_wrappers=1:nokey=1",
        path
    ])
    return float(out.strip())

def filesize(path):
    return os.path.getsize(path)

# ---------------- SCENE DETECTION ---------------- #

def detect_scenes_ffmpeg(input_file, threshold=0.4):
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
                    times.append(float(part.split(":")[1]))

    proc.wait()
    return sorted(times)

# ---------------- ANALYSIS ---------------- #

def analyze_scene(input_file, start, duration):
    # motion proxy
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
                val = float(line.split("YAVG:")[1].split()[0])
                motion_vals.append(val)
            except:
                pass

    proc.wait()
    motion = sum(motion_vals)/len(motion_vals) if motion_vals else 0.1

    # edge proxy
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

def decide(scene):
    motion, edges = scene["motion"], scene["edges"]

    if edges > 0.25:
        res = (960,540)
    elif motion > 0.5:
        res = (768,432)
    else:
        res = (640,360)

    if motion < 0.15:
        fps = 18
    elif motion < 0.3:
        fps = 24
    else:
        fps = 30

    if motion > 0.5:
        mode = "2pass"
    else:
        mode = "crf"

    crf = 34 if edges > 0.2 else 38

    return res, fps, mode, crf

# ---------------- ENCODING ---------------- #

def encode_scene(args):
    scene, bitrate, workdir = args
    infile = scene["file"]
    outfile = scene["out"]

    w,h,fps,mode,crf = scene["params"]

    vf = f"fps={fps},scale={w}:{h}:flags=lanczos,atadenoise=0.02:0.02:0.02"
    if scene["edges"] > 0.25:
        vf += ",unsharp=5:5:0.8:3:3:0.4"

    if mode == "crf":
        cmd = [
            "ffmpeg","-y","-i",infile,
            "-vf",vf,
            "-c:v","libsvtav1",
            "-crf",str(crf),
            "-b:v","0",
            "-cpu-used","0",
            "-row-mt","1",
            "-threads","2",
            "-an",
            outfile
        ]
        run(cmd)
    else:
        log = outfile + ".log"

        run([
            "ffmpeg","-y","-i",infile,
            "-vf",vf,
            "-c:v","libsvtav1",
            "-b:v",str(bitrate),
            "-pass","1",
            "-passlogfile",log,
            "-an",
            "-f","null","/dev/null"
        ])

        run([
            "ffmpeg","-y","-i",infile,
            "-vf",vf,
            "-c:v","libaom-av1",
            "-b:v",str(bitrate),
            "-pass","2",
            "-passlogfile",log,
            "-c:a","libopus","-b:a","48k",
            outfile
        ])

    return outfile

# ---------------- PIPELINE ---------------- #

def main():
    if len(sys.argv) < 3:
        print("Usage: smart_av1_encoder.py input.mp4 10")
        sys.exit(1)

    INPUT = sys.argv[1]
    TARGET_MB = float(sys.argv[2])
    WORKDIR = "smart_encode_work"
    os.makedirs(WORKDIR, exist_ok=True)

    duration = ffprobe_duration(INPUT)

    print("Detecting scenes...")
    cuts = detect_scenes_ffmpeg(INPUT)

    timestamps = [0.0] + cuts + [duration]

    scenes = []
    for i in range(len(timestamps)-1):
        start = timestamps[i]
        end = timestamps[i+1]
        dur = end - start

        chunk = os.path.join(WORKDIR, f"chunk_{i:03d}.mp4")

        run([
            "ffmpeg","-y",
            "-ss",str(start),
            "-i",INPUT,
            "-t",str(dur),
            "-c","copy",
            chunk
        ])

        motion, edges = analyze_scene(INPUT, start, dur)

        res,fps,mode,crf = decide({"motion":motion,"edges":edges})

        scenes.append({
            "file":chunk,
            "out":chunk.replace(".mp4","_av1.mp4"),
            "motion":motion,
            "edges":edges,
            "duration":dur,
            "params":(res[0],res[1],fps,mode,crf)
        })

    target_bits = TARGET_MB * 8 * 1024 * 1024
    audio_bits = DEFAULT_AUDIO_BITRATE * duration
    video_bits = target_bits - audio_bits

    # initial allocation
    total_motion = sum(s["motion"] for s in scenes) or 1
    for s in scenes:
        weight = s["motion"] / total_motion
        s["bitrate"] = int(video_bits * weight / s["duration"])

    for iteration in range(MAX_ITER):
        print(f"--- ITERATION {iteration} ---")

        with Pool(cpu_count()) as p:
            outputs = p.map(
                encode_scene,
                [(s, s["bitrate"], WORKDIR) for s in scenes]
            )

        total_size = sum(filesize(o) for o in outputs)
        print("Total size:", total_size / (1024*1024), "MB")

        target_size = TARGET_MB * 1024 * 1024
        error = total_size - target_size

        if abs(error) < target_size * 0.05:
            break

        ratio = target_size / total_size

        for s in scenes:
            s["bitrate"] = int(s["bitrate"] * ratio)

    # concat
    listfile = os.path.join(WORKDIR,"concat.txt")
    with open(listfile,"w") as f:
        for o in sorted(outputs):
            f.write(f"file '{os.path.abspath(o)}'\n")

    run([
        "ffmpeg","-y",
        "-f","concat","-safe","0",
        "-i",listfile,
        "-c","copy",
        "final_output.mp4"
    ])

    print("Done → final_output.mp4")


if __name__ == "__main__":
    main()
