#!/usr/bin/env python3
"""Run repeated SPF runtime profiling trials and aggregate reviewer-ready results."""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_GEOJSON_PATH = BASE_DIR / "data" / "riseholme_poles_trunk.geojson"
DEFAULT_OUTPUT_ROOT = BASE_DIR / "results" / "runtime_profile"
SPF_SCRIPT = BASE_DIR / "scripts" / "spf_lidar.py"
DEFAULT_DATA_CANDIDATES = [
    BASE_DIR / "data" / "2025" / "ICRA2",
    BASE_DIR / "data" / "2025" / "rh_run1",
    BASE_DIR / "data" / "2025" / "rh_run2",
]

RUNTIME_STAGE_FIELDS = [
    "io_sec",
    "semantic_inference_sec",
    "lidar_association_sec",
    "motion_update_sec",
    "measurement_update_sec",
    "pose_post_sec",
    "resample_sec",
    "visualization_sec",
    "stats_write_sec",
    "other_sec",
]


def _default_data_path() -> Path:
    for candidate in DEFAULT_DATA_CANDIDATES:
        if (candidate / "data.csv").exists():
            return candidate
    return DEFAULT_DATA_CANDIDATES[0]


def _mean_std(values: List[float]) -> Tuple[float | None, float | None]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None, None
    if len(clean) == 1:
        return float(clean[0]), 0.0
    return float(statistics.mean(clean)), float(statistics.stdev(clean))


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _run_cmd(cmd: List[str], cwd: Path, env: Dict[str, str], log_path: Path) -> float:
    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    duration = time.perf_counter() - start
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        f.write("$ " + " ".join(cmd) + "\n\n")
        f.write(proc.stdout or "")
        f.write(f"\n[exit_code={proc.returncode} duration_sec={duration:.3f}]\n")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)} (see {log_path})")
    return duration


def _read_cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.lower().startswith("model name"):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    return parts[1].strip()
    return platform.processor() or "unknown"


def _read_memory_gb() -> float | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None
    for line in meminfo.read_text().splitlines():
        if line.startswith("MemTotal:"):
            parts = line.split()
            if len(parts) >= 2:
                kb = float(parts[1])
                return kb / (1024.0 * 1024.0)
    return None


def _query_nvidia_smi() -> List[Dict[str, str]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return []
    if proc.returncode != 0:
        return []
    gpus = []
    for line in (proc.stdout or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            gpus.append(
                {
                    "name": parts[0],
                    "driver_version": parts[1],
                    "memory_total_mib": parts[2],
                }
            )
    return gpus


def _query_torch_runtime(python_exec: Path) -> Dict[str, object]:
    probe = [
        str(python_exec),
        "-c",
        (
            "import json, torch; "
            "info={'torch_version': torch.__version__, "
            "'cuda_available': bool(torch.cuda.is_available()), "
            "'cuda_version': torch.version.cuda, "
            "'device_count': int(torch.cuda.device_count()), "
            "'devices': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}; "
            "print(json.dumps(info))"
        ),
    ]
    proc = subprocess.run(
        probe,
        cwd=str(BASE_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return {"probe_error": (proc.stdout or "").strip()}
    lines = [line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]
    if not lines:
        return {"probe_error": "empty torch probe output"}
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        return {"probe_error": lines[-1]}


def _collect_hardware_snapshot(python_exec: Path) -> Dict[str, object]:
    return {
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
            "cpu_count_logical": os.cpu_count(),
            "cpu_model": _read_cpu_model(),
            "memory_total_gb": _read_memory_gb(),
        },
        "nvidia_smi": _query_nvidia_smi(),
        "torch_runtime": _query_torch_runtime(python_exec),
    }


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_trial_command(
    python_exec: Path,
    trial_dir: Path,
    data_path: Path,
    geojson_path: Path,
    seed: int,
    frame_stride: int,
    max_frames: int,
    warmup_frames: int,
    particle_count: int,
    segment_chunk: int,
    require_cuda: bool,
    miss_penalty: float,
    wrong_hit_penalty: float,
    gps_weight: float,
) -> List[str]:
    cmd = [
        str(python_exec),
        str(SPF_SCRIPT),
        "--seed",
        str(seed),
        "--output-folder",
        str(trial_dir),
        "--data-path",
        str(data_path),
        "--geojson-path",
        str(geojson_path),
        "--frame-stride",
        str(max(1, frame_stride)),
        "--max-frames",
        str(max_frames),
        "--particle-count",
        str(max(1, particle_count)),
        "--segment-chunk",
        str(max(32, segment_chunk)),
        "--miss-penalty",
        str(miss_penalty),
        "--wrong-hit-penalty",
        str(wrong_hit_penalty),
        "--gps-weight",
        str(gps_weight),
        "--no-visualization",
        "--profile-runtime",
        "--profile-warmup-frames",
        str(max(0, warmup_frames)),
    ]
    if require_cuda:
        cmd.append("--require-cuda")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repeated runtime profiling and aggregate results.")
    parser.add_argument("--python-exec", type=Path, default=BASE_DIR / ".venv" / "bin" / "python")
    parser.add_argument("--data-path", type=Path, default=_default_data_path())
    parser.add_argument("--geojson-path", type=Path, default=DEFAULT_GEOJSON_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed-base", type=int, default=11)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--measured-frames", type=int, default=200)
    parser.add_argument("--warmup-frames", type=int, default=20)
    parser.add_argument("--particle-count", type=int, default=100)
    parser.add_argument("--segment-chunk", type=int, default=4096)
    parser.add_argument("--miss-penalty", type=float, default=4.0)
    parser.add_argument("--wrong-hit-penalty", type=float, default=4.0)
    parser.add_argument("--gps-weight", type=float, default=0.5)
    parser.add_argument("--require-cuda", dest="require_cuda", action="store_true", default=True)
    parser.add_argument("--allow-cpu", dest="require_cuda", action="store_false")
    args = parser.parse_args()

    if args.trials <= 0:
        raise ValueError("--trials must be positive")
    if args.measured_frames <= 0:
        raise ValueError("--measured-frames must be positive")

    python_exec = args.python_exec.expanduser()
    if not python_exec.is_absolute():
        python_exec = (BASE_DIR / python_exec).resolve()

    data_path = args.data_path.expanduser()
    if not data_path.is_absolute():
        data_path = (BASE_DIR / data_path).resolve()
    geojson_path = args.geojson_path.expanduser()
    if not geojson_path.is_absolute():
        geojson_path = (BASE_DIR / geojson_path).resolve()

    if not python_exec.exists():
        raise FileNotFoundError(f"Missing python executable: {python_exec}")
    if not SPF_SCRIPT.exists():
        raise FileNotFoundError(f"Missing SPF script: {SPF_SCRIPT}")
    if not data_path.exists():
        known = ", ".join(str(p) for p in DEFAULT_DATA_CANDIDATES)
        raise FileNotFoundError(f"Missing data path: {data_path}. Known candidates: {known}")
    if not geojson_path.exists():
        raise FileNotFoundError(f"Missing geojson path: {geojson_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    hw_tag = "gpu" if args.require_cuda else "cpu"
    run_dir = args.output_root.resolve() / f"{timestamp}_runtime_{hw_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    hardware = _collect_hardware_snapshot(python_exec)
    max_frames = int(args.warmup_frames + args.measured_frames)
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(BASE_DIR / ".tmp_mpl")
    (BASE_DIR / ".tmp_mpl").mkdir(parents=True, exist_ok=True)

    trial_rows: List[Dict[str, object]] = []
    summaries: List[Dict[str, object]] = []

    for trial_idx in range(args.trials):
        trial_num = trial_idx + 1
        seed = args.seed_base + trial_idx
        trial_dir = run_dir / f"trial_{trial_num:02d}_seed_{seed}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        cmd = _build_trial_command(
            python_exec=python_exec,
            trial_dir=trial_dir,
            data_path=data_path,
            geojson_path=geojson_path,
            seed=seed,
            frame_stride=args.frame_stride,
            max_frames=max_frames,
            warmup_frames=args.warmup_frames,
            particle_count=args.particle_count,
            segment_chunk=args.segment_chunk,
            require_cuda=args.require_cuda,
            miss_penalty=args.miss_penalty,
            wrong_hit_penalty=args.wrong_hit_penalty,
            gps_weight=args.gps_weight,
        )
        print(f"[trial {trial_num}/{args.trials}] seed={seed} -> {' '.join(cmd)}")
        duration = _run_cmd(cmd, cwd=BASE_DIR, env=env, log_path=trial_dir / "run.log")

        summary_path = trial_dir / "runtime_profile_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing runtime summary for trial {trial_num}: {summary_path}")
        summary = json.loads(summary_path.read_text())
        summaries.append(summary)

        pipeline = summary.get("pipeline", {})
        profile = summary.get("profile", {})
        row: Dict[str, object] = {
            "trial": trial_num,
            "seed": seed,
            "command_runtime_sec": float(duration),
            "processed_frame_hz": pipeline.get("processed_frame_hz"),
            "effective_input_hz_with_stride": pipeline.get("effective_input_hz_with_stride"),
            "processed_frame_sec_mean": pipeline.get("processed_frame_sec_mean"),
            "processed_frame_sec_p95": pipeline.get("processed_frame_sec_p95"),
            "profiled_frames": profile.get("profiled_frames"),
            "warmup_frames": profile.get("warmup_frames"),
            "frame_stride": profile.get("frame_stride"),
        }
        for stage_name in RUNTIME_STAGE_FIELDS:
            comp = summary.get("components", {}).get(stage_name, {})
            row[f"{stage_name}_mean_ms"] = comp.get("mean_ms")
            row[f"{stage_name}_p95_ms"] = comp.get("p95_ms")
            row[f"{stage_name}_share"] = comp.get("share_of_frame_time")
        trial_rows.append(row)

    hz_mean, hz_std = _mean_std([row["processed_frame_hz"] for row in trial_rows])
    eff_hz_mean, eff_hz_std = _mean_std([row["effective_input_hz_with_stride"] for row in trial_rows])

    component_rows: List[Dict[str, object]] = []
    for stage_name in RUNTIME_STAGE_FIELDS:
        mean_ms_vals = [row.get(f"{stage_name}_mean_ms") for row in trial_rows]
        p95_ms_vals = [row.get(f"{stage_name}_p95_ms") for row in trial_rows]
        share_vals = [row.get(f"{stage_name}_share") for row in trial_rows]
        mean_ms_mean, mean_ms_std = _mean_std(mean_ms_vals)
        p95_ms_mean, p95_ms_std = _mean_std(p95_ms_vals)
        share_mean, share_std = _mean_std(share_vals)
        component_rows.append(
            {
                "component": stage_name,
                "mean_ms_mean": mean_ms_mean,
                "mean_ms_std": mean_ms_std,
                "p95_ms_mean": p95_ms_mean,
                "p95_ms_std": p95_ms_std,
                "share_of_frame_time_mean": share_mean,
                "share_of_frame_time_std": share_std,
                "equivalent_hz_from_mean_ms": (1000.0 / mean_ms_mean) if mean_ms_mean and mean_ms_mean > 1e-12 else None,
            }
        )

    summary = {
        "run_dir": str(run_dir),
        "config": {
            "trials": args.trials,
            "seed_base": args.seed_base,
            "frame_stride": args.frame_stride,
            "measured_frames": args.measured_frames,
            "warmup_frames": args.warmup_frames,
            "max_frames": max_frames,
            "particle_count": args.particle_count,
            "segment_chunk": args.segment_chunk,
            "miss_penalty": args.miss_penalty,
            "wrong_hit_penalty": args.wrong_hit_penalty,
            "gps_weight": args.gps_weight,
            "require_cuda": bool(args.require_cuda),
            "python_exec": str(python_exec),
            "data_path": str(data_path),
            "geojson_path": str(geojson_path),
        },
        "hardware": hardware,
        "pipeline_hz": {
            "processed_frame_hz_mean": hz_mean,
            "processed_frame_hz_std": hz_std,
            "effective_input_hz_with_stride_mean": eff_hz_mean,
            "effective_input_hz_with_stride_std": eff_hz_std,
        },
        "artifacts": {
            "trial_metrics_csv": str(run_dir / "runtime_profile_trials.csv"),
            "component_breakdown_csv": str(run_dir / "runtime_profile_components.csv"),
            "summary_json": str(run_dir / "runtime_profile_experiment_summary.json"),
        },
    }

    _write_csv(run_dir / "runtime_profile_trials.csv", trial_rows)
    _write_csv(run_dir / "runtime_profile_components.csv", component_rows)
    (run_dir / "runtime_profile_experiment_summary.json").write_text(json.dumps(summary, indent=2))

    print(
        f"[done] processed_frame_hz={_fmt(hz_mean)} +/- {_fmt(hz_std)} "
        f"(stride-adjusted {_fmt(eff_hz_mean)} +/- {_fmt(eff_hz_std)})"
    )
    print(f"[done] artifacts in {run_dir}")


if __name__ == "__main__":
    main()
