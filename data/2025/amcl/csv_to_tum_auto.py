#!/usr/bin/env python3
"""
csv_to_tum_stamp_auto.py

Convert a ROS CSV (e.g., from `rostopic echo -p`) into TUM format:
    timestamp x y z q_x q_y q_z q_w

Timestamp priority (converted to float seconds since UNIX epoch):
1) field.header.stamp.secs + field.header.stamp.nsecs
2) field.header.stamp   (may be float or 19-digit concatenated stamp)
3) header.stamp / stamp (same handling)
4) %time                (same handling)

Position columns (tries these in order):
  field.pose.pose.position.[x|y|z], pose.pose.position.[x|y|z], position.[x|y|z], x|y|z

Orientation columns (quaternion in x,y,z,w):
  field.pose.pose.orientation.[x|y|z|w], pose.pose.orientation.[x|y|z|w],
  orientation.[x|y|z|w], q_x|q_y|q_z|q_w, qx|qy|qz|qw

Usage:
  python csv_to_tum_stamp_auto.py your.csv
  # writes your.tum
"""

import argparse
import os
import math
import pandas as pd

TS_CANDIDATES = [
    "field.header.stamp", "header.stamp", "stamp", "%time"
]

POS_CANDS = {
    "x": ["field.pose.pose.position.x", "pose.pose.position.x", "position.x", "x"],
    "y": ["field.pose.pose.position.y", "pose.pose.position.y", "position.y", "y"],
    "z": ["field.pose.pose.position.z", "pose.pose.position.z", "position.z", "z"],
}

ORI_CANDS = {
    "q_x": ["field.pose.pose.orientation.x", "pose.pose.orientation.x", "orientation.x", "q_x", "qx"],
    "q_y": ["field.pose.pose.orientation.y", "pose.pose.orientation.y", "orientation.y", "q_y", "qy"],
    "q_z": ["field.pose.pose.orientation.z", "pose.pose.orientation.z", "orientation.z", "q_z", "qz"],
    "q_w": ["field.pose.pose.orientation.w", "pose.pose.orientation.w", "orientation.w", "q_w", "qw"],
}

def pick(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None

def parse_timestamp_series(series: pd.Series) -> pd.Series:
    """
    Convert a timestamp column to float seconds.
    - If value is a 19-digit integer (>= 1e18), treat as secs*1e9 + nsecs (concatenated).
    - Else try float directly.
    """
    out = []
    for v in series:
        if pd.isna(v):
            out.append(float("nan"))
            continue
        s = str(v).strip()
        # Try integer path first (handles plain digits)
        try:
            iv = int(s)
            # Heuristic: ROS concatenated stamps are ~19 digits (secs*1e9 + nsecs)
            if abs(iv) >= 10**18:
                secs = iv // 1_000_000_000
                nsecs = iv % 1_000_000_000
                out.append(secs + nsecs * 1e-9)
            else:
                # This is already seconds (int seconds)
                out.append(float(iv))
            continue
        except ValueError:
            pass
        # Fallback: float (handles scientific notation, normal floats)
        try:
            out.append(float(s))
        except ValueError:
            out.append(float("nan"))
    return pd.Series(out, index=series.index, dtype="float64")

def get_timestamp(df: pd.DataFrame) -> pd.Series:
    # 1) secs + nsecs pair
    sec_col  = pick(df, ["field.header.stamp.secs", "header.stamp.secs", "stamp.secs", "secs"])
    nsec_col = pick(df, ["field.header.stamp.nsecs", "header.stamp.nsecs", "stamp.nsecs", "nsecs"])
    if sec_col and nsec_col:
        secs  = pd.to_numeric(df[sec_col], errors="coerce")
        nsecs = pd.to_numeric(df[nsec_col], errors="coerce")
        return secs + nsecs * 1e-9

    # 2) one of the single-stamp columns (might be float or 19-digit)
    stamp_col = pick(df, TS_CANDIDATES)
    if stamp_col:
        return parse_timestamp_series(df[stamp_col])

    raise ValueError("No usable timestamp columns found. "
                     "Tried secs/nsecs and one of: " + ", ".join(TS_CANDIDATES))

def need_column(df, candidates, key_name):
    col = pick(df, candidates)
    if col is None:
        raise ValueError(f"Missing required '{key_name}' column. Looked for: {candidates}")
    return col

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_file", help="Input CSV file")
    ap.add_argument("--delimiter", default=",", help="CSV delimiter (default ',')")
    args = ap.parse_args()

    base, _ = os.path.splitext(args.csv_file)
    out_path = base + ".tum"

    df = pd.read_csv(args.csv_file, delimiter=args.delimiter)

    # Build timestamp (float seconds)
    t = get_timestamp(df)

    # Get pose columns
    x_col  = need_column(df, POS_CANDS["x"], "x")
    y_col  = need_column(df, POS_CANDS["y"], "y")
    z_col  = need_column(df, POS_CANDS["z"], "z")
    qx_col = need_column(df, ORI_CANDS["q_x"], "q_x")
    qy_col = need_column(df, ORI_CANDS["q_y"], "q_y")
    qz_col = need_column(df, ORI_CANDS["q_z"], "q_z")
    qw_col = need_column(df, ORI_CANDS["q_w"], "q_w")

    out_df = pd.DataFrame({
        "timestamp": t,
        "x":  pd.to_numeric(df[x_col], errors="coerce"),
        "y":  pd.to_numeric(df[y_col], errors="coerce"),
        "z":  pd.to_numeric(df[z_col], errors="coerce"),
        "q_x": pd.to_numeric(df[qx_col], errors="coerce"),
        "q_y": pd.to_numeric(df[qy_col], errors="coerce"),
        "q_z": pd.to_numeric(df[qz_col], errors="coerce"),
        "q_w": pd.to_numeric(df[qw_col], errors="coerce"),
    })

    out_df = out_df.dropna(subset=["timestamp","x","y","z","q_x","q_y","q_z","q_w"])
    out_df = out_df.sort_values("timestamp")

    with open(out_path, "w") as f:
        for _, r in out_df.iterrows():
            f.write("{:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f}\n".format(
                r["timestamp"], r["x"], r["y"], r["z"], r["q_x"], r["q_y"], r["q_z"], r["q_w"]
            ))

    print(f"Wrote {len(out_df)} poses to {out_path}")

if __name__ == "__main__":
    main()

