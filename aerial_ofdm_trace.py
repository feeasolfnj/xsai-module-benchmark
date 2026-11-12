"""Summarize Aerial receive launch geometry and overlap from an Nsight Systems SQLite export."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3


TABLE = "CUPTI_ACTIVITY_KIND_KERNEL"


def overlap(rows):
    events = Counter()
    for row in rows:
        if row["end"] <= row["start"]:
            raise ValueError("Invalid kernel interval in CUDA trace")
        events[row["start"]] += 1
        events[row["end"]] -= 1
    active = peak = 0
    occupied_ns = overlap_ns = 0
    last = min(events)
    for stamp, change in sorted(events.items()):
        if active:
            occupied_ns += stamp - last
        if active > 1:
            overlap_ns += stamp - last
        active += change
        peak = max(peak, active)
        last = stamp
    return {"peak_overlapping_kernels": peak, "time_with_kernel_ns": occupied_ns,
            "time_with_multiple_kernels_ns": overlap_ns,
            "overlap_fraction_of_active_time": overlap_ns / occupied_ns}


def summarize(path):
    geometry = ("gridX", "gridY", "gridZ", "blockX", "blockY", "blockZ")
    resource = ("registersPerThread", "staticSharedMemory", "dynamicSharedMemory")
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        columns = {row["name"] for row in db.execute(f"PRAGMA table_info({TABLE})")}
        if not {"start", "end", "demangledName", *geometry} <= columns:
            raise ValueError("Nsight export is missing CUDA kernel geometry or timestamps")
        fields = ("start", "end", *geometry, *(name for name in (*resource, "graphNodeId") if name in columns))
        projection = ", ".join(f'k."{name}"' for name in fields)
        rows = list(db.execute(f"SELECT {projection} FROM {TABLE} k JOIN StringIds n ON k.demangledName=n.id "
                               "WHERE n.value LIKE '%ofdmDeMod_fft_kernel%' ORDER BY k.start"))
    if not rows:
        raise ValueError("No native Aerial OFDM receive kernels in the CUDA trace")
    launch_fields = (*geometry, *(name for name in resource if name in columns))
    signatures = Counter(tuple(row[name] for name in launch_fields) for row in rows)
    graph_rows = [row for row in rows if "graphNodeId" in columns and row["graphNodeId"]]
    return {"kernel_records": len(rows), "launch_configurations": [
                {**dict(zip(launch_fields, values, strict=True)), "count": count}
                for values, count in signatures.items()],
            "all_rx_intervals": overlap(rows), "graph_kernel_records": len(graph_rows),
            "graph_rx_intervals": overlap(graph_rows) if graph_rows else None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    print(json.dumps(summarize(parser.parse_args().sqlite), indent=2))
