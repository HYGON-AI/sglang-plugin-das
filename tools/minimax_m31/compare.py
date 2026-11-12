#!/usr/bin/env python3
import csv
import json
from math import isclose, isfinite
import sys
from pathlib import Path
from statistics import median

SIZES = (1, 2, 4, 8, 16, 32, 40, 48, 64)
HISTORIC_RUN = {"prefill": "20261109_210818", "decode": "20261109_212215"}
METRICS = {
    "prefill": (("mean_ttft_ms", 1),),
    "decode": (("mean_tpot_ms", 1), ("output_throughput", -1)),
}


HISTORIC_CSV = "Minimax-M3.1 - M3.1 BW1100 W4A8性能数据.csv"
CSV_METRIC = {
    "mean_ttft_ms": "mean_TTFT_ms",
    "mean_tpot_ms": "mean_TPOT_ms",
    "output_throughput": "output_token_tps",
}


def historical_csv(path):
    values = {}
    header = None
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.reader(stream):
            if len(row) < 3:
                continue
            if row[2] == "phase":
                header = row
            mode = {"prefill": "prefill", "decode/RSC": "decode"}.get(row[2])
            if mode is None:
                continue
            record = dict(zip(header, row))
            bs = int(record["max-concurrency"])
            for metric, _ in METRICS[mode]:
                key = mode, bs, metric
                if key in values:
                    raise ValueError("Duplicate historic row: " + repr(key))
                values[key] = float(record[CSV_METRIC[metric]])
    return values


def read(path):
    return json.loads(path.read_text())


def case(folder, mode, size):
    result = read(folder / "PASS.json")
    request = read(folder / "measurement.json")
    expected_cache = 131072 if mode == "prefill" else 147328
    if result["completed"] != size or result["cache_min"] != expected_cache:
        raise ValueError("Invalid completion or cache count in " + str(folder))
    if request["input_total"] != 147456 or request["output_len"] != (1 if mode == "prefill" else 1024):
        raise ValueError("Input or output length mismatch in " + str(folder))
    if request["shared_prefix"] != (mode == "prefill"):
        raise ValueError("Shared prefix setting mismatch in " + str(folder))
    if mode == "decode" and (not result["natural_acceptance"] or result["verify_count"] <= 0):
        raise ValueError("Missing natural DSpark acceptance in " + str(folder))
    return result, request["rows_sha256"]


def compare(replay, baseline):
    rows = []
    versions = set()
    csv_values = historical_csv(baseline / HISTORIC_CSV)
    for mode, metrics in METRICS.items():
        for size in SIZES:
            historical_path = baseline / mode / "results" / HISTORIC_RUN[mode]
            historical, digest = case(historical_path / "measurements" / ("bs%d" % size), mode, size)
            results = {"baseline": [], "plugin": []}
            for pair in (1, 2, 3):
                for role, samples in results.items():
                    lane = replay / mode / ("pair-%d" % pair) / role
                    if (lane / "status").read_text().strip() != "passed":
                        raise ValueError("Lane has not completed " + str(lane))
                    if role == "plugin":
                        versions.add(tuple(
                            (lane / name).read_text().strip()
                            for name in ("plugin-site.txt", "adapter-sha.txt", "community-sha.txt")
                        ))
                    values, measured_digest = case(lane / "measurements" / ("bs%d" % size), mode, size)
                    if measured_digest != digest:
                        raise ValueError("Frozen input mismatch " + str(lane))
                    samples.append(values)
            for metric, direction in metrics:
                old = csv_values[mode, size, metric]
                if not isclose(old, historical[metric], rel_tol=0, abs_tol=1e-6):
                    raise ValueError(f"Historical CSV disagrees with raw result for {mode}/{size}/{metric}")
                row = {"mode": mode, "bs": size, "metric": metric, "historical_csv": old}
                for role, samples in results.items():
                    values = [x[metric] for x in samples]
                    if any(not isfinite(x) or x <= 0 for x in values):
                        raise ValueError(f"Invalid performance measurement for {mode}/{size}/{role}/{metric}")
                    for repeat, value in enumerate(values, 1):
                        row["%s_%d" % (role, repeat)] = value
                    row[role + "_median"] = median(values)
                    row[role + "_cache_rates"] = json.dumps([x["cache_rate"] for x in samples])
                    row[role + "_accept_lengths"] = json.dumps([x.get("weighted_acceptance") for x in samples])
                delta = row["plugin_median"] / row["baseline_median"] - 1
                row["delta_pct"] = 100 * delta
                row["delta_to_historical_pct"] = 100 * (row["plugin_median"] / row["historical_csv"] - 1)
                regression = direction * delta
                row["pass"] = regression <= 0.05 or isclose(regression, 0.05, rel_tol=0, abs_tol=1e-12)
                rows.append(row)
    if len(versions) != 1:
        raise ValueError("Plugin rounds use different revisions")
    return rows


def main():
    replay, baseline = map(Path, sys.argv[1:])
    rows = compare(replay, baseline)
    temp = replay / "comparison.csv.tmp"
    with temp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(replay / "comparison.csv")
    passed = sum(row["pass"] for row in rows)
    print("Passed {}/{} metric and batch-size checks".format(passed, len(rows)))
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
