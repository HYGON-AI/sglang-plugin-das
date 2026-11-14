#!/usr/bin/env python3
import csv
import json
import sys
from pathlib import Path

DATASETS = {"humaneval": (164, "pass@1", "pass_at_k"),
            "gsm8k": (1319, "accuracy", "mean"),
            "hmmt_nov25": (30, "accuracy", "mean")}


def read_records(paths):
    records = []
    for path in sorted(paths):
        with path.open(encoding="utf-8") as stream:
            records.extend(json.loads(line) for line in stream
                           if line.endswith("\n") and line.strip())
    return records


def result_for(lane, stage, dataset):
    total, metric, aggregation = DATASETS[dataset]
    result = dict(stage=stage, dataset=dataset, metric=metric, expected=total,
                  predicted=0, reviewed=0, correct=None, accuracy_pct=None,
                  api_errors=0, truncated=0, report="", valid=False)
    output = lane / "evalscope"
    reports = sorted(output.glob("**/reports/**/" + dataset + ".json"))
    if len(reports) > 1:
        raise ValueError("Multiple reports for {} in {}".format(dataset, lane))
    if reports:
        report = reports[0]
        session, model = report.parents[2], report.parent.name
        result["report"] = str(report)
        predictions = read_records((session / "predictions" / model).glob(dataset + "*.jsonl"))
        reviews = read_records((session / "reviews" / model).glob(dataset + "*.jsonl"))
    else:
        predictions = read_records(output.glob("**/predictions/*/" + dataset + "*.jsonl"))
        reviews = []
    result["predicted"] = len(predictions)
    result["reviewed"] = len(reviews)
    result["api_errors"] = sum(bool(p["model_output"].get("error")) for p in predictions)
    result["truncated"] = sum(any(c.get("stop_reason") in ("length", "max_tokens")
                                  for c in p["model_output"].get("choices", ()))
                              for p in predictions)
    if result["api_errors"] or len(predictions) != total or len(reviews) != total:
        return result

    correct = []
    for review in reviews:
        score = review["sample_score"]["score"]
        value = score["value"].get("accuracy", score["value"].get("acc"))
        if score["status"] != "success" or value not in (0, 1):
            return result
        correct.append(int(value))
    ids = [{r["index"] for r in items} for items in (predictions, reviews)]
    raw = json.loads(report.read_text(encoding="utf-8"))
    reported = next(m["score"] for m in raw["metrics"]
                    if m["identity"]["aggregation"] == aggregation)
    if (len(ids[0]) != total or ids[0] != ids[1] or raw["num"] != total
            or raw["dataset_name"] != dataset or abs(reported - sum(correct) / total) > 0.00051):
        return result
    result.update(valid=True, correct=sum(correct), accuracy_pct=100 * sum(correct) / total)
    return result


def main(run):
    root = run.parent.parent
    lines = [
        "# MiniMax-M3.1 Flash 插件精度评测", "",
        "在 nmz16 上依次测试独立 Prefill 和 Decode 服务。两端使用相同的 EvalScope 客户端和本地数据。", "",
        "评测使用全量 HumanEval、GSM8K、HMMT Nov25，开启 thinking，并发 32，最大输出 32768，seed 42。Decode 使用 full graph，跳过默认的单请求启动预热。", "",
    ]
    rows = []
    for stage in ("prefill", "decode"):
        lines.extend(["## " + stage.title(), ""])
        current = run / stage / "current.path"
        if not current.is_file():
            lines.extend(["等待运行。", ""])
            continue
        lane = Path(current.read_text().strip()).resolve()
        status = (lane / "status").read_text().strip() if (lane / "status").is_file() else "queued"
        rc_path = lane / "exit-code"
        outcome = "运行状态为 {}。".format(status)
        if rc_path.is_file():
            outcome += "退出码为 {}。".format(rc_path.read_text().strip())
        lines.extend([outcome, ""])
        for dataset in DATASETS:
            row = result_for(lane, stage, dataset)
            rows.append(row)
            if row["valid"]:
                link = Path(row["report"]).relative_to(root)
                lines.append("- {} {} 为 {}/{}（{:.4f}%），达到输出上限 {} 题、API 错误 {} 题。 [原始报告]({})".format(
                    dataset, row["metric"], row["correct"], row["expected"], row["accuracy_pct"],
                    row["truncated"], row["api_errors"], link))
            else:
                lines.append("- {} 已归档预测 {}/{}，判分 {}，API 错误 {}，待完整验收。".format(
                    dataset, row["predicted"], row["expected"], row["reviewed"], row["api_errors"]))
        log = (lane / "evalscope.log").relative_to(root)
        if (root / log).exists():
            lines.extend(["", "[客户端日志]({})".format(log)])
        lines.append("")
    baseline = root.parent.parent / "2MiniMax-M3.1-channel-int4-w4a8-int8-w8a8" / "3eval_accuracy1110" / "accuracy-summary.json"
    if baseline.is_file():
        historical = json.loads(baseline.read_text(encoding="utf-8"))
        lines.extend(["## 历史旧版结果", ""])
        for name in DATASETS:
            item = historical[name]
            lines.append("- {} 为 {}/{}（{:.4f}%）。".format(
                name, item["num_correct"], item["num_samples"], item["accuracy_pct"]))
        lines.append("")
    if rows:
        report = run / "accuracy-summary.csv"
        temp = report.with_suffix(".tmp")
        with temp.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temp.replace(report)
    report = root / "精度评测总结.md"
    temp = report.with_suffix(".tmp")
    temp.write_text("\n".join(lines), encoding="utf-8")
    temp.replace(report)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
