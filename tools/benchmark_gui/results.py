import csv
import json
import os


def load_results(output_dir):
    """
    安全加载 request_metrics.csv 和 summary.json。
    任一文件缺失或损坏都不抛异常，返回空 rows / 空 summary。
    """
    csv_path = os.path.join(output_dir, "request_metrics.csv")
    json_path = os.path.join(output_dir, "summary.json")

    rows = []
    if os.path.isfile(csv_path):
        try:
            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        except Exception:
            rows = []

    summary = {}
    if os.path.isfile(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
        except Exception:
            summary = {}

    return rows, summary, csv_path, json_path


def as_float(value):
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def find_nested_dict(data, target_key):
    """递归寻找 summary JSON 中指定 key 的第一个字典值。"""
    if isinstance(data, dict):
        if target_key in data and isinstance(data[target_key], dict):
            return data[target_key]
        for value in data.values():
            found = find_nested_dict(value, target_key)
            if found is not None:
                return found
    elif isinstance(data, list):
        for value in data:
            found = find_nested_dict(value, target_key)
            if found is not None:
                return found
    return None