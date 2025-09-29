import gzip
import json
import os
from typing import Dict, Iterable, List, Tuple


# 确保 path 存在
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def update_result(
    result_file_path: str,
    batch_cnt: int,
    cur_batch_result: Dict[str, int],
    completed: bool = False,
) -> None:
    """每个文件仅由一个进程负责，避免并发写入冲突。"""
    if not os.path.exists(result_file_path):
        result_data = {"batches": {}, "batch_cnt": 0, "completed": False}
    else:
        with open(result_file_path, "r", encoding="utf-8") as rf:
            result_data = json.load(rf)

    if completed == True:
        result_data["completed"] = True
        with open(result_file_path, "w", encoding="utf-8") as wf:
            json.dump(result_data, wf, indent=4)
        return

    result_data["batch_cnt"] = batch_cnt
    result_data["completed"] = completed
    result_data["batches"][f"batch_{batch_cnt}"] = cur_batch_result
    with open(result_file_path, "w", encoding="utf-8") as wf:
        json.dump(result_data, wf, indent=4)


def result_dir(dataset_name: str, debug: bool) -> str:
    return os.path.join("./debug_results" if debug else "./results", dataset_name)


def result_path_for(dataset_name: str, filename: str, debug: bool) -> str:
    ensure_dir(result_dir(dataset_name, debug))
    return os.path.join(result_dir(dataset_name, debug), f"{filename}.json")


# 流式读取 gzip 文件
def iter_dataset_c4(gz_file_path: str) -> Iterable[str]:
    with gzip.open(gz_file_path, "rt", encoding="utf-8") as f_in:
        for line in f_in:
            try:
                item = json.loads(line)
                yield item["text"]
            except Exception:
                # 跳过坏行
                continue
