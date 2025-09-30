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


def build_resume_list(
    dataset_name: str, data_path: str, debug: bool
) -> List[Tuple[str, int]]:
    """读取结果目录，生成 (filename, resume_batch_cnt) 列表"""
    rdir = result_dir(dataset_name, debug)
    ensure_dir(rdir)

    filename2batchcnt: Dict[str, int] = {}
    for file in os.listdir(rdir):
        if not file.endswith(".json"):
            continue
        fpath = os.path.join(rdir, file)
        try:
            with open(fpath, "r", encoding="utf-8") as rf:
                result_data = json.load(rf)
            is_completed = result_data.get("completed", False)
            batch_cnt = int(result_data.get("batch_cnt", 0))
            if "c4" in dataset_name:
                filename = file[: -len(".json")]
                filename2batchcnt[filename] = -1 if is_completed else batch_cnt
        except Exception:
            # 结果文件损坏则从 0 重新开始
            filename = file[: -len(".json")]
            filename2batchcnt[filename] = 0

    resume_list: List[Tuple[str, int]] = []
    for file in os.listdir(data_path):
        if not file.endswith(".json.gz"):
            continue
        if "c4" in dataset_name:
            filename = file[: -len(".json.gz")]
            resume_batch_cnt = filename2batchcnt.get(filename, 0)
            if resume_batch_cnt != -1:
                resume_list.append((filename, resume_batch_cnt))

    return resume_list
