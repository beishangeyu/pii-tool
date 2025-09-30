#!/usr/bin/env python3
"""
使用示例：

python multi_presidio.py \
  --dataset_name c4_en \
  --data_path /path/to/c4 \
  --batch_size 1000 \
  --workers 8 \
  --max_tasks_per_child 50

workers 是进程数默认 CPU 核心数-1。debug 模式会只处理一个批次并打印细节。
"""
from __future__ import annotations
import json, os, gzip, argparse, sys, math
from typing import List, Dict, Iterable, Tuple
from multiprocessing import Pool, cpu_count, set_start_method
from functools import partial
from itertools import islice


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
def iter_dataset(gz_file_path: str, datasetname: str) -> Iterable[str]:
    if datasetname == "c4":
        with gzip.open(gz_file_path, "rt", encoding="utf-8") as f_in:
            for line in f_in:
                try:
                    item = json.loads(line)
                    yield item["text"]
                except Exception:
                    # 跳过坏行
                    continue
    elif datasetname=='googlenq':
        with gzip.open(gz_file_path, "rt", encoding="utf-8") as f_in:
            for line in f_in:
                try:
                    item = json.loads(line)
                    yield item["question_text"]
                except Exception:
                    # 跳过坏行
                    continue


def batched(iterable: Iterable, n: int) -> Iterable[List]:
    """等价于流式 chunks(iterable, n)。"""
    it = iter(iterable)
    while True:
        chunk = list(islice(it, n))
        if not chunk:
            return
        yield chunk


# 顶层函数：子进程的入口（可被pickle）
def _run_one(args):
    dataset_name, data_path, batch_size, debug, filename, resume_batch_cnt = args
    return process_batches_for_file(
        dataset_name=dataset_name,
        data_path=data_path,
        filename=filename,
        batch_size=batch_size,
        resume_batch_cnt=resume_batch_cnt,
        debug=debug,
    )


def process_batches_for_file(
    dataset_name: str,
    data_path: str,
    filename: str,
    batch_size: int,
    resume_batch_cnt: int,
    debug: bool,
) -> Dict:
    """子进程执行体：顺序处理一个文件的所有 batch，并按批次落盘。"""
    # 延迟导入，避免主进程初始化 & 提高稳定性
    from presidio_analyzer import AnalyzerEngine  # type: ignore

    gz_file_path = os.path.join(data_path, filename + ".json.gz")
    rpath = result_path_for(dataset_name, filename, debug)

    analyzer = AnalyzerEngine()

    # 计算起始偏移：跳过已完成的批次
    start_skip = resume_batch_cnt * batch_size
    line_iter = iter_dataset(gz_file_path, dataset_name)

    # 跳过已完成行
    for _ in range(start_skip):
        try:
            next(line_iter)
        except StopIteration:
            break

    total_processed_batches = 0
    batch_index = resume_batch_cnt

    for i, batch_items in enumerate(batched(line_iter, batch_size)):
        batch_index = resume_batch_cnt + i + 1
        entity2cnt: Dict[str, int] = {}
        for item in batch_items:
            text = item
            if not text:
                continue
            try:
                results = analyzer.analyze(text=text, language="en")
                for res in results:
                    entity2cnt[res.entity_type] = entity2cnt.get(res.entity_type, 0) + 1
            except Exception:
                # 单条失败跳过，确保“不因一条坏样本中断整个进程”
                continue
        # 写出批次结果
        update_result(
            result_file_path=rpath,
            batch_cnt=batch_index,
            cur_batch_result=entity2cnt,
            completed=False,
        )
        total_processed_batches += 1
        if debug:
            print(
                f"[DEBUG] {filename}: processed batch {batch_index}, entities={entity2cnt}"
            )
            break  # debug 下只处理 1 个 batch

    # 标记文件完成（若 debug 则不标完成）
    if not debug:
        # 标记完成
        update_result(
            rpath,
            None,
            None,
            completed=True,
        )

    return {
        "filename": filename,
        "last_batch_cnt": batch_index,
        "batches_processed": total_processed_batches,
        "completed": (not debug),
    }


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=max(cpu_count() - 1, 1))
    parser.add_argument(
        "--max_tasks_per_child",
        type=int,
        default=0,
        help="每个子进程处理的任务数量上限，>0 可降低内存碎片",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    resume_list = build_resume_list(args.dataset_name, args.data_path, args.debug)

    if args.debug:
        # debug：只跑前 1 个文件
        resume_list = resume_list[:1]

    if not resume_list:
        print("No files to process. All completed or no input found.")
        return

    # 进度显示
    total_files = len(resume_list)
    processed = 0

    pool_kwargs = {"processes": args.workers}
    if args.max_tasks_per_child and args.max_tasks_per_child > 0:
        pool_kwargs["maxtasksperchild"] = args.max_tasks_per_child

    from tqdm import tqdm

    task_args = [
        (
            args.dataset_name,
            args.data_path,
            (args.batch_size if not args.debug else 10),
            args.debug,
            fn,
            rbc,
        )
        for (fn, rbc) in resume_list
    ]

    with Pool(**pool_kwargs) as pool:
        for summary in tqdm(
            pool.imap_unordered(_run_one, task_args),
            total=total_files,
            desc=f"Processing {args.dataset_name}",
        ):
            processed += 1
            # 可选择打印或汇总 summary
            if args.debug:
                print("[DEBUG] summary:", summary)

    print(f"Done. Files processed: {processed}/{total_files}.")


if __name__ == "__main__":
    main()
