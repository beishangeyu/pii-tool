from __future__ import annotations
import json, os, argparse
from typing import List, Dict, Tuple
from multiprocessing import Pool, cpu_count, set_start_method
from utils import (
    result_dir,
    result_path_for,
    ensure_dir,
    update_result,
    iter_dataset,
    batched,
    strs2csv,
    delete_file
)



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
    """子进程执行体：顺序处理一个文件的所有 batch, 并按批次落盘。"""
    from piianalyzer.analyzer import PiiAnalyzer


    # TODO 添加新的数据集时这里需要修改
    if dataset_name == "c4" or dataset_name == "dolma":
        file_path = os.path.join(data_path, filename + ".json.gz")
    elif dataset_name == "googlenq":
        file_path = os.path.join(data_path, filename + ".jsonl.gz")

    rpath = result_path_for(dataset_name, filename, debug)

    # 计算起始偏移：跳过已完成的批次
    start_skip = resume_batch_cnt * batch_size
    line_iter = iter_dataset(file_path, dataset_name)

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
        csv_file_path = strs2csv(str_list=batch_items, filename=filename, batchcnt=batch_index)
        piianalyzer = PiiAnalyzer(csv_file_path)
        # key是种类, value是找到了哪些单词
        analysis:Dict[str, List[str]] = piianalyzer.analysis()
        for entity_type, entities in analysis.items():
            entity2cnt[entity_type] = entity2cnt.get(entity_type, 0) + len(entities)
        # 写出批次结果
        update_result(
            result_file_path=rpath,
            batch_cnt=batch_index,
            cur_batch_result=entity2cnt,
            completed=False,
        )
        delete_file(file_path=csv_file_path, debug=debug)
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
    # 读取结果目录, 这里的结果都是json格式
    for file in os.listdir(rdir):
        if not file.endswith(".json"):
            continue
        fpath = os.path.join(rdir, file)
        try:
            with open(fpath, "r", encoding="utf-8") as rf:
                result_data = json.load(rf)
            is_completed = result_data.get("completed", False)
            batch_cnt = int(result_data.get("batch_cnt", 0))
            filename = file[: -len(".json")]
            filename2batchcnt[filename] = -1 if is_completed else batch_cnt
        except Exception:
            filename = file[: -len(".json")]
            filename2batchcnt[filename] = 0

    # TODO 添加新的数据集这里需要修改
    # 遍历数据目录, 这里的格式不同数据集可能不同
    resume_list: List[Tuple[str, int]] = []
    for file in os.listdir(data_path):
        if "c4" in dataset_name or "dolma" in dataset_name:
            filename = file[: -len(".json.gz")]
            resume_batch_cnt = filename2batchcnt.get(filename, 0)
            if resume_batch_cnt != -1:
                resume_list.append((filename, resume_batch_cnt))
        elif "googlenq" in dataset_name:
            filename = file[: -len(".jsonl.gz")]
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
    ensure_dir("./tmp_csv")  # 确保临时csv目录存在
    

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
