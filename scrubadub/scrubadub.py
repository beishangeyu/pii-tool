import json, argparse
import scrubadub
from typing import List, Dict, Iterable, Tuple
from multiprocessing import Pool, cpu_count, set_start_method
import argparse
from .utils import build_resume_list


def save_metrics_to_csv(all_batch_metrics, filename):
    pass


def transfer_labels(data):
    pass


def calculate_metrics(results, labels):
    pass


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
    """直接从 JSONL 文件分批处理"""
    all_results = []
    batch_count = 0
    current_batch = []
    total_lines = 0
    all_metrics = []
    total_tp = total_fp = total_fn = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    data = json.loads(line)
                    current_batch.append(data)
                    total_lines += 1

                    # 当批次达到指定大小时处理
                    if len(current_batch) >= batch_size:
                        batch_count += 1
                        print(f"处理批次 {batch_count}，行数: {total_lines}")

                        labels = transfer_labels(current_batch)
                        result = process_batch(current_batch, batch_count)
                        if result:
                            all_results.append(result)
                        metrics = calculate_metrics(result, labels)
                        print(f"批次 {batch_count} 处理完成，指标: {metrics}")
                        all_metrics.append(
                            {
                                "batch": batch_count,
                                "precision": metrics["precision"],
                                "recall": metrics["recall"],
                                "f1": metrics["f1"],
                                "true_positives": metrics["true_positives"],
                                "false_positives": metrics["false_positives"],
                                "false_negatives": metrics["false_negatives"],
                            }
                        )
                        total_tp += metrics["true_positives"]
                        total_fp += metrics["false_positives"]
                        total_fn += metrics["false_negatives"]
                        current_batch = []  # 清空当前批次

                except json.JSONDecodeError as e:
                    print(f"第 {line_num} 行 JSON 解析错误: {e}")
                    continue

    # 处理最后一批（如果有剩余数据）
    if current_batch:
        batch_count += 1
        print(f"处理最后批次 {batch_count}，总行数: {total_lines}")
        result = process_batch(current_batch, batch_count)
        if result:
            all_results.append(result)
        metrics = calculate_metrics(result, labels)
        print(f"批次 {batch_count} 处理完成，指标: {metrics}")
        all_metrics.append(
            {
                "batch": batch_count,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "true_positives": metrics["true_positives"],
                "false_positives": metrics["false_positives"],
                "false_negatives": metrics["false_negatives"],
            }
        )
        total_tp += metrics["true_positives"]
        total_fp += metrics["false_positives"]
        total_fn += metrics["false_negatives"]

    print(f"所有 {batch_count} 个批次处理完成，总计 {total_lines} 行")
    final_precision = (
        total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    )
    final_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    final_f1 = (
        2 * (final_precision * final_recall) / (final_precision + final_recall)
        if (final_precision + final_recall) > 0
        else 0
    )

    all_metrics.append(
        {
            "batch": "final",
            "precision": final_precision,
            "recall": final_recall,
            "f1": final_f1,
            "true_positives": total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn,
        }
    )

    save_metrics_to_csv(all_metrics, "./ai4privacy_results/" + filename + ".csv")
    return all_metrics


def process_batch(batch_data, batch_num):
    """处理一批数据"""
    preds = {
        "credential": [],
        "credit_card": [],
        "email": [],
        "phone": [],
        "twitter": [],
        "url": [],
        "social_security_number": [],
    }
    for data in batch_data:
        # TAG
        results = scrubadub.list_filth(data["source_text"])
        for result in results:
            if result.type != "unknown":
                preds[result.type].append(data["source_text"][result.beg : result.end])
    return preds


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
    data_names = [
        "1english_openpii_8k",
        "dutch_openpii_7k",
        "french_openpii_8k",
        "german_openpii_8k",
        "italian_openpiii_8k",
        "spanish_openpii_8k",
    ]
    for data_name in data_names:
        input_jsonl_path = f"../ai4privacy/data/validation/{data_name}.jsonl"
        batch_size = 1000  # 每批处理的行数
        all_metrics = process_batches_for_file(input_jsonl_path, data_name, batch_size)
    print("所有批次的指标:", all_metrics)
