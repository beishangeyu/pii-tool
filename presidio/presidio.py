from presidio_analyzer import AnalyzerEngine
import json, os
import argparse
from tqdm import tqdm
import gzip
from typing import List, Dict


# 更新结果文件
def update_result(result_file_path, batch_cnt, cur_batch_result, completed=False):
    # 如果不存在, 初始化
    if not os.path.exists(result_file_path):
        result_data = {"batches": {}, "batch_cnt": 0, "completed": False}
    # 如果存在, 读取
    else:
        with open(result_file_path, "r", encoding="utf-8") as rf:
            result_data = json.load(rf)
    # 更新
    result_data["batch_cnt"] = batch_cnt
    result_data["completed"] = completed
    result_data["batches"][f"batch_{batch_cnt}"] = cur_batch_result
    # 写回文件
    with open(result_file_path, "w", encoding="utf-8") as wf:
        json.dump(result_data, wf, indent=4)


def process(
    data_list: List[str],
    dataset_name: str,
    filename: str,
    batch_size,
    resume_batch_cnt,
    debug: bool,
):
    start_idx = resume_batch_cnt * batch_size
    # 划分batch
    batches = [
        data_list[i : i + batch_size]
        for i in range(start_idx, len(data_list), batch_size)
    ]
    if debug:
        result_file_path = f"./debug_results/{dataset_name}/{filename}.json"
    else:
        result_file_path = f"./results/{dataset_name}/{filename}.json"
    analyzer = AnalyzerEngine()
    # 储存不同类型entity的数量
    for i, batch in enumerate(batches):
        entity2cnt = {}
        batch_cnt = resume_batch_cnt + i + 1
        for text in batch:
            results = analyzer.analyze(text=text, language="en")
            for result in results:
                entity2cnt[result.entity_type] = (
                    entity2cnt.get(result.entity_type, 0) + 1
                )
        # 记录batch结果
        update_result(
            result_file_path=result_file_path,
            batch_cnt=batch_cnt,
            cur_batch_result=entity2cnt,
            completed=False if i < len(batches) - 1 else True,
        )
        if debug:
            print(
                f"Debug mode: Processed batch {batch_cnt} for file {filename}, entities: {entity2cnt}"
            )
            break


# NOTE 断点续传
def get_resume_file_list(dataset_name, data_path, debug):

    # 统计每个结果文件的批次数
    if debug:
        result_path = f"./debug_results/{dataset_name}/"
    else:
        result_path = f"./results/{dataset_name}/"
    os.makedirs(result_path, exist_ok=True)
    filename2batchcnt = {}
    # 读取已经存在的结果文件
    for file in os.listdir(result_path):
        result_file_path = os.path.join(result_path, file)
        with open(result_file_path, "r", encoding="utf-8") as rf:
            result_data = json.load(rf)
            is_completed = result_data.get("completed", False)
            batch_cnt = result_data.get("batch_cnt")
            assert batch_cnt > 0, f"batch_cnt: {batch_cnt}, in file: {file}"
            if "c4" in dataset_name:
                filename = file.split(".json")[0]
                filename2batchcnt[filename] = batch_cnt if not is_completed else -1

    # 构建需要续传的文件列表
    resume_file_list = []
    for file in os.listdir(data_path):
        if "c4" in dataset_name:
            filename = file.split(".json.gz")[0]
            resume_batch_cnt = filename2batchcnt.get(filename, 0)
            # -1 表示已经处理完
            if resume_batch_cnt != -1:
                resume_file_list.append(
                    {"filename": filename, "resume_batch_cnt": resume_batch_cnt}
                )

    return resume_file_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    resume_file_list = get_resume_file_list(
        args.dataset_name, args.data_path, args.debug
    )
    for item in tqdm(
        resume_file_list, desc=f"Processing {args.dataset_name}", disable=args.debug
    ):
        filename = item["filename"]
        resume_batch_cnt = item["resume_batch_cnt"]
        if args.debug:
            print(
                f"Debug mode: Processing file {filename}, resume from batch {resume_batch_cnt}"
            )
        data_list = []
        if "c4" in args.dataset_name:
            gz_file_path = os.path.join(args.data_path, filename + ".json.gz")
            with gzip.open(gz_file_path, "rt", encoding="utf-8") as f_in:
                for line in f_in:
                    item = json.loads(line)
                    data_list.append(item["text"])
        process(
            data_list=data_list,
            dataset_name=args.dataset_name,
            filename=filename,
            batch_size=args.batch_size if not args.debug else 10,
            resume_batch_cnt=resume_batch_cnt,
            debug=args.debug,
        )
        if args.debug:
            break
