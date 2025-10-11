from __future__ import annotations
import json, os, argparse
from typing import List, Dict, Tuple
from utils import (
    result_dir,
    result_path_for,
    ensure_dir,
    update_result,
    iter_dataset,
    batched,
    split_inputs_if_long,
)
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from tqdm import tqdm


def process_batches_for_file(
    dataset_name: str,
    data_path: str,
    filename: str,
    batch_size: int,
    resume_batch_cnt: int,
    debug: bool,
    tokenizer,
    pipe,
) -> Dict:
    if debug:
        batch_size = 1

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
        try:
            # 拆分超过最长窗口的输入
            inputs = split_inputs_if_long(
                batch_items, tokenizer, max_len=1024, is_debug=debug
            )
            results = pipe(inputs)
            for sentence_results in results:
                for entity_info in sentence_results:
                    ent = entity_info["entity"]
                    entity2cnt[ent] = entity2cnt.get(ent, 0) + 1
        except Exception as e:
            print(
                f"Error processing batch {batch_index} in file {filename}, skip it. the reason is: {e}"
            )
            if debug:
                print(f"Inputs:\n", "".join(inputs))
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
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    resume_list = build_resume_list(args.dataset_name, args.data_path, args.debug)

    if args.debug:
        # debug：只跑前 1 个文件
        resume_list = resume_list[:1]

    if not resume_list:
        print("No files to process. All completed or no input found.")
        return

    # 需要tokenizer计算token数目
    tokenizer = AutoTokenizer.from_pretrained("bigcode/starpii")
    model = AutoModelForTokenClassification.from_pretrained("bigcode/starpii")

    pipe = pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        device=args.device,
    )

    for file in tqdm(
        resume_list, desc=f"starpii processing {args.dataset_name}", disable=args.debug
    ):
        process_batches_for_file(
            dataset_name=args.dataset_name,
            data_path=args.data_path,
            filename=file[0],
            batch_size=args.batch_size,
            resume_batch_cnt=file[1],
            debug=args.debug,
            tokenizer=tokenizer,
            pipe=pipe,
        )


if __name__ == "__main__":
    main()
