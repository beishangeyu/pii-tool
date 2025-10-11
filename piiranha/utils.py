import gzip
import json
import os
from typing import Dict, Iterable, List, Tuple
from itertools import islice
import re


def split_by_newline(text: str) -> List[str]:
    # 按一个或多个连续换行符分割，通常用于分离段落
    paragraphs = re.split(r"\n+", text)

    # 过滤掉空字符串，并进行清理
    cleaned_paragraphs = []
    for p in paragraphs:
        p_strip = p.strip()
        if p_strip:
            cleaned_paragraphs.append(p_strip)

    return cleaned_paragraphs


def split_inputs_if_long(
    text: List[str], tokenizer, max_len: int = 256, is_debug: bool = False
) -> List[str]:
    inputs = []

    for s in text:
        s_stripped = s.strip()
        # 1. 检查原始文本（或段落）长度
        token_count = len(tokenizer(s_stripped)["input_ids"])

        if token_count <= max_len:
            inputs.append(s_stripped)
        else:
            # 2. **新增逻辑：尝试按换行符（段落）分割**
            segments = split_by_newline(s_stripped)

            # 用于收集最终满足长度要求的句子或段落
            valid_parts = []

            for segment in segments:
                segment_token_count = len(tokenizer(segment)["input_ids"])

                if segment_token_count <= max_len:
                    # 段落/行分割后满足长度要求
                    valid_parts.append(segment)
                else:
                    # 3. **如果段落/行仍太长，则按句末标点符号继续分割**

                    # 使用原有的句子分割逻辑
                    sentences = re.split(r"(?<=[。！？.!?])\s*", segment)

                    # 4. 再次检查分割后的句子是否满足长度
                    sentences = [
                        sen.strip()
                        for sen in sentences
                        if sen.strip() and len(tokenizer(sen)["input_ids"]) <= max_len
                    ]

                    valid_parts.extend(sentences)

                    if is_debug and sentences:
                        print(
                            "splited inputs example (Sentence Split):\n", sentences[0]
                        )

            # 5. 将处理后的所有部分加入到最终输出
            if valid_parts:
                inputs.extend(valid_parts)
            elif is_debug and not valid_parts:
                print(
                    f"Warning: Segment too long ({token_count} tokens) and could not be broken down into valid parts."
                )

    return inputs


def batched(iterable: Iterable, n: int) -> Iterable[List]:
    """等价于流式 chunks(iterable, n)。"""
    it = iter(iterable)
    while True:
        chunk = list(islice(it, n))
        if not chunk:
            return
        yield chunk


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
            try:
                result_data = json.load(rf)
            # 有时候出错中断后, 会出现json为空的情况
            except json.JSONDecodeError:
                result_data = {"batches": {}, "batch_cnt": 0, "completed": False}

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


# TODO 添加新的数据集时这里需要修改
# 流式读取数据集
def iter_dataset(file_path: str, datasetname: str) -> Iterable[str]:
    if datasetname == "c4" or datasetname == "dolma":
        try:
            with gzip.open(file_path, "rt", encoding="utf-8") as f_in:
                for line in f_in:
                    try:
                        item = json.loads(line)
                        yield item["text"]
                    except Exception:
                        continue
        except Exception as e:
            print(
                f"Error reading file: {file_path} in iter_dataset function. Error: {e}"
            )
    elif datasetname == "googlenq":
        with gzip.open(file_path, "rt", encoding="utf-8") as f_in:
            for line in f_in:
                try:
                    item = json.loads(line)
                    yield item["question_text"]
                except Exception:
                    continue


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
