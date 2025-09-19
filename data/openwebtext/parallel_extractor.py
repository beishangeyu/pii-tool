#!/usr/bin/env python3
"""
并行处理所有 tar 文件的脚本

这个脚本可以同时处理多个 tar 文件，充分利用多核 CPU
"""

import os
import sys
import multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# 添加当前目录到 Python 路径
sys.path.append(str(Path(__file__).parent))

from openwebtext_extractor import OpenWebTextExtractor


def process_single_tar_worker(args):
    """工作进程函数"""
    tar_file, output_dir, max_texts, worker_id = args
    
    try:
        # 为每个进程创建独立的提取器
        extractor = OpenWebTextExtractor()
        
        # 生成输出文件
        output_path = Path(output_dir)
        output_file = output_path / f"{tar_file.stem}.jsonl"
        
        print(f"[Worker {worker_id}] 开始处理: {tar_file.name}")
        
        with open(output_file, 'w', encoding='utf-8') as jsonl_file:
            texts_count = extractor.process_single_tar(tar_file, jsonl_file, max_texts)
        
        if texts_count > 0:
            file_size = output_file.stat().st_size / (1024 * 1024)  # MB
            result = {
                'status': 'success',
                'tar_file': tar_file.name,
                'texts_count': texts_count,
                'file_size': file_size,
                'output_file': str(output_file)
            }
            print(f"[Worker {worker_id}] ✓ 完成 {tar_file.name}: {texts_count} 个文本, {file_size:.2f} MB")
        else:
            result = {
                'status': 'no_texts',
                'tar_file': tar_file.name,
                'texts_count': 0,
                'file_size': 0,
                'output_file': str(output_file)
            }
            print(f"[Worker {worker_id}] ⚠ {tar_file.name}: 没有提取到文本")
            
        return result
        
    except Exception as e:
        print(f"[Worker {worker_id}] ✗ 处理 {tar_file.name} 失败: {e}")
        
        # 删除可能不完整的输出文件
        try:
            output_file = Path(output_dir) / f"{tar_file.stem}.jsonl"
            if output_file.exists():
                output_file.unlink()
        except:
            pass
            
        return {
            'status': 'error',
            'tar_file': tar_file.name,
            'error': str(e),
            'texts_count': 0,
            'file_size': 0
        }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="并行处理 OpenWebText 数据")
    parser.add_argument('--subsets-dir', default='/mnt/liaozy25/openwebtext/subsets',
                        help='subsets 目录路径')
    parser.add_argument('--output-dir', default='/mnt/liaozy25/openwebtext/extracted_texts_parallel',
                        help='输出目录路径')
    parser.add_argument('--max-texts', type=int, default=None,
                        help='每个文件最大提取文本数量')
    parser.add_argument('--workers', type=int, default=None,
                        help='并行工作进程数（默认为CPU核心数）')
    parser.add_argument('--dry-run', action='store_true',
                        help='仅显示将要处理的文件，不实际处理')
    
    args = parser.parse_args()
    
    # 检查输入目录
    subsets_path = Path(args.subsets_dir)
    if not subsets_path.exists():
        print(f"错误: subsets 目录不存在: {args.subsets_dir}")
        return
    
    # 查找所有 tar 文件
    tar_files = sorted(list(subsets_path.glob("*.tar")))
    if not tar_files:
        print(f"错误: 在 {args.subsets_dir} 中没有找到 tar 文件")
        return
    
    # 创建输出目录
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 设置工作进程数
    num_workers = args.workers or min(mp.cpu_count(), len(tar_files))
    
    print(f"发现 {len(tar_files)} 个 tar 文件")
    print(f"输出目录: {output_path}")
    print(f"并行工作进程: {num_workers}")
    if args.max_texts:
        print(f"每文件最大文本数: {args.max_texts}")
    
    if args.dry_run:
        print("\n将要处理的文件:")
        for i, tar_file in enumerate(tar_files, 1):
            output_file = output_path / f"{tar_file.stem}.jsonl"
            print(f"  {i:2d}. {tar_file.name} -> {output_file.name}")
        return
    
    print("\n开始并行处理...")
    print("=" * 60)
    
    start_time = time.time()
    
    # 准备工作参数
    work_args = [
        (tar_file, args.output_dir, args.max_texts, i % num_workers + 1)
        for i, tar_file in enumerate(tar_files)
    ]
    
    # 并行处理
    results = []
    completed = 0
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有任务
        future_to_tar = {
            executor.submit(process_single_tar_worker, arg): arg[0]
            for arg in work_args
        }
        
        # 收集结果
        for future in as_completed(future_to_tar):
            tar_file = future_to_tar[future]
            try:
                result = future.result()
                results.append(result)
                completed += 1
                
                progress = completed / len(tar_files) * 100
                print(f"进度: {completed}/{len(tar_files)} ({progress:.1f}%)")
                
            except Exception as e:
                print(f"任务异常 {tar_file.name}: {e}")
                results.append({
                    'status': 'exception',
                    'tar_file': tar_file.name,
                    'error': str(e),
                    'texts_count': 0,
                    'file_size': 0
                })
    
    # 统计结果
    end_time = time.time()
    total_time = end_time - start_time
    
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] in ['error', 'exception']]
    no_texts = [r for r in results if r['status'] == 'no_texts']
    
    total_texts = sum(r['texts_count'] for r in successful)
    total_size = sum(r['file_size'] for r in successful)
    
    print("\n" + "=" * 60)
    print("处理完成！")
    print(f"总耗时: {total_time:.1f} 秒")
    print(f"成功处理: {len(successful)} 个文件")
    print(f"无文本文件: {len(no_texts)} 个文件")
    print(f"失败文件: {len(failed)} 个文件")
    print(f"总文本数量: {total_texts:,}")
    print(f"总文件大小: {total_size:.2f} MB")
    
    if failed:
        print(f"\n失败的文件:")
        for r in failed:
            print(f"  - {r['tar_file']}: {r.get('error', 'Unknown error')}")
    
    print(f"\n输出目录: {output_path}")
    jsonl_files = list(output_path.glob("*.jsonl"))
    print(f"生成了 {len(jsonl_files)} 个 JSONL 文件")


if __name__ == "__main__":
    main()