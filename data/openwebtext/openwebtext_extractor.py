#!/usr/bin/env python3
"""
OpenWebText 数据提取工具

这个脚本可以：
1. 测试处理单个 tar 文件
2. 批量处理所有 tar 文件
3. 提取嵌套压缩文件中的文本并汇总为 JSONL 格式
"""

import os
import tarfile
import lzma
import json
import tempfile
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm
import logging


class OpenWebTextExtractor:
    def __init__(self, log_level=logging.INFO):
        """初始化提取器"""
        # 配置日志
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('openwebtext_extract.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def safe_extract_tar(self, tar_path, extract_path):
        """安全解压 tar 文件"""
        try:
            with tarfile.open(tar_path, 'r:*') as tar:
                # 检查所有成员文件的安全性
                safe_members = []
                for member in tar.getmembers():
                    if member.name.startswith('/') or '..' in member.name:
                        self.logger.warning(f"跳过不安全的路径: {member.name}")
                        continue
                    safe_members.append(member)
                
                # 提取安全的文件
                for member in tqdm(safe_members, desc="解压 tar", leave=False):
                    tar.extract(member, path=extract_path)
                    
            return True
        except Exception as e:
            self.logger.error(f"解压 tar 文件失败 {tar_path}: {e}")
            return False

    def extract_text_from_xz(self, xz_path):
        """从 xz 文件中提取文本内容"""
        try:
            # xz 文件包含多个文本文件，需要用 tar 再次解压
            with lzma.open(xz_path, 'rb') as xz_file:
                # 创建临时 tar 文件
                with tempfile.NamedTemporaryFile(suffix='.tar') as temp_tar:
                    temp_tar.write(xz_file.read())
                    temp_tar.flush()
                    
                    # 解压 tar 内容
                    texts = []
                    try:
                        with tarfile.open(temp_tar.name, 'r') as tar:
                            for member in tar.getmembers():
                                if member.isfile() and member.name.endswith('.txt'):
                                    try:
                                        f = tar.extractfile(member)
                                        if f:
                                            content = f.read().decode('utf-8', errors='ignore').strip()
                                            if content:
                                                texts.append(content)
                                    except Exception as e:
                                        self.logger.debug(f"读取文件 {member.name} 失败: {e}")
                    except tarfile.ReadError:
                        # 可能不是 tar 格式，直接读取为文本
                        with lzma.open(xz_path, 'rt', encoding='utf-8', errors='ignore') as f:
                            content = f.read().strip()
                            if content:
                                texts.append(content)
                    
                    return texts if texts else None
                    
        except Exception as e:
            self.logger.error(f"解压 xz 文件失败 {xz_path}: {e}")
            return None

    def process_single_tar(self, tar_path, output_jsonl, max_texts=None):
        """处理单个 tar 文件"""
        tar_path = Path(tar_path)
        self.logger.info(f"开始处理: {tar_path.name}")
        
        # 创建临时目录
        with tempfile.TemporaryDirectory(prefix=f"extract_{tar_path.stem}_") as temp_dir:
            temp_path = Path(temp_dir)
            
            # 1. 解压 tar 文件
            if not self.safe_extract_tar(tar_path, temp_path):
                return 0
            
            # 2. 查找所有 xz 文件
            xz_files = list(temp_path.rglob("*.xz"))
            self.logger.info(f"找到 {len(xz_files)} 个 xz 文件")
            
            if not xz_files:
                self.logger.warning(f"在 {tar_path.name} 中没有找到 xz 文件")
                return 0
            
            # 3. 处理每个 xz 文件
            texts_extracted = 0
            
            for xz_file in tqdm(xz_files, desc=f"处理 {tar_path.name}"):
                if max_texts and texts_extracted >= max_texts:
                    break
                
                # 提取文本内容（可能包含多个文本）
                text_contents = self.extract_text_from_xz(xz_file)
                
                if text_contents:
                    for content in text_contents:
                        if max_texts and texts_extracted >= max_texts:
                            break
                            
                        # 创建 JSON 记录
                        json_record = {
                            "text": content,
                            "metadata": {
                                "source_tar": tar_path.name,
                                "source_xz": xz_file.name,
                                "text_length": len(content),
                                "text_id": texts_extracted
                            }
                        }
                        
                        # 写入 JSONL 文件
                        output_jsonl.write(json.dumps(json_record, ensure_ascii=False) + '\n')
                        texts_extracted += 1
                        
                        if texts_extracted % 1000 == 0:
                            self.logger.info(f"已提取 {texts_extracted} 个文本")
            
            self.logger.info(f"从 {tar_path.name} 提取了 {texts_extracted} 个文本")
            return texts_extracted

    def extract_all(self, subsets_dir, output_dir, max_texts_per_tar=None):
        """提取所有 tar 文件中的文本，每个 tar 文件生成独立的 JSONL 文件"""
        subsets_path = Path(subsets_dir)
        output_path = Path(output_dir)
        
        # 创建输出目录
        output_path.mkdir(exist_ok=True)
        
        # 查找所有 tar 文件
        tar_files = sorted(list(subsets_path.glob("*.tar")))
        
        if not tar_files:
            self.logger.error(f"在 {subsets_path} 中没有找到 tar 文件")
            return
        
        self.logger.info(f"找到 {len(tar_files)} 个 tar 文件")
        self.logger.info(f"输出目录: {output_path}")
        
        # 处理所有文件
        total_texts = 0
        successful_files = 0
        failed_files = []
        
        for i, tar_file in enumerate(tar_files, 1):
            # 为每个 tar 文件生成独立的 JSONL 文件名
            output_file = output_path / f"{tar_file.stem}.jsonl"
            
            self.logger.info(f"[{i}/{len(tar_files)}] 开始处理: {tar_file.name}")
            
            try:
                with open(output_file, 'w', encoding='utf-8') as jsonl_file:
                    texts_count = self.process_single_tar(
                        tar_file, jsonl_file, max_texts_per_tar
                    )
                    total_texts += texts_count
                
                if texts_count > 0:
                    successful_files += 1
                    file_size = output_file.stat().st_size / (1024 * 1024)  # MB
                    self.logger.info(f"✓ 完成 {tar_file.name}: {texts_count} 个文本, {file_size:.2f} MB")
                else:
                    self.logger.warning(f"⚠ {tar_file.name}: 没有提取到文本")
                    
            except Exception as e:
                self.logger.error(f"✗ 处理 {tar_file.name} 失败: {e}")
                failed_files.append(tar_file.name)
                # 删除可能不完整的输出文件
                try:
                    if output_file.exists():
                        output_file.unlink()
                except:
                    pass
        
        # 统计结果
        self.logger.info("=" * 60)
        self.logger.info(f"处理完成！")
        self.logger.info(f"成功处理: {successful_files}/{len(tar_files)} 个文件")
        self.logger.info(f"总文本数量: {total_texts}")
        self.logger.info(f"输出目录: {output_path}")
        
        if failed_files:
            self.logger.warning(f"失败的文件: {', '.join(failed_files)}")
        
        # 显示输出目录信息
        jsonl_files = list(output_path.glob("*.jsonl"))
        total_size = sum(f.stat().st_size for f in jsonl_files) / (1024 * 1024)  # MB
        self.logger.info(f"生成了 {len(jsonl_files)} 个 JSONL 文件，总大小: {total_size:.2f} MB")

    def test_single_file(self, tar_file, output_file, max_texts=100):
        """测试处理单个文件"""
        self.logger.info("=== 测试模式 ===")
        
        with open(output_file, 'w', encoding='utf-8') as jsonl_file:
            texts_count = self.process_single_tar(tar_file, jsonl_file, max_texts)
        
        self.logger.info(f"测试完成，提取了 {texts_count} 个文本")
        
        # 显示前几条记录
        self.logger.info("前3条记录预览：")
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= 3:
                        break
                    record = json.loads(line)
                    text_preview = record['text'][:200] + "..." if len(record['text']) > 200 else record['text']
                    self.logger.info(f"记录 {i+1}: {text_preview}")
        except Exception as e:
            self.logger.error(f"读取输出文件失败: {e}")


def main():
    parser = argparse.ArgumentParser(description="OpenWebText 数据提取工具")
    parser.add_argument('--mode', choices=['test', 'all'], default='test',
                        help='运行模式: test (测试单个文件) 或 all (处理所有文件)')
    parser.add_argument('--subsets-dir', default='/mnt/liaozy25/openwebtext/subsets',
                        help='subsets 目录路径')
    parser.add_argument('--output', default='/mnt/liaozy25/openwebtext/extracted_texts',
                        help='输出目录路径（all模式）或文件路径（test模式）')
    parser.add_argument('--test-file', default=None,
                        help='测试模式下指定的 tar 文件路径')
    parser.add_argument('--max-texts', type=int, default=None,
                        help='最大提取文本数量（用于测试）')
    parser.add_argument('--verbose', action='store_true',
                        help='显示详细日志')
    
    args = parser.parse_args()
    
    # 初始化提取器
    log_level = logging.DEBUG if args.verbose else logging.INFO
    extractor = OpenWebTextExtractor(log_level)
    
    if args.mode == 'test':
        # 测试模式
        if args.test_file:
            test_file = args.test_file
        else:
            # 使用第一个 tar 文件进行测试
            subsets_path = Path(args.subsets_dir)
            tar_files = list(subsets_path.glob("*.tar"))
            if not tar_files:
                print(f"在 {args.subsets_dir} 中没有找到 tar 文件")
                return
            test_file = tar_files[0]
        
        # 测试模式使用文件路径
        if not args.output.endswith('.jsonl'):
            test_output = args.output + '_test.jsonl'
        else:
            test_output = args.output
        
        max_texts = args.max_texts or 100
        
        extractor.test_single_file(test_file, test_output, max_texts)
        
    else:
        # 处理所有文件，使用目录路径
        output_dir = args.output
        if args.output.endswith('.jsonl'):
            output_dir = args.output.replace('.jsonl', '_dir')
        
        extractor.extract_all(args.subsets_dir, output_dir, args.max_texts)


if __name__ == "__main__":
    main()