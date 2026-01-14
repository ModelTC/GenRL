#!/usr/bin/env python3
"""
处理prompt数据，从源目录读取所有txt文件，随机选择1024条作为test，其余作为train
保存为JSON格式（每行一个JSON对象）
"""
import json
import os
import random
from pathlib import Path
from tqdm import tqdm


def read_prompt_file(file_path: Path) -> str:
    """读取单个prompt文件，返回完整内容"""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    return content


def process_prompts(
    source_dir: str, output_dir: str, test_size: int = 1024, seed: int = 42
):
    """处理prompt数据

    Args:
        source_dir: 源目录路径，包含所有txt文件
        output_dir: 输出目录路径
        test_size: test集大小
        seed: 随机种子
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)

    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)

    # 设置随机种子
    random.seed(seed)

    # 获取所有txt文件
    print(f"正在扫描源目录: {source_dir}")
    txt_files = list(source_path.glob("*.txt"))
    print(f"找到 {len(txt_files)} 个txt文件")

    if len(txt_files) < test_size:
        raise ValueError(f"文件数量 ({len(txt_files)}) 少于test_size ({test_size})")

    # 读取所有prompt
    print("正在读取所有prompt...")
    prompts = []
    for txt_file in tqdm(txt_files, desc="读取文件"):
        try:
            prompt = read_prompt_file(txt_file)
            if prompt:  # 只保留非空prompt
                prompts.append(prompt)
        except Exception as e:
            print(f"警告: 读取文件 {txt_file} 时出错: {e}")
            continue

    print(f"成功读取 {len(prompts)} 条prompt")

    # 随机打乱
    print("正在随机打乱...")
    random.shuffle(prompts)

    # 分割test和train
    test_prompts = prompts[:test_size]
    train_prompts = prompts[test_size:]

    print(f"Test集: {len(test_prompts)} 条")
    print(f"Train集: {len(train_prompts)} 条")

    # 保存为JSON格式（每行一个JSON对象）
    test_file = output_path / "test.json"
    train_file = output_path / "train.json"

    print(f"正在保存test集到 {test_file}...")
    with open(test_file, "w", encoding="utf-8") as f:
        for prompt in tqdm(test_prompts, desc="写入test"):
            json.dump({"prompt": prompt}, f, ensure_ascii=False)
            f.write("\n")

    print(f"正在保存train集到 {train_file}...")
    with open(train_file, "w", encoding="utf-8") as f:
        for prompt in tqdm(train_prompts, desc="写入train"):
            json.dump({"prompt": prompt}, f, ensure_ascii=False)
            f.write("\n")

    print("完成！")
    print(f"Test文件: {test_file} ({test_file.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"Train文件: {train_file} ({train_file.stat().st_size / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    source_dir = "/mnt/lm_data_afs/wuzhuguanyu/Self-Forcing/prompts/good_prompts/"
    output_dir = "/mnt/miaohua/huangyushi/VideoGRPO/datasets/filtered_prompts"

    process_prompts(
        source_dir=source_dir, output_dir=output_dir, test_size=1024, seed=42
    )
