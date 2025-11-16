#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-SoVITS 训练管理器
负责完整的训练流程：数据预处理 → Stage 1 训练 → Stage 2 训练
"""

import os
import sys
import json
import yaml
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import time


class GPTSoVITSTrainer:
    """GPT-SoVITS 完整训练管理器"""
    
    def __init__(self, sovits_path: str = None):
        """
        初始化训练器
        
        Args:
            sovits_path: GPT-SoVITS 项目路径
        """
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # GPT-SoVITS 路径
        if sovits_path is None:
            self.sovits_path = os.path.join(self.base_dir, "GPT-SoVITS-main")
        else:
            self.sovits_path = sovits_path
            
        if not os.path.exists(self.sovits_path):
            raise FileNotFoundError(f"GPT-SoVITS 路径不存在: {self.sovits_path}")
        
        # Python 解释器
        self.python_exec = sys.executable
        
        # 训练输出根目录
        self.exp_root = os.path.join(self.base_dir, "training_experiments")
        os.makedirs(self.exp_root, exist_ok=True)
        
        # 预训练模型路径
        self.pretrained_models = {
            "bert": os.path.join(self.sovits_path, "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"),
            "ssl": os.path.join(self.sovits_path, "GPT_SoVITS/pretrained_models/chinese-hubert-base"),
            # v2 版本模型路径
            "s1_v2": os.path.join(self.sovits_path, "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"),
            "s2G_v2": os.path.join(self.sovits_path, "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"),
            "s2D_v2": os.path.join(self.sovits_path, "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2D2333k.pth"),
            "s2_config": os.path.join(self.sovits_path, "GPT_SoVITS/configs/s2.json"),
        }
        
        print(f"✅ GPT-SoVITS 训练器初始化完成")
        print(f"📁 GPT-SoVITS 路径: {self.sovits_path}")
        print(f"📁 实验输出路径: {self.exp_root}")
    
    def prepare_training_data(
        self, 
        speaker_name: str, 
        audio_files: List[Dict],
        audio_text_map: Optional[Dict[str, str]] = None
    ) -> str:
        """
        准备训练数据
        
        Args:
            speaker_name: 说话者名称
            audio_files: 音频文件列表 [{"path": "xxx.wav", "duration": 10.5}, ...]
            audio_text_map: 音频文件到文本的映射 {"xxx.wav": "这是文本内容"}
            
        Returns:
            实验目录路径
        """
        print(f"\n🎯 开始准备训练数据: {speaker_name}")
        print(f"   音频文件数量: {len(audio_files)}")
        
        # 创建实验目录
        exp_dir = os.path.join(self.exp_root, speaker_name)
        os.makedirs(exp_dir, exist_ok=True)
        
        # 创建输入目录
        input_wav_dir = os.path.join(exp_dir, "input_wavs")
        os.makedirs(input_wav_dir, exist_ok=True)
        
        # 复制音频文件
        print(f"📋 复制音频文件到: {input_wav_dir}")
        copied_files = []  # 记录实际复制的文件名
        for i, audio_info in enumerate(audio_files):
            src_path = audio_info["path"]
            if not os.path.exists(src_path):
                print(f"   ⚠️  跳过不存在的文件: {src_path}")
                continue
                
            # 使用统一的命名格式
            ext = os.path.splitext(src_path)[1]
            dst_filename = f"{speaker_name}_{i:04d}{ext}"
            dst_path = os.path.join(input_wav_dir, dst_filename)
            
            shutil.copy2(src_path, dst_path)
            copied_files.append(dst_filename)  # 保存实际文件名
            print(f"   ✅ {i+1}/{len(audio_files)}: {dst_filename}")
        
        # 创建文本标注文件
        text_file = os.path.join(exp_dir, "input_text.txt")
        
        # 检查是否已存在有效的文本标注文件
        if os.path.exists(text_file):
            # 检查文件内容是否有效（不是占位文本）
            with open(text_file, "r", encoding="utf-8") as f:
                first_line = f.readline()
                if "这是一段训练语音" not in first_line and len(first_line.strip()) > 50:
                    print(f"📝 使用已存在的文本标注文件")
                    print(f"✅ 数据准备完成: {exp_dir}")
                    return exp_dir
        
        if audio_text_map:
            # 使用提供的文本映射
            print(f"📝 使用提供的文本标注")
            with open(text_file, "w", encoding="utf-8") as f:
                for i, (audio_info, dst_filename) in enumerate(zip(audio_files, copied_files)):
                    basename = os.path.basename(audio_info["path"])
                    text = audio_text_map.get(basename, "这是一段语音。")
                    # 格式: 文件名|说话者|语言|文本
                    f.write(f"{input_wav_dir}/{dst_filename}|{speaker_name}|ZH|{text}\n")
        else:
            # 使用 ASR 自动识别文本
            print(f"📝 使用 ASR 自动识别音频文本...")
            asr_success = False
            
            try:
                # 动态导入 ASR 模块
                import sys
                # 将 GPT-SoVITS 根目录加入 sys.path（不是 tools/asr 子目录！）
                if self.sovits_path not in sys.path:
                    sys.path.insert(0, self.sovits_path)
                
                # 导入 FasterWhisper ASR（现在从 tools.asr 模块导入）
                from tools.asr.fasterwhisper_asr import execute_asr
                
                # 调用 ASR（会自动生成 {speaker_name}.list 文件）
                print(f"   🎤 正在识别音频内容（可能需要几分钟）...")
                # 使用绝对路径或模型名称
                model_path = os.path.join(self.sovits_path, "tools", "asr", "models", "faster-whisper-large-v3")
                if not os.path.exists(model_path):
                    # 如果本地路径不存在，使用模型名称让 ASR 自动下载
                    model_path = "large-v3"
                    print(f"   📥 本地模型不存在，将自动下载 faster-whisper-{model_path}...")
                
                asr_output = execute_asr(
                    input_folder=input_wav_dir,
                    output_folder=exp_dir,
                    model_path=model_path,
                    language="zh",  # 中文
                    precision="float16"
                )
                
                # 将 ASR 输出复制为 input_text.txt
                if os.path.exists(asr_output):
                    shutil.copy(asr_output, text_file)
                    print(f"   ✅ ASR 识别完成，已生成文本标注")
                    asr_success = True
                else:
                    print(f"   ⚠️  ASR 输出文件不存在: {asr_output}")
                    
            except Exception as e:
                print(f"   ⚠️  ASR 识别失败: {e}")
                import traceback
                traceback.print_exc()
            
            # 如果 ASR 失败，使用占位文本（会导致训练失败）
            if not asr_success:
                print(f"   ❌ 回退到占位文本（训练将失败，请安装 ASR 依赖或手动提供文本）")
                with open(text_file, "w", encoding="utf-8") as f:
                    for dst_filename in copied_files:
                        f.write(f"{input_wav_dir}/{dst_filename}|{speaker_name}|ZH|这是一段训练语音。\n")
        
        print(f"✅ 数据准备完成: {exp_dir}")
        return exp_dir
    
    def run_data_preprocessing(self, exp_dir: str, speaker_name: str) -> bool:
        """
        运行数据预处理（步骤 1a, 1b, 1c）
        
        Args:
            exp_dir: 实验目录
            speaker_name: 说话者名称
            
        Returns:
            是否成功
        """
        print(f"\n🔧 开始数据预处理...")
        
        # 清理之前的预处理输出（避免脚本跳过已存在但不完整的文件）
        for subdir in ["2-name2text", "3-bert", "4-cnhubert", "5-wav32k", "6-name2semantic"]:
            dir_path = os.path.join(exp_dir, subdir)
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
                print(f"   🧹 清理旧数据: {subdir}")
        
        # 清理旧的训练 checkpoint（避免 epoch 冲突）
        for checkpoint_dir in ["logs_s1/ckpt", "logs_s2_v2"]:
            dir_path = os.path.join(exp_dir, checkpoint_dir)
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
                print(f"   🧹 清理旧 checkpoint: {checkpoint_dir}")
        
        # 删除旧的 tsv/txt 文件
        for old_file in ["2-name2text-0.txt", "6-name2semantic-0.tsv"]:
            file_path = os.path.join(exp_dir, old_file)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"   🧹 清理旧文件: {old_file}")
        
        input_wav_dir = os.path.join(exp_dir, "input_wavs")
        input_text = os.path.join(exp_dir, "input_text.txt")
        
        if not os.path.exists(input_text):
            print(f"❌ 文本文件不存在: {input_text}")
            return False
        
        # 设置环境变量
        env = os.environ.copy()
        env.update({
            "inp_text": input_text,
            "inp_wav_dir": input_wav_dir,
            "exp_name": speaker_name,
            "opt_dir": exp_dir,
            "i_part": "0",
            "all_parts": "1",
            "bert_pretrained_dir": self.pretrained_models["bert"],
            "cnhubert_base_dir": self.pretrained_models["ssl"],  # 步骤 1b 需要
            "pretrained_s2G": self.pretrained_models["s2G_v2"],   # 步骤 1c 需要
            "s2config_path": self.pretrained_models["s2_config"], # 步骤 1c 需要
            "is_half": "True",
            # 添加 PYTHONPATH 以便导入 GPT-SoVITS 的内部模块
            "PYTHONPATH": self.sovits_path + (f":{env.get('PYTHONPATH', '')}" if env.get('PYTHONPATH') else ""),
        })
        
        # 步骤 1a: 文本处理和 BERT 特征提取
        print(f"   [1/3] 文本处理和 BERT 特征提取...")
        script_1a = os.path.join(self.sovits_path, "GPT_SoVITS/prepare_datasets/1-get-text.py")
        
        # 使用包装脚本来正确设置环境
        wrapper_script = os.path.join(self.base_dir, "run_preprocessing_script.py")
        cmd_1a = [self.python_exec, wrapper_script, self.sovits_path, script_1a]
        
        result = subprocess.run(cmd_1a, env=env, capture_output=True, text=True)
        
        # 显示输出（用于调试）
        if result.stdout:
            print(f"   📋 步骤 1a 输出:")
            for line in result.stdout.strip().split('\n')[:10]:  # 显示前10行
                print(f"      {line}")
        
        if result.returncode != 0:
            print(f"   ❌ 步骤 1a 失败:")
            print(result.stderr)
            return False
        
        # 验证输出文件
        bert_dir = os.path.join(exp_dir, "2-name2text")
        if os.path.exists(bert_dir):
            bert_files = [f for f in os.listdir(bert_dir) if f.endswith('.bert.pt')]
            print(f"   ✅ 步骤 1a 完成 ({len(bert_files)} .bert.pt)")
        else:
            print(f"   ✅ 步骤 1a 完成")
        
        # 步骤 1b: SSL 特征提取
        print(f"   [2/3] SSL 特征提取（HuBERT）...")
        script_1b = os.path.join(self.sovits_path, "GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py")
        
        # 使用包装脚本来正确设置环境
        wrapper_script = os.path.join(self.base_dir, "run_preprocessing_script.py")
        cmd_1b = [self.python_exec, wrapper_script, self.sovits_path, script_1b]
        
        result = subprocess.run(cmd_1b, env=env, capture_output=True, text=True)
        
        # 显示输出（即使成功也显示，用于调试）
        if result.stdout:
            print(f"   📋 步骤 1b 输出:")
            for line in result.stdout.strip().split('\n')[:20]:  # 显示前20行
                print(f"      {line}")
        
        if result.returncode != 0:
            print(f"   ❌ 步骤 1b 失败:")
            if result.stderr:
                print(result.stderr)
            return False
        
        # 验证输出文件
        hubert_dir = os.path.join(exp_dir, "4-cnhubert")
        wav32_dir = os.path.join(exp_dir, "5-wav32k")
        hubert_files = [f for f in os.listdir(hubert_dir) if f.endswith('.pt')] if os.path.exists(hubert_dir) else []
        # 注意：wav32k 目录中的文件可能有各种扩展名（.mp3, .wav 等），所以统计所有文件
        wav32_files = os.listdir(wav32_dir) if os.path.exists(wav32_dir) else []
        
        print(f"   ✅ 步骤 1b 完成 ({len(hubert_files)} .pt, {len(wav32_files)} 音频文件)")
        
        # 步骤 1c: 语义特征提取
        print(f"   [3/3] 语义特征提取...")
        
        # 先检查步骤 1b 的输出完整性
        print(f"   🔍 检查步骤 1b 输出:")
        input_text_file = os.path.join(exp_dir, "input_text.txt")
        with open(input_text_file, "r", encoding="utf-8") as f:
            expected_files = []
            for line in f.read().strip().split('\n'):
                if line:
                    wav_name = os.path.basename(line.split('|')[0])
                    expected_files.append(wav_name)
        
        missing_pt = []
        for wav_name in expected_files:
            pt_file = os.path.join(exp_dir, "4-cnhubert", f"{wav_name}.pt")
            if not os.path.exists(pt_file):
                missing_pt.append(wav_name)
        
        if missing_pt:
            print(f"   ⚠️  警告: {len(missing_pt)} 个文件缺少 .pt 特征文件:")
            for fname in missing_pt[:5]:  # 只显示前5个
                print(f"      - {fname}")
        else:
            print(f"   ✅ 所有 {len(expected_files)} 个文件都有 .pt 特征")
        
        script_1c = os.path.join(self.sovits_path, "GPT_SoVITS/prepare_datasets/3-get-semantic.py")
        
        # 使用包装脚本来正确设置环境
        wrapper_script = os.path.join(self.base_dir, "run_preprocessing_script.py")
        cmd_1c = [self.python_exec, wrapper_script, self.sovits_path, script_1c]
        
        result = subprocess.run(cmd_1c, env=env, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"   ❌ 步骤 1c 失败:")
            print(result.stderr)
            return False
        
        # 输出标准输出以便调试
        if result.stdout:
            print(f"   📋 步骤 1c 输出:")
            for line in result.stdout.strip().split('\n')[:10]:  # 只显示前10行
                print(f"      {line}")
        
        # 验证输出文件是否存在且非空
        semantic_file = os.path.join(exp_dir, "6-name2semantic-0.tsv")
        
        # 关键修复：检查并修正 semantic 文件格式
        # GPT-SoVITS 的 dataset.py 期望文件没有 header，但 pandas 默认会把第一行当作 header
        # 我们不添加列名，而是确保 GPT-SoVITS 正确读取数据
        if not os.path.exists(semantic_file):
            print(f"   ❌ 步骤 1c 失败: 输出文件不存在: {semantic_file}")
            return False
        
        # 检查生成的 semantic 数据行数（使用与训练相同的方式加载）
        import pandas as pd
        try:
            semantic_df = pd.read_csv(semantic_file, delimiter="\t", encoding="utf-8", header=None)
            semantic_count = len(semantic_df)
        except Exception as e:
            print(f"   ⚠️  警告: 无法读取 semantic 文件: {e}")
            semantic_count = 0
        
        print(f"   📊 生成了 {semantic_count} 条有效 semantic 数据（期望 {len(expected_files)} 条）")
        
        if semantic_count != len(expected_files):
            print(f"   ⚠️  警告: semantic 数据数量与预期不符！")
            print(f"      差异: {len(expected_files) - semantic_count} 条数据缺失或无效")
            
            # 显示前几行数据以便诊断
            print(f"   🔍 检查文件内容:")
            with open(semantic_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                print(f"      总行数（包含标题/空行）: {len(lines)}")
                if len(lines) > 0:
                    print(f"      第一行: {lines[0][:100] if len(lines[0]) > 100 else lines[0].strip()}")
                if len(lines) > 1:
                    print(f"      第二行: {lines[1][:100] if len(lines[1]) > 100 else lines[1].strip()}")
                if len(lines) > len(expected_files):
                    print(f"      最后一行: {lines[-1][:100] if len(lines[-1]) > 100 else lines[-1].strip()}")
        
        if os.path.getsize(semantic_file) == 0:
            print(f"   ❌ 步骤 1c 失败: 输出文件为空: {semantic_file}")
            print(f"   💡 提示: 检查是否有足够的音频样本和正确的文本标注")
            
            # 诊断信息
            print(f"\n   🔍 诊断信息:")
            
            # 检查 input_text.txt
            input_text_file = os.path.join(exp_dir, "input_text.txt")
            if os.path.exists(input_text_file):
                with open(input_text_file, "r", encoding="utf-8") as f:
                    lines = f.read().strip().split('\n')
                    print(f"   📄 input_text.txt: {len(lines)} 行")
                    if lines:
                        print(f"      第一行: {lines[0][:100]}")
            else:
                print(f"   ❌ input_text.txt 不存在")
            
            # 检查 4-cnhubert 目录
            hubert_dir = os.path.join(exp_dir, "4-cnhubert")
            if os.path.exists(hubert_dir):
                hubert_files = [f for f in os.listdir(hubert_dir) if f.endswith('.pt')]
                print(f"   📁 4-cnhubert: {len(hubert_files)} 个 .pt 文件")
            else:
                print(f"   ❌ 4-cnhubert 目录不存在")
            
            # 检查 5-wav32k 目录
            wav32_dir = os.path.join(exp_dir, "5-wav32k")
            if os.path.exists(wav32_dir):
                wav_files = os.listdir(wav32_dir)
                print(f"   📁 5-wav32k: {len(wav_files)} 个音频文件")
            else:
                print(f"   ❌ 5-wav32k 目录不存在")
            
            # 输出脚本的标准输出和错误（即使为空也显示）
            print(f"\n   📋 步骤 1c 标准输出:")
            if result.stdout:
                print(result.stdout)
            else:
                print("      (无输出)")
            
            print(f"\n   📋 步骤 1c 错误输出:")
            if result.stderr:
                print(result.stderr)
            else:
                print("      (无错误)")
            
            return False
        
        print(f"   ✅ 步骤 1c 完成")
        
        print(f"✅ 数据预处理完成")
        return True
    
    def train_stage1_gpt(
        self, 
        exp_dir: str, 
        speaker_name: str,
        epochs: int = 15,
        batch_size: int = 8,
        save_every_epoch: int = 5
    ) -> Optional[str]:
        """
        训练 Stage 1: GPT 模型（文本到语义）
        
        Args:
            exp_dir: 实验目录
            speaker_name: 说话者名称
            epochs: 训练轮数
            batch_size: 批次大小
            save_every_epoch: 每隔几个 epoch 保存一次
            
        Returns:
            训练好的模型路径，如果失败返回 None
        """
        print(f"\n🎓 开始 Stage 1 训练（GPT 模型）...")
        print(f"   Epochs: {epochs}, Batch Size: {batch_size}")
        
        # 训练前数据验证（使用与训练相同的方式加载）
        print(f"\n   🔍 验证训练数据...")
        semantic_file = os.path.join(exp_dir, "6-name2semantic-0.tsv")
        phoneme_file = os.path.join(exp_dir, "2-name2text-0.txt")
        
        # 使用 pandas 加载 semantic 数据（与训练脚本一致）
        import pandas as pd
        try:
            semantic_df = pd.read_csv(semantic_file, delimiter="\t", encoding="utf-8", header=None)
            semantic_count = len(semantic_df)
        except Exception as e:
            print(f"   ❌ 错误: 无法加载 semantic 数据: {e}")
            return None
        
        # 加载 phoneme 数据
        phoneme_data = {}
        with open(phoneme_file, "r", encoding="utf8") as f:
            lines = f.read().strip("\n").split("\n")
            for line in lines:
                tmp = line.split("\t")
                if len(tmp) == 4:
                    phoneme_data[tmp[0]] = [tmp[1], tmp[2], tmp[3]]
        phoneme_count = len(phoneme_data)
        
        print(f"   📊 Semantic 数据: {semantic_count} 条")
        print(f"   📊 Phoneme 数据: {phoneme_count} 条")
        
        if semantic_count != phoneme_count:
            print(f"   ❌ 错误: 数据数量不匹配！")
            print(f"      Semantic: {semantic_count} 条")
            print(f"      Phoneme: {phoneme_count} 条")
            print(f"      差异: {abs(semantic_count - phoneme_count)} 条")
            
            # 列出不匹配的文件（semantic_df 没有 header，使用数字索引）
            if semantic_count < phoneme_count:
                semantic_names = set(semantic_df.iloc[:, 0].tolist())  # 第 0 列是文件名
                phoneme_names = set(phoneme_data.keys())
                missing = phoneme_names - semantic_names
                if missing:
                    print(f"      缺少 semantic 数据的文件:")
                    for fname in list(missing)[:5]:
                        print(f"         - {fname}")
            elif semantic_count > phoneme_count:
                semantic_names = set(semantic_df.iloc[:, 0].tolist())  # 第 0 列是文件名
                phoneme_names = set(phoneme_data.keys())
                extra = semantic_names - phoneme_names
                if extra:
                    print(f"      缺少 phoneme 数据的文件:")
                    for fname in list(extra)[:5]:
                        print(f"         - {fname}")
            
            print(f"\n   💡 解决方案:")
            print(f"      1. 检查哪个文件导致数据不匹配（见上方列表）")
            print(f"      2. 删除该说话者，重新上传音频文件")
            print(f"      3. 确保所有音频文件格式正确、时长合适（1-60秒）")
            print(f"      4. 如果问题持续，查看预处理日志的详细输出")
            
            return None  # 数据不匹配时停止训练
        
        if semantic_count == 0:
            print(f"   ❌ 错误: Semantic 数据为空，无法训练")
            return None
        
        # 加载配置模板（使用 v2 版本配置）
        config_template = os.path.join(self.sovits_path, "GPT_SoVITS/configs/s1longer-v2.yaml")
        
        with open(config_template, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        # 修改配置
        config["train"]["epochs"] = epochs
        config["train"]["batch_size"] = batch_size
        config["train"]["save_every_n_epoch"] = save_every_epoch
        config["train"]["if_save_latest"] = True
        config["train"]["if_save_every_weights"] = True
        config["train"]["half_weights_save_dir"] = os.path.join(exp_dir, "logs_s1")  # 半精度权重保存目录
        config["train"]["exp_name"] = speaker_name
        config["pretrained_s1"] = self.pretrained_models["s1_v2"]
        
        # 设置数据路径（注意：预处理脚本生成的文件名包含 -0 后缀）
        config["train_semantic_path"] = os.path.join(exp_dir, "6-name2semantic-0.tsv")
        config["train_phoneme_path"] = os.path.join(exp_dir, "2-name2text-0.txt")  # 注意：脚本生成的文件名有 -0 后缀
        config["output_dir"] = os.path.join(exp_dir, "logs_s1")
        
        # 保存临时配置
        temp_config = os.path.join(exp_dir, "s1_config.yaml")
        with open(temp_config, "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True)
        
        # 执行训练
        script_s1 = os.path.join(self.sovits_path, "GPT_SoVITS/s1_train.py")
        cmd = [self.python_exec, script_s1, "--config_file", temp_config]
        
        # 设置环境变量和工作目录
        env = os.environ.copy()
        env["PYTHONPATH"] = self.sovits_path + (f":{env.get('PYTHONPATH', '')}" if env.get('PYTHONPATH') else "")
        
        print(f"   🚀 开始训练...")
        print(f"   命令: {' '.join(cmd)}")
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            cwd=self.sovits_path
        )
        
        # 实时输出日志
        for line in process.stdout:
            print(f"   {line.strip()}")
        
        process.wait()
        
        if process.returncode != 0:
            print(f"   ❌ Stage 1 训练失败")
            return None
        
        # 查找训练好的模型
        logs_dir = os.path.join(exp_dir, "logs_s1")
        if not os.path.exists(logs_dir):
            print(f"   ❌ 模型目录不存在: {logs_dir}")
            return None
        
        # 查找最新的 checkpoint
        ckpt_files = [f for f in os.listdir(logs_dir) if f.endswith(".ckpt")]
        if not ckpt_files:
            print(f"   ❌ 未找到训练好的模型")
            return None
        
        # 按文件修改时间排序，获取最新的
        ckpt_files.sort(key=lambda x: os.path.getmtime(os.path.join(logs_dir, x)), reverse=True)
        model_path = os.path.join(logs_dir, ckpt_files[0])
        
        print(f"   ✅ Stage 1 训练完成")
        print(f"   📦 模型路径: {model_path}")
        
        return model_path
    
    def _convert_checkpoint_to_weight(
        self, 
        checkpoint_path: str, 
        output_dir: str, 
        speaker_name: str,
        config_path: str
    ) -> Optional[str]:
        """
        将训练 checkpoint 转换为包含 config 的完整权重文件
        
        Args:
            checkpoint_path: checkpoint 文件路径
            output_dir: 输出目录
            speaker_name: 说话者名称
            config_path: 配置文件路径
            
        Returns:
            转换后的权重文件路径，失败返回 None
        """
        try:
            import torch
            import json
            from collections import OrderedDict
            
            # 加载 checkpoint
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            
            # 加载配置
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 创建权重字典
            opt = OrderedDict()
            opt["weight"] = OrderedDict()
            
            # 从 checkpoint 提取模型权重
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
            
            # 过滤并转换权重
            for key, value in state_dict.items():
                if "enc_q" in key:
                    continue
                try:
                    opt["weight"][key] = value.half()
                except:
                    opt["weight"][key] = value
            
            # 添加配置（转换为 HParams 格式）
            class HParams:
                def __init__(self, **kwargs):
                    for k, v in kwargs.items():
                        if isinstance(v, dict):
                            setattr(self, k, HParams(**v))
                        else:
                            setattr(self, k, v)
            
            # 将 config 转换为 HParams
            hps = HParams(**config)
            opt["config"] = hps
            
            # 添加训练信息
            epoch = checkpoint.get("epoch", 8)
            iteration = checkpoint.get("iteration", checkpoint.get("step", 0))
            opt["info"] = f"{epoch}epoch_{iteration}iteration"
            
            # 生成输出文件名
            output_filename = f"{speaker_name}_e{epoch}.pth"
            output_path = os.path.join(output_dir, output_filename)
            
            # 保存权重文件
            torch.save(opt, output_path)
            
            print(f"   ✅ 转换成功: {output_filename}")
            return output_path
            
        except Exception as e:
            print(f"   ❌ 转换失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def train_stage2_sovits(
        self,
        exp_dir: str,
        speaker_name: str,
        epochs: int = 10,
        batch_size: int = 8,
        save_every_epoch: int = 4
    ) -> Optional[str]:
        """
        训练 Stage 2: SoVITS 模型（音色克隆）
        
        Args:
            exp_dir: 实验目录
            speaker_name: 说话者名称
            epochs: 训练轮数
            batch_size: 批次大小
            save_every_epoch: 每隔几个 epoch 保存一次
            
        Returns:
            训练好的模型路径，如果失败返回 None
        """
        print(f"\n🎤 开始 Stage 2 训练（SoVITS 模型）...")
        print(f"   Epochs: {epochs}, Batch Size: {batch_size}")
        
        # 加载配置模板
        config_template = os.path.join(self.sovits_path, "GPT_SoVITS/configs/s2.json")
        
        with open(config_template, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        # 修改配置
        config["train"]["epochs"] = epochs
        config["train"]["batch_size"] = batch_size
        config["train"]["save_every_epoch"] = save_every_epoch
        config["train"]["if_save_latest"] = True
        config["train"]["if_save_every_weights"] = True
        config["train"]["name"] = speaker_name
        config["train"]["gpu_numbers"] = "0"  # 使用第一个 GPU
        config["train"]["pretrained_s2G"] = self.pretrained_models["s2G_v2"]
        config["train"]["pretrained_s2D"] = self.pretrained_models["s2D_v2"]
        
        # 设置顶层 name（s2_train.py 需要）
        config["name"] = speaker_name
        
        # 设置模型版本（v2 模型）
        config["model"]["version"] = "v2"
        
        # 设置保存权重目录（process_ckpt.py 需要）
        config["save_weight_dir"] = os.path.join(exp_dir, f"logs_s2_{config['model']['version']}")
        
        # 设置数据路径（exp_dir 应该在 data 字段下）
        config["data"]["exp_dir"] = exp_dir
        config["data"]["training_files"] = os.path.join(exp_dir, "2-name2text.txt")
        config["data"]["validation_files"] = os.path.join(exp_dir, "2-name2text.txt")
        
        # 设置 wav 文件路径（Stage 2 需要）
        config["data"]["wav_path"] = os.path.join(exp_dir, "5-wav32k")
        
        # 确保必需的文件和目录存在
        wav_dir = os.path.join(exp_dir, "5-wav32k")
        if not os.path.exists(wav_dir):
            raise ValueError(f"WAV 目录不存在: {wav_dir}，请确保数据预处理已完成")
        
        cnhubert_dir = os.path.join(exp_dir, "4-cnhubert")
        if not os.path.exists(cnhubert_dir):
            raise ValueError(f"HuBERT 目录不存在: {cnhubert_dir}，请确保数据预处理已完成")
        
        # 确保 2-name2text.txt 存在（从 2-name2text-0.txt 复制）
        name2text_src = os.path.join(exp_dir, "2-name2text-0.txt")
        name2text_dst = os.path.join(exp_dir, "2-name2text.txt")
        if not os.path.exists(name2text_dst):
            if os.path.exists(name2text_src):
                shutil.copy2(name2text_src, name2text_dst)
                print(f"   📋 创建文本标注文件: {name2text_dst}")
            else:
                raise ValueError(f"文本标注文件不存在: {name2text_src}，请确保数据预处理已完成")
        
        # 创建 checkpoint 保存目录
        logs_s2_dir = os.path.join(exp_dir, f"logs_s2_{config['model']['version']}")
        os.makedirs(logs_s2_dir, exist_ok=True)
        print(f"   📁 创建 checkpoint 目录: {logs_s2_dir}")
        
        # 保存临时配置
        temp_config = os.path.join(exp_dir, "s2_config.json")
        with open(temp_config, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        # 执行训练
        script_s2 = os.path.join(self.sovits_path, "GPT_SoVITS/s2_train.py")
        cmd = [self.python_exec, script_s2, "--config", temp_config]
        
        # 设置环境变量和工作目录
        env = os.environ.copy()
        env["PYTHONPATH"] = self.sovits_path + (f":{env.get('PYTHONPATH', '')}" if env.get('PYTHONPATH') else "")
        
        print(f"   🚀 开始训练...")
        print(f"   命令: {' '.join(cmd)}")
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            cwd=self.sovits_path
        )
        
        # 实时输出日志
        for line in process.stdout:
            print(f"   {line.strip()}")
        
        process.wait()
        
        if process.returncode != 0:
            print(f"   ❌ Stage 2 训练失败")
            return None
        
        # 查找训练好的模型（使用 v2 模型路径）
        logs_dir = os.path.join(exp_dir, "logs_s2_v2")
        if not os.path.exists(logs_dir):
            print(f"   ❌ 模型目录不存在: {logs_dir}")
            return None
        
        # 优先查找最终导出的权重文件（包含 config 的完整模型）
        # 格式如: 三奶奶_e8.pth, 三奶奶_e10.pth
        weight_files = [
            f for f in os.listdir(logs_dir) 
            if f.endswith(".pth") 
            and speaker_name in f 
            and "_e" in f
            and not f.startswith("G_")
            and not f.startswith("D_")
        ]
        
        if weight_files:
            # 找到了最终权重文件，按文件名中的 epoch 排序
            weight_files.sort(reverse=True)
            model_path = os.path.join(logs_dir, weight_files[0])
            print(f"   ✅ Stage 2 训练完成")
            print(f"   📦 找到最终权重文件: {model_path}")
            return model_path
        
        # 如果没有找到最终权重文件，查找 checkpoint 文件
        print(f"   ⚠️  未找到最终权重文件，尝试使用 checkpoint")
        g_files = [f for f in os.listdir(logs_dir) if f.startswith("G_") and f.endswith(".pth")]
        if not g_files:
            print(f"   ❌ 未找到训练好的模型")
            return None
        
        # 按文件修改时间排序，获取最新的
        g_files.sort(key=lambda x: os.path.getmtime(os.path.join(logs_dir, x)), reverse=True)
        checkpoint_path = os.path.join(logs_dir, g_files[0])
        
        # 尝试从 checkpoint 转换为权重文件
        print(f"   🔧 从 checkpoint 转换为权重文件...")
        weight_path = self._convert_checkpoint_to_weight(
            checkpoint_path, 
            logs_dir, 
            speaker_name, 
            config_path=os.path.join(exp_dir, "s2_config.json")
        )
        
        if weight_path:
            print(f"   ✅ Stage 2 训练完成")
            print(f"   📦 模型路径: {weight_path}")
            return weight_path
        else:
            print(f"   ⚠️  转换失败，使用 checkpoint（可能无法用于推理）")
            print(f"   📦 模型路径: {checkpoint_path}")
            return checkpoint_path
    
    def train_speaker_complete(
        self,
        speaker_name: str,
        audio_files: List[Dict],
        audio_text_map: Optional[Dict[str, str]] = None,
        s1_epochs: int = 15,
        s2_epochs: int = 10,
        batch_size: int = 8
    ) -> Dict:
        """
        完整训练流程：数据准备 → 预处理 → Stage 1 → Stage 2
        
        Args:
            speaker_name: 说话者名称
            audio_files: 音频文件列表
            audio_text_map: 音频文本映射（可选）
            s1_epochs: Stage 1 训练轮数
            s2_epochs: Stage 2 训练轮数
            batch_size: 批次大小
            
        Returns:
            训练结果字典
        """
        start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"🚀 开始完整训练流程: {speaker_name}")
        print(f"{'='*60}")
        
        result = {
            "speaker_name": speaker_name,
            "status": "failed",
            "start_time": datetime.now().isoformat(),
            "gpt_model": None,
            "sovits_model": None,
            "exp_dir": None,
            "error": None
        }
        
        try:
            # 步骤 1: 准备数据
            exp_dir = self.prepare_training_data(speaker_name, audio_files, audio_text_map)
            result["exp_dir"] = exp_dir
            
            # 步骤 2: 数据预处理
            if not self.run_data_preprocessing(exp_dir, speaker_name):
                result["error"] = "数据预处理失败"
                return result
            
            # 步骤 3: Stage 1 训练
            gpt_model = self.train_stage1_gpt(exp_dir, speaker_name, s1_epochs, batch_size)
            if gpt_model is None:
                result["error"] = "Stage 1 训练失败"
                return result
            result["gpt_model"] = gpt_model
            
            # 步骤 4: Stage 2 训练
            sovits_model = self.train_stage2_sovits(exp_dir, speaker_name, s2_epochs, batch_size)
            if sovits_model is None:
                result["error"] = "Stage 2 训练失败"
                return result
            result["sovits_model"] = sovits_model
            
            # 训练成功
            result["status"] = "completed"
            result["end_time"] = datetime.now().isoformat()
            result["duration"] = time.time() - start_time
            
            print(f"\n{'='*60}")
            print(f"🎉 训练完成！")
            print(f"   耗时: {result['duration']:.1f} 秒")
            print(f"   GPT 模型: {gpt_model}")
            print(f"   SoVITS 模型: {sovits_model}")
            print(f"{'='*60}\n")
            
        except Exception as e:
            result["error"] = str(e)
            print(f"\n❌ 训练失败: {e}")
            import traceback
            traceback.print_exc()
        
        return result


if __name__ == "__main__":
    # 测试代码
    trainer = GPTSoVITSTrainer()
    print("✅ 训练器初始化成功")

