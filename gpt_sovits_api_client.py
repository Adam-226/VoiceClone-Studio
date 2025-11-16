#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-SoVITS API 客户端
用于调用 GPT-SoVITS API 服务进行语音生成
"""

import requests
import os
from typing import Optional, List, Dict
import time


class GPTSoVITSAPIClient:
    """GPT-SoVITS API 客户端"""
    
    def __init__(self, api_url: str = "http://127.0.0.1:9880"):
        """
        初始化 API 客户端
        
        Args:
            api_url: API 服务地址
        """
        self.api_url = api_url.rstrip("/")
        self.tts_endpoint = f"{self.api_url}/tts"
        self.set_gpt_weights_endpoint = f"{self.api_url}/set_gpt_weights"
        self.set_sovits_weights_endpoint = f"{self.api_url}/set_sovits_weights"
        self.control_endpoint = f"{self.api_url}/control"
        
        print(f"🌐 GPT-SoVITS API 客户端初始化")
        print(f"   API地址: {self.api_url}")
    
    def check_api_health(self) -> bool:
        """检查 API 服务是否可用"""
        try:
            response = requests.get(f"{self.api_url}/", timeout=2)
            return response.status_code in [200, 404]  # 404 也表示服务在运行
        except:
            return False
    
    def set_gpt_weights(self, weights_path: str) -> bool:
        """
        设置 GPT 模型权重
        
        Args:
            weights_path: 模型权重路径
            
        Returns:
            是否成功
        """
        try:
            print(f"   🔧 设置 GPT 模型: {weights_path}")
            response = requests.get(
                self.set_gpt_weights_endpoint,
                params={"weights_path": weights_path},
                timeout=30
            )
            if response.status_code == 200:
                print(f"   ✅ GPT 模型设置成功")
                return True
            else:
                print(f"   ❌ GPT 模型设置失败 (HTTP {response.status_code})")
                try:
                    error_info = response.json()
                    print(f"   错误信息: {error_info}")
                except:
                    print(f"   响应内容: {response.text}")
                return False
        except Exception as e:
            print(f"   ❌ 设置 GPT 权重失败: {e}")
            return False
    
    def set_sovits_weights(self, weights_path: str) -> bool:
        """
        设置 SoVITS 模型权重
        
        Args:
            weights_path: 模型权重路径
            
        Returns:
            是否成功
        """
        try:
            print(f"   🔧 设置 SoVITS 模型: {weights_path}")
            response = requests.get(
                self.set_sovits_weights_endpoint,
                params={"weights_path": weights_path},
                timeout=30
            )
            if response.status_code == 200:
                print(f"   ✅ SoVITS 模型设置成功")
                return True
            else:
                print(f"   ❌ SoVITS 模型设置失败 (HTTP {response.status_code})")
                try:
                    error_info = response.json()
                    print(f"   错误信息: {error_info}")
                except:
                    print(f"   响应内容: {response.text}")
                return False
        except Exception as e:
            print(f"   ❌ 设置 SoVITS 权重失败: {e}")
            return False
    
    def generate_speech(
        self,
        text: str,
        ref_audio_path: str,
        output_path: str,
        text_lang: str = "zh",
        prompt_text: str = "",
        prompt_lang: str = "zh",
        aux_ref_audio_paths: Optional[List[str]] = None,
        top_k: int = 5,
        top_p: float = 1.0,
        temperature: float = 1.0,
        text_split_method: str = "cut5",
        batch_size: int = 1,
        speed_factor: float = 1.0,
        streaming_mode: bool = False
    ) -> bool:
        """
        生成语音
        
        Args:
            text: 要合成的文本
            ref_audio_path: 参考音频路径
            output_path: 输出音频路径
            text_lang: 文本语言 (zh/en/ja/etc)
            prompt_text: 参考音频的文本
            prompt_lang: 参考音频的语言
            aux_ref_audio_paths: 辅助参考音频路径列表（多音频融合）
            top_k: top-k 采样
            top_p: top-p 采样
            temperature: 采样温度
            text_split_method: 文本分割方法
            batch_size: 批次大小
            speed_factor: 语速控制
            streaming_mode: 是否流式返回
            
        Returns:
            是否成功
        """
        try:
            # 构建请求数据
            payload = {
                "text": text,
                "text_lang": text_lang,
                "ref_audio_path": ref_audio_path,
                "prompt_text": prompt_text,
                "prompt_lang": prompt_lang,
                "top_k": top_k,
                "top_p": top_p,
                "temperature": temperature,
                "text_split_method": text_split_method,
                "batch_size": batch_size,
                "speed_factor": speed_factor,
                "streaming_mode": streaming_mode,
            }
            
            # 添加辅助参考音频
            if aux_ref_audio_paths:
                payload["aux_ref_audio_paths"] = aux_ref_audio_paths
            
            # 发送请求
            response = requests.post(
                self.tts_endpoint,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                # 保存音频
                with open(output_path, "wb") as f:
                    f.write(response.content)
                return True
            else:
                print(f"❌ API 返回错误: {response.status_code}")
                try:
                    error_info = response.json()
                    print(f"   错误信息: {error_info}")
                except:
                    print(f"   错误内容: {response.text[:200]}")
                return False
                
        except Exception as e:
            print(f"❌ 生成语音失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate_with_trained_model(
        self,
        text: str,
        gpt_model_path: str,
        sovits_model_path: str,
        ref_audio_path: str,
        output_path: str,
        text_lang: str = "zh",
        prompt_text: str = "",
        aux_ref_audio_paths: Optional[List[str]] = None,
        **kwargs
    ) -> bool:
        """
        使用训练好的模型生成语音
        
        Args:
            text: 要合成的文本
            gpt_model_path: GPT 模型路径
            sovits_model_path: SoVITS 模型路径
            ref_audio_path: 参考音频路径
            output_path: 输出音频路径
            text_lang: 文本语言
            prompt_text: 参考音频文本
            aux_ref_audio_paths: 辅助参考音频
            **kwargs: 其他生成参数
            
        Returns:
            是否成功
        """
        print(f"🎯 使用训练好的模型生成语音")
        print(f"   GPT 模型: {os.path.basename(gpt_model_path)}")
        print(f"   SoVITS 模型: {os.path.basename(sovits_model_path)}")
        print(f"   参考音频: {os.path.basename(ref_audio_path)}")
        
        # 检查文件是否存在
        if not os.path.exists(gpt_model_path):
            print(f"❌ GPT 模型文件不存在: {gpt_model_path}")
            return False
        if not os.path.exists(sovits_model_path):
            print(f"❌ SoVITS 模型文件不存在: {sovits_model_path}")
            return False
        if not os.path.exists(ref_audio_path):
            print(f"❌ 参考音频文件不存在: {ref_audio_path}")
            return False
        
        print(f"   ✅ 所有文件路径检查通过")
        
        # 设置模型权重
        if not self.set_gpt_weights(gpt_model_path):
            return False
        
        if not self.set_sovits_weights(sovits_model_path):
            return False
        
        # 等待模型加载
        time.sleep(2)
        
        # 生成语音
        return self.generate_speech(
            text=text,
            ref_audio_path=ref_audio_path,
            output_path=output_path,
            text_lang=text_lang,
            prompt_text=prompt_text,
            prompt_lang=text_lang,
            aux_ref_audio_paths=aux_ref_audio_paths,
            **kwargs
        )
    
    def restart_service(self) -> bool:
        """重启 API 服务"""
        try:
            response = requests.post(
                self.control_endpoint,
                json={"command": "restart"},
                timeout=5
            )
            return True
        except:
            return False


if __name__ == "__main__":
    # 测试代码
    client = GPTSoVITSAPIClient()
    
    if client.check_api_health():
        print("✅ API 服务运行正常")
    else:
        print("❌ API 服务不可用")
        print("   请启动 GPT-SoVITS API 服务:")
        print("   cd GPT-SoVITS-main && python api_v2.py -p 9880")

