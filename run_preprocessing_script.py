#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-SoVITS 预处理脚本包装器
确保正确设置 Python 路径后再运行预处理脚本
"""
import sys
import os

def main():
    if len(sys.argv) < 3:
        print("Usage: python run_preprocessing_script.py <sovits_path> <script_path>")
        sys.exit(1)
    
    sovits_path = sys.argv[1]
    script_path = sys.argv[2]
    
    # 关键修复：将 GPT_SoVITS 目录添加到 sys.path
    # 因为 text 模块在 GPT_SoVITS/text/ 下
    gpt_sovits_module_path = os.path.join(sovits_path, "GPT_SoVITS")
    if gpt_sovits_module_path not in sys.path:
        sys.path.insert(0, gpt_sovits_module_path)
    
    # 也添加主目录，以防其他导入需要
    if sovits_path not in sys.path:
        sys.path.insert(0, sovits_path)
    
    # 切换工作目录到 GPT-SoVITS-main
    original_cwd = os.getcwd()
    os.chdir(sovits_path)
    
    try:
        # 读取并执行脚本
        with open(script_path, 'r', encoding='utf-8') as f:
            script_content = f.read()
        
        # 创建完整的全局命名空间（包含内置模块）
        script_globals = {
            '__name__': '__main__',
            '__file__': script_path,
            '__builtins__': __builtins__,  # 添加内置模块
        }
        
        # 执行脚本
        exec(script_content, script_globals)
        
    except Exception as e:
        print(f"Error executing script: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # 恢复原始工作目录
        os.chdir(original_cwd)

if __name__ == "__main__":
    main()

