#!/bin/bash
# 启动 GPT-SoVITS API 服务

echo "========================================="
echo "🚀 启动 GPT-SoVITS API 服务"
echo "========================================="
echo ""

# 检查虚拟环境
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  未检测到激活的虚拟环境"
    echo "💡 请先运行: source venv/bin/activate"
    exit 1
fi

echo "✅ 使用虚拟环境: $VIRTUAL_ENV"
echo ""

# 检查 GPT-SoVITS 目录
SOVITS_DIR="GPT-SoVITS-main"
if [ ! -d "$SOVITS_DIR" ]; then
    echo "❌ 未找到 GPT-SoVITS: $SOVITS_DIR"
    echo "💡 请先运行: ./install_sovits.sh"
    exit 1
fi

echo "✅ GPT-SoVITS 目录: $SOVITS_DIR"
echo ""

# 进入 GPT-SoVITS 目录
cd "$SOVITS_DIR" || exit 1

# 检查 api_v2.py
if [ ! -f "api_v2.py" ]; then
    echo "❌ 未找到 api_v2.py"
    exit 1
fi

# 启动 API 服务
echo "========================================="
echo "📡 API 服务地址: http://127.0.0.1:9880"
echo "💡 提示: 按 Ctrl+C 停止服务"
echo "========================================="
echo ""

python api_v2.py -a 127.0.0.1 -p 9880

