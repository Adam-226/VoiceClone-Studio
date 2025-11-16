"""
音频生成API服务 - 基于 GPT-SoVITS
支持上传多个音频语料，训练定制化模型，生成高质量语音
"""
import os
import uuid
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import shutil

# 配置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 使用 GPT-SoVITS 音频生成器
from audio_generator_sovits import AudioGeneratorSoVITS as AudioGenerator
print("🚀 使用 GPT-SoVITS 引擎（支持多样本训练和高质量语音生成）")

# 创建必要的目录
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("static", exist_ok=True)

app = FastAPI(
    title="AI音频生成系统 - GPT-SoVITS版",
    description="支持从多个音频样本训练定制化声音模型",
    version="2.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

# 初始化音频生成器
audio_gen = AudioGenerator()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """返回主页面"""
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/upload_audio")
async def upload_audio(
    files: list[UploadFile] = File(...),
    speaker_name: str = Form(...)
):
    """
    批量上传音频语料（支持一次上传多个文件）
    
    Args:
        files: 音频文件列表（wav, mp3等格式）
        speaker_name: 说话者名称
    
    GPT-SoVITS 模式：
        - ✅ 支持批量上传多个音频文件
        - 上传后需要训练模型才能使用
        - 推荐上传 5-20 个音频样本
    """
    if not files:
        raise HTTPException(status_code=400, detail="请至少上传一个音频文件")
    
    uploaded_files = []
    failed_files = []
    
    try:
        for file in files:
            try:
                # 生成唯一的文件名
                file_extension = os.path.splitext(file.filename)[1]
                unique_filename = f"{speaker_name}_{uuid.uuid4().hex[:8]}{file_extension}"
                file_path = os.path.join("uploads", unique_filename)
                
                # 保存上传的文件
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                
                # 处理音频并提取特征
                result = audio_gen.process_reference_audio(file_path, speaker_name)
                
                uploaded_files.append({
                    "filename": file.filename,
                    "saved_path": file_path,
                    "duration": result.get("duration", 0)
                })
                
            except Exception as e:
                failed_files.append({
                    "filename": file.filename,
                    "error": str(e)
                })
        
        # 获取最终的说话者信息
        speakers = audio_gen.list_speakers()
        speaker_info = next((s for s in speakers if s["name"] == speaker_name), None)
        
        total_uploaded = len(uploaded_files)
        total_failed = len(failed_files)
        total_audio_count = speaker_info["audio_count"] if speaker_info else total_uploaded
        
        if total_uploaded == 0:
            raise HTTPException(
                status_code=500, 
                detail=f"所有文件上传失败: {failed_files}"
            )
        
        message = f"成功上传 {total_uploaded} 个音频文件！"
        if total_failed > 0:
            message += f" ({total_failed} 个失败)"
        message += f" 说话者 '{speaker_name}' 现有 {total_audio_count} 个音频样本"
        
        return {
            "status": "success",
            "message": message,
            "speaker_name": speaker_name,
            "uploaded_count": total_uploaded,
            "failed_count": total_failed,
            "total_audio_count": total_audio_count,
            "trained": speaker_info["trained"] if speaker_info else False,
            "uploaded_files": uploaded_files,
            "failed_files": failed_files,
            "recommendation": f"已上传 {total_audio_count} 个样本，推荐上传 10-15 个音频样本后进行训练"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量上传失败: {str(e)}")

@app.post("/generate_audio")
async def generate_audio(
    text: str = Form(...),
    speaker_name: str = Form(...),
    language: str = Form(default="zh-cn")
):
    """
    根据文字提示生成音频
    
    Args:
        text: 要转换成语音的文字
        speaker_name: 使用的说话者名称
        language: 语言代码（zh-cn为中文）
    """
    try:
        # 生成音频
        output_path = audio_gen.generate_speech(text, speaker_name, language)
        
        if not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail="音频生成失败")
        
        return {
            "status": "success",
            "message": "音频生成成功",
            "audio_url": f"/download_audio/{os.path.basename(output_path)}",
            "text": text,
            "speaker": speaker_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"音频生成失败: {str(e)}")

@app.get("/download_audio/{filename}")
async def download_audio(filename: str):
    """下载生成的音频文件"""
    file_path = os.path.join("outputs", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(file_path, media_type="audio/wav", filename=filename)

@app.get("/list_speakers")
async def list_speakers():
    """列出所有已学习的说话者"""
    speakers = audio_gen.list_speakers()
    return {
        "status": "success",
        "speakers": speakers,
        "count": len(speakers)
    }

@app.delete("/delete_speaker/{speaker_name}")
async def delete_speaker(speaker_name: str):
    """删除指定的说话者数据"""
    try:
        audio_gen.delete_speaker(speaker_name)
        return {
            "status": "success",
            "message": f"已删除说话者 '{speaker_name}' 的数据"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")

# ============== 新增：GPT-SoVITS 训练相关 API ==============

@app.post("/train_speaker")
async def train_speaker(
    background_tasks: BackgroundTasks,
    speaker_name: str = Form(...),
    epochs: int = Form(default=8),
    batch_size: int = Form(default=4)
):
    """
    训练说话者模型（GPT-SoVITS）
    
    Args:
        speaker_name: 说话者名称
        epochs: 训练轮数（默认8，约5-10分钟）
        batch_size: 批次大小（默认4）
    
    注意：
        - 至少需要 3 个音频样本（推荐 5-20 个）
        - 训练时间：5-15 分钟（取决于样本数量和 GPU）
        - 训练过程在后台进行，可以通过 /training_status 查询进度
    """
    try:
        # 检查是否有 train_speaker 方法（GPT-SoVITS 模式）
        if not hasattr(audio_gen, 'train_speaker'):
            raise HTTPException(
                status_code=400,
                detail="当前模式不支持训练功能。音频生成器未正确初始化"
            )
        
        # 获取说话者信息
        speakers = audio_gen.list_speakers()
        speaker_info = next((s for s in speakers if s["name"] == speaker_name), None)
        
        if not speaker_info:
            raise HTTPException(status_code=404, detail=f"未找到说话者 '{speaker_name}'")
        
        if speaker_info["audio_count"] < 3:
            raise HTTPException(
                status_code=400,
                detail=f"训练需要至少3个音频样本，当前只有 {speaker_info['audio_count']} 个"
            )
        
        # 在后台开始训练
        def train_task():
            try:
                audio_gen.train_speaker(speaker_name, epochs, batch_size)
            except Exception as e:
                print(f"❌ 训练失败: {str(e)}")
        
        background_tasks.add_task(train_task)
        
        return {
            "status": "success",
            "message": f"开始训练说话者 '{speaker_name}' 的模型",
            "speaker_name": speaker_name,
            "audio_count": speaker_info["audio_count"],
            "epochs": epochs,
            "estimated_time": f"{epochs * 1-2} 分钟",
            "note": "训练在后台进行，请通过 /training_status/{speaker_name} 查询进度"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"启动训练失败: {str(e)}")

@app.get("/training_status/{speaker_name}")
async def get_training_status(speaker_name: str):
    """
    查询训练状态
    
    Returns:
        status: training, completed, failed, not_started
        progress: 0-100
        其他训练信息
    """
    try:
        if not hasattr(audio_gen, 'get_training_status'):
            return {"status": "not_supported", "message": "当前模式不支持训练"}
        
        status = audio_gen.get_training_status(speaker_name)
        return {
            "status": "success",
            "speaker_name": speaker_name,
            "training_status": status
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")

@app.get("/system_info")
async def get_system_info():
    """
    获取系统信息
    
    Returns:
        使用的引擎、设备、支持的功能等
    """
    import torch
    
    info = {
        "engine": "GPT-SoVITS",
        "version": "2.0.0",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cuda_available": torch.cuda.is_available(),
        "supports_training": hasattr(audio_gen, 'train_speaker'),
        "supports_multi_sample": True,
    }
    
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
    
    return {
        "status": "success",
        "system_info": info
    }

# ============== API 文档 ==============

@app.get("/api/docs_info")
async def api_docs_info():
    """返回 API 使用说明"""
    return {
        "title": "AI音频生成系统 - GPT-SoVITS版",
        "version": "2.0.0",
        "description": "支持从多个音频样本训练定制化声音模型",
        "workflow": {
            "step1": "上传音频样本（/upload_audio）- 推荐 5-20 个",
            "step2": "训练模型（/train_speaker）- 约 5-15 分钟",
            "step3": "生成语音（/generate_audio）- 使用训练好的模型",
        },
        "endpoints": {
            "POST /upload_audio": "上传音频样本",
            "POST /train_speaker": "训练说话者模型",
            "GET /training_status/{name}": "查询训练状态",
            "POST /generate_audio": "生成语音",
            "GET /list_speakers": "列出所有说话者",
            "DELETE /delete_speaker/{name}": "删除说话者",
            "GET /system_info": "获取系统信息"
        },
        "docs_url": "/docs",
        "redoc_url": "/redoc"
    }

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("🎵 AI音频生成系统 - GPT-SoVITS版 v2.0")
    print("=" * 60)
    print("🚀 引擎: GPT-SoVITS（多样本训练）")
    print(f"🔧 设备: {'CUDA (GPU)' if audio_gen.device == 'cuda' else 'CPU'}")
    print("=" * 60)
    print("📍 访问地址: http://localhost:8000")
    print("📚 API文档: http://localhost:8000/docs")
    print("📖 使用说明: http://localhost:8000/api/docs_info")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)

