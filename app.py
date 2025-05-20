import inferless
from pydantic import BaseModel, Field
from typing import Optional
from pathlib import Path
from datetime import datetime
from omegaconf import OmegaConf
import argparse

# 从 LatentSync 仓库中导入主推理函数
from scripts.inference import main as latent_sync_main

# 配置文件和模型权重路径（在仓库根目录）
CONFIG_PATH = Path("configs/unet/stage2.yaml")
CHECKPOINT_PATH = Path("checkpoints/latentsync_unet.pt")

# 1. 定义输入 schema
@inferless.request
class RequestObjects(BaseModel):
    video_path: str = Field(..., description="客户端上传的视频本地路径 or URL")
    audio_path: str = Field(..., description="客户端上传的音频本地路径 or URL")
    guidance_scale: float = Field(default=2.0, ge=1.0, le=3.0)
    inference_steps: int = Field(default=20, ge=1)
    seed: Optional[int] = Field(default=1247)

# 2. 定义输出 schema
@inferless.response
class ResponseObjects(BaseModel):
    output_video_path: str = Field(..., description="生成后的视频在服务器上的路径或下载 URL")

# 3. 创建 Inferless 服务实例，指定 GPU 型号（A100/T4 等）
app = inferless.Cls(gpu="A100")

# 4. 在容器启动时加载模型和配置
@app.load
def initialize():
    global config
    # 加载模型配置
    config = OmegaConf.load(CONFIG_PATH)
    # 如有必要，可在此处做一次模型预热或缓存

# 5. 处理每次推理请求
@app.infer
def infer(request: RequestObjects) -> ResponseObjects:
    # 更新推理参数到 config
    config.run.guidance_scale = request.guidance_scale
    config.run.inference_steps = request.inference_steps

    # 构造输出目录
    output_dir = Path("/mnt/data/temp")  # Inferless 容器内可写目录
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    basename = Path(request.video_path).stem
    output_path = str(output_dir / f"{basename}_{timestamp}.mp4")

    # 构造 argparse.Namespace，复用 LatentSync 脚本接口
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_ckpt_path", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--video_out_path", type=str, required=True)
    parser.add_argument("--inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1247)

    args = parser.parse_args([
        "--inference_ckpt_path", str(CHECKPOINT_PATH),
        "--video_path", request.video_path,
        "--audio_path", request.audio_path,
        "--video_out_path", output_path,
        "--inference_steps", str(request.inference_steps),
        "--guidance_scale", str(request.guidance_scale),
        "--seed", str(request.seed or 1247),
    ])

    # 调用主推理函数
    latent_sync_main(config=config, args=args)

    # 返回结果
    return ResponseObjects(output_video_path=output_path)

# 6. 为本地调试或 remote-run 提供入口
@inferless.local_entry_point
def entry_point(dynamic_params):
    req = RequestObjects(**dynamic_params)
    return infer(req)
