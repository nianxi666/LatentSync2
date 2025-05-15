import os
os.environ["MPLBACKEND"] = "agg"  # 使用非交互式后端
import matplotlib.pyplot as plt
from pathlib import Path
from scripts.inference import main
from omegaconf import OmegaConf
import argparse
from datetime import datetime

CONFIG_PATH = Path("configs/unet/stage2.yaml")
CHECKPOINT_PATH = Path("checkpoints/latentsync_unet.pt")


def process_video(video_path, audio_path, guidance_scale, inference_steps, seed):
    # Create the temp directory if it doesn't exist
    output_dir = Path("./temp")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert paths to absolute Path objects and normalize them
    video_file_path = Path(video_path)
    video_path = video_file_path.absolute().as_posix()
    audio_path = Path(audio_path).absolute().as_posix()

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Set the output path for the processed video
    output_path = str(output_dir / f"{video_file_path.stem}_{current_time}.mp4")

    config = OmegaConf.load(CONFIG_PATH)

    config["run"].update(
        {
            "guidance_scale": guidance_scale,
            "inference_steps": inference_steps,
        }
    )

    # Parse the arguments
    args = create_args(video_path, audio_path, output_path, inference_steps, guidance_scale, seed)

    try:
        result = main(
            config=config,
            args=args,
        )
        print("Processing completed successfully.")
        return output_path  # Ensure the output path is returned
    except Exception as e:
        print(f"Error during processing: {str(e)}")


def create_args(video_path: str, audio_path: str, output_path: str, inference_steps: int, guidance_scale: float, seed: int) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_ckpt_path", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--video_out_path", type=str, required=True)
    parser.add_argument("--inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1247)

    return parser.parse_args(
        [
            "--inference_ckpt_path",
            CHECKPOINT_PATH.absolute().as_posix(),
            "--video_path",
            video_path,
            "--audio_path",
            audio_path,
            "--video_out_path",
            output_path,
            "--inference_steps",
            str(inference_steps),
            "--guidance_scale",
            str(guidance_scale),
            "--seed",
            str(seed),
        ]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video with audio.")
    parser.add_argument("--input_video", type=str, required=True, help="Input video file path.")
    parser.add_argument("--input_audio", type=str, required=True, help="Input audio file path.")
    parser.add_argument("--guidance_scale", type=float, default=2.0, help="Guidance scale.")
    parser.add_argument("--inference_steps", type=int, default=20, help="Number of inference steps.")
    parser.add_argument("--seed", type=int, default=1247, help="Random seed.")

    args = parser.parse_args()

    output_path = process_video(args.input_video, args.input_audio, args.guidance_scale, args.inference_steps, args.seed)
    print(f"Output video saved to: {output_path}")
