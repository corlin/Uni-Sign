import os
import subprocess
import argparse

def download_checkpoint(output_directory, checkpoint_name):
    os.makedirs(output_directory, exist_ok=True)
    
    base_url = "https://huggingface.co/ZechengLi19/Uni-Sign/resolve/main"
    
    available_checkpoints = {
        "csl_stage1": "csl_stage1_weight.pth",
        "csl_stage2": "csl_stage2_weight.pth",
        "csl_daily_pose": "csl_daily_pose_only_slt.pth",
        "csl_daily_rgb_pose": "csl_daily_rgb_pose_slt.pth",
        "openasl_pose": "openasl_pose_only_slt.pth",
        "wlasl_pose": "wlasl_pose_only_islr.pth",
        "wlasl_rgb_pose": "wlasl_rgb_pose_islr.pth",
        "how2sign_pose": "how2sign_pose_only_slt.pth"
    }
    
    if checkpoint_name not in available_checkpoints:
        print(f"Error: Checkpoint '{checkpoint_name}' not found.")
        print("Available checkpoints:", ", ".join(available_checkpoints.keys()))
        return

    filename = available_checkpoints[checkpoint_name]
    url = f"{base_url}/{filename}"
    file_path = os.path.join(output_directory, filename)
    
    if os.path.exists(file_path):
        print(f"File {file_path} already exists. Skipping download.")
        return

    print(f"Downloading {filename} from Hugging Face...")
    command = ["wget", "-O", file_path, url]
    try:
        subprocess.run(command, check=True)
        print(f"Successfully downloaded {filename} to {file_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to download {filename}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download Uni-Sign checkpoints.')
    parser.add_argument('--output_directory', type=str, help='Path to save the checkpoints', default="./pretrained_weight")
    parser.add_argument('--checkpoint', type=str, help='Name of the checkpoint to download', choices=[
        "csl_stage1", "csl_stage2", "csl_daily_pose", "csl_daily_rgb_pose", 
        "openasl_pose", "wlasl_pose", "wlasl_rgb_pose", "how2sign_pose"
    ], default="csl_stage2")
    
    args = parser.parse_args()
    download_checkpoint(args.output_directory, args.checkpoint)
