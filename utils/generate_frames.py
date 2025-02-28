"""
Frame generator for CoinRun environments.
Generates and saves frames from different procedurally generated CoinRun levels.
"""
import os
import logging
import gym
from PIL import Image
import torch

NUM_LEVELS = 10
MAX_FRAMES_PER_LEVEL = 1000
TARGET_SIZE = (128, 128)
OUTPUT_DIR = "coinrun_frames"


def setup_output_directory(dir_path, clean=True):
    os.makedirs(dir_path, exist_ok=True)
    
    if clean:
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                logging.error(f"Error deleting file {file_path}: {e}")


def generate_level_frames(level_id, output_dir, max_frames=1000, target_size=(128, 128)):
    env = gym.make(
        "procgen:procgen-coinrun-v0", 
        render_mode="rgb_array",
        distribution_mode="hard",
        start_level=level_id,
        num_levels=1
    )
    
    observation = env.reset()
    frame_count = 0
    prev_action = None
    
    try:
        for _ in range(max_frames):
            while True:
                action = env.action_space.sample()
                if action != prev_action:
                    break
            prev_action = action
            
            frame = env.render(mode="rgb_array")
            
            img = Image.fromarray(frame)
            img = img.resize(target_size, Image.Resampling.LANCZOS) # using 128x128
            
            img.save(f"{output_dir}/frame_{level_id}_{frame_count:04d}.png")
            frame_count += 1
            
            observation, reward, terminated, truncated = env.step(action)
            
            if terminated:
                break
    finally:
        env.close()
    
    return frame_count


def main():
    """Main function to generate frames from multiple CoinRun levels."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    setup_output_directory(OUTPUT_DIR)
    logging.info(f"Generating frames for {NUM_LEVELS} levels")
    
    total_frames = 0
    for i in range(NUM_LEVELS):
        frames = generate_level_frames(
            i, OUTPUT_DIR, MAX_FRAMES_PER_LEVEL, TARGET_SIZE
        )
        total_frames += frames
        logging.info(f"Generated {frames} frames for level {i}")
    
    logging.info(f"Completed generating {total_frames} frames across {NUM_LEVELS} levels")


if __name__ == "__main__":
    main()