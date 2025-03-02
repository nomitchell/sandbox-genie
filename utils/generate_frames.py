import os
import logging
import gym
from PIL import Image
import torch
import multiprocessing as mp
from functools import partial
import cv2
import numpy as np

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


def generate_level_frames(level_id, output_dir, max_frames=1000, target_size=(128, 128), batch_size=100):
    env = gym.make(
        "procgen:procgen-coinrun-v0", 
        render_mode="rgb_array",
        distribution_mode="hard",
        start_level=level_id,
        num_levels=1
    )
    
    observation = env.reset()
    frame_count = 0
    frames_buffer = []
    
    try:
        for _ in range(max_frames):
            action = env.action_space.sample()
            
            frame = env.render(mode="rgb_array")
            
            frame_resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_LANCZOS4)
            
            frames_buffer.append((frame_resized, f"{output_dir}/frame_{level_id}_{frame_count:04d}.png"))
            frame_count += 1
            
            if len(frames_buffer) >= batch_size:
                for frame, path in frames_buffer:
                    cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                frames_buffer = []
            
            observation, reward, terminated, truncated = env.step(action)
            
            if terminated:
                break
                
        for frame, path in frames_buffer:
            cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
    finally:
        env.close()
    
    return frame_count


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    setup_output_directory(OUTPUT_DIR)
    logging.info(f"Generating frames for {NUM_LEVELS} levels")
    
    with mp.Pool(processes=min(mp.cpu_count(), NUM_LEVELS)) as pool:
        partial_generate = partial(
            generate_level_frames, 
            output_dir=OUTPUT_DIR, 
            max_frames=MAX_FRAMES_PER_LEVEL, 
            target_size=TARGET_SIZE
        )
        results = pool.map(partial_generate, range(NUM_LEVELS))
    
    total_frames = sum(results)
    logging.info(f"Completed generating {total_frames} frames across {NUM_LEVELS} levels")


if __name__ == "__main__":
    main()