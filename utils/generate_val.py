import gym
from PIL import Image
import os

def main():
    n_levels = 1

    output_dir = "val_coinrun_frames"
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")

    for i in range(n_levels):
        env = gym.make("procgen:procgen-coinrun-v0", render_mode="rgb_array", start_level=(i + 100), num_levels=1, distribution_mode="hard",)
        observation = env.reset()
        
        frame_count = 0

        target_size = (128, 128)  # Resize to 128x128 pixels

        for _ in range(100):
            frame = env.render(mode="rgb_array")

            # Convert the frame to an image
            img = Image.fromarray(frame)
            img = img.resize(target_size, Image.Resampling.LANCZOS)

            img.save(f"{output_dir}/frame_{str(i)}_{frame_count:04d}.png")
            frame_count += 1

            action = env.action_space.sample()
            observation, reward, terminated, truncated = env.step(action)

            if terminated or truncated:
                observation = env.reset()

        env.close()
        print(f"Saved {frame_count} frames in the '{output_dir}' directory.")

if __name__ == "__main__":
    main()