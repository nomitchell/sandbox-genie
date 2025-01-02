import gym
from PIL import Image
import os

def main():
    n_levels = 10

    output_dir = "coinrun_frames"
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        try:
            if os.path.isfile(file_path):  # Only delete files
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")

    for i in range(n_levels):
        # Create the environment
        env = gym.make("procgen:procgen-coinrun-v0", render_mode="rgb_array", start_level=i, num_levels=1)
        observation = env.reset()
        
        frame_count = 0

        target_size = (128, 128)  # Resize to 128x128 pixels

        for _ in range(1008):
            # Capture the current frame
            frame = env.render(mode="rgb_array")  # Render the frame as a NumPy array

            # Convert the frame to an image
            img = Image.fromarray(frame)
            img = img.resize(target_size, Image.Resampling.LANCZOS)

            # Save the frame as an image
            img.save(f"{output_dir}/frame_{str(i)}_{frame_count:04d}.png")
            frame_count += 1

            # Take a random action
            action = env.action_space.sample()
            observation, reward, terminated, truncated = env.step(action)

            # Reset if the episode is over
            if terminated or truncated:
                observation = env.reset()

        # Close the environment
        env.close()
        print(f"Saved {frame_count} frames in the '{output_dir}' directory.")

if __name__ == "__main__":
    main()
