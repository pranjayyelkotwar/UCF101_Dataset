import os
from moviepy.editor import VideoFileClip
from tqdm import tqdm

# Paths
input_dir = "videos"  # Input folder containing 101 directories
output_video_dir = "converted_videos"
output_audio_dir = "extracted_audio"

# Create output directories if not exist
os.makedirs(output_video_dir, exist_ok=True)
os.makedirs(output_audio_dir, exist_ok=True)

# Function to process videos
def process_videos(input_dir, output_video_dir, output_audio_dir):
    # Collect all .avi files
    video_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".avi"):
                video_files.append((root, file))
    
    # Progress bar
    for root, file in tqdm(video_files, desc="Processing videos", unit="file"):
        input_path = os.path.join(root, file)
        
        # Create output subdirectory structure
        relative_path = os.path.relpath(root, input_dir)
        video_output_path = os.path.join(output_video_dir, relative_path)
        audio_output_path = os.path.join(output_audio_dir, relative_path)
        
        os.makedirs(video_output_path, exist_ok=True)
        os.makedirs(audio_output_path, exist_ok=True)

        # Output file paths
        base_name = os.path.splitext(file)[0]
        mp4_path = os.path.join(video_output_path, f"{base_name}.mp4")
        wav_path = os.path.join(audio_output_path, f"{base_name}.wav")

        try:
            # Convert to MP4 and extract audio
            clip = VideoFileClip(input_path)
            clip.write_videofile(mp4_path, codec="libx264", audio_codec="aac")
            clip.audio.write_audiofile(wav_path)
            clip.close()
        except Exception as e:
            print(f"Failed to process {input_path}: {e}")

# Process the videos
process_videos(input_dir, output_video_dir, output_audio_dir)