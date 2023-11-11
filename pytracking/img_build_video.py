import cv2
import os

# Path to the directory containing frames
frames_dir = '.tracking_results/avitmp/bird-2/img'

# Get the list of frame filenames
frame_files = sorted(os.listdir(frames_dir))

# Define the output video path
output_video_path = '.tracking_results/avitmp/bird-2/bird2.mp4'

# Get the first frame to obtain video properties
first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
height, width, channels = first_frame.shape

# Define the video codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video_path, fourcc, 25.0, (width, height))

# Iterate over the frame files and write each frame to the video
for frame_file in frame_files:
    frame_path = os.path.join(frames_dir, frame_file)
    frame = cv2.imread(frame_path)
    video_writer.write(frame)

# Release the video writer
video_writer.release()

print("Video saved successfully.")
