import numpy as np


def create_array(video, video_trimmed):
    # Add every 6th frame to the new arrays
    for i in range(0, video.shape[2]-6, 6):
        video_trimmed[:, :, i // 6] = video[:, :, i]
    return video_trimmed


# Set session name
session = r"20230124_115956__CHS030_B4_APOP_control"

# Set video array names
video_1_name = r"CHS030_B4_2023-01-24-12-00-34_cam_1"
video_2_name = r"CHS030_B4_2023-01-24-12-00-34_cam_2"

# Set filepath
filepath = fr"Z:\Behavioural_data\EyeCam\Preprocessed\{session}"

# Load the arrays
video_1 = np.load(fr"{filepath}\{video_1_name}.npy")
video_2 = np.load(fr"{filepath}\{video_2_name}.npy")

# Initialise new arrays with 1/6th of the frames
video_trimmed_1 = np.zeros((video_1.shape[0], video_1.shape[1], video_1.shape[2] // 6), dtype=np.uint8)
video_trimmed_2 = np.zeros((video_2.shape[0], video_2.shape[1], video_2.shape[2] // 6), dtype=np.uint8)

# Add the frames to the new arrays
video_trimmed_1 = create_array(video_1, video_trimmed_1)
video_trimmed_2 = create_array(video_2, video_trimmed_2)

# Print the shapes of the new arrays
print(f"Shape video 1: {video_trimmed_1.shape}")
print(f"Shape video 2: {video_trimmed_2.shape}")

# Save the new arrays
np.save(fr"{filepath}\{video_1_name}_trimmed.npy", video_trimmed_1)
np.save(fr"{filepath}\{video_2_name}_trimmed.npy", video_trimmed_2)
    