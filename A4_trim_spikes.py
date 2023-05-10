import numpy as np
import h5py


def binarization(triggers):
    # Transform data into a binary array
    for i in range(triggers.shape[1]):
        if triggers[:, i] > 3500:
            triggers[:, i] = 1
        else:
            triggers[:, i] = 0
    return triggers


# Set session name
session = r"20230124_115956__CHS030_B4_APOP_control"

# Set filepath
filepath = fr"Z:\Behavioural_data\EyeCam\Preprocessed\{session}"

# Load the HDF5-format .mat file
with h5py.File(fr"{filepath}\spikes_triggers.mat", "r") as file:
    # Access the data within the file
    spikes_triggers = file["spikes_triggers"][:]

# Load the HDF5-format .mat file
with h5py.File(fr"{filepath}\video_triggers.mat", "r") as file:
    # Access the data within the file
    video_triggers = file["video_triggers"][:]

# Create binary arrays
spikes_triggers = binarization(spikes_triggers)
video_triggers = binarization(video_triggers)

# Count the number of spikes
diff_spikes = np.diff(spikes_triggers)
count_spikes = np.sum(diff_spikes == 1)
print(f"Number of spikes: {count_spikes}")

# Count the number of frames
diff_video = np.diff(video_triggers)
count_video = np.sum(diff_video == 1)
print(f"Number of frames: {count_video}")

# Find the first frame
first_frame = np.argmax(diff_video == 1)
print(f"First frame: {first_frame}")

# Find the location of the last completed spike before the first frame
diff_spikes_first_frame = diff_spikes[0, first_frame::-1]
last_spike_start = diff_spikes_first_frame.size - np.argmax(diff_spikes_first_frame == -1)
print(f"Last completed spike before first frame: {last_spike_start}")

# Calculate the number of fluorescent values to trim from start to the first frame
spikes_to_delete = np.sum(diff_spikes[0, :last_spike_start] == -1)
print(f"Trim first {spikes_to_delete // 6} fluorescent values")

# Find the location of the last completed frame
rev_diff_video = diff_video[0, ::-1]
last_frame = diff_video.shape[1] - np.argmax(rev_diff_video == -1) + 1
print(f"Last completed frame: {last_frame}")

# Find the location of the last completed spike right after the last frame
rev_diff_spikes = diff_spikes[0, last_frame::-1]
last_last_spikes = np.argmax(diff_spikes[0, last_frame:] == 1) + 1
print(f"Amount of completed spikes right after last frame: {last_last_spikes}")

# Load the spikes.npy file
spikes = np.load(fr"{filepath}\spikes.npy")
print(f"Original number of fluorescent values: {spikes.shape[1]}")

# Trim spikes.npy
spikes = spikes[:, (spikes_to_delete // 6):(spikes.shape[1] - last_last_spikes // 6)]
print(f"New number of fluorescent values: {spikes.shape[1]}")

# Trim excess spikes from the first triggers
spikes = spikes[:, (spikes.shape[1] - count_video // 6):]
print(f"Final number of fluorescent values: {spikes.shape[1]}")

# Save the trimmed spikes.npy file
np.save(fr"{filepath}\spikes_trimmed.npy", spikes)
