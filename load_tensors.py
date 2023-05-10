import numpy as np
import torch


def load_data(spikes_name, video_1_name, video_2_name, filepath):
    # Load trimmed spikes
    spikes = np.load(fr"{filepath}\{spikes_name}.npy")
    print(f"Spikes shape: {spikes.shape}")

    # Load trimmed videos
    video_1 = np.load(fr"{filepath}\{video_1_name}.npy")
    video_2 = np.load(fr"{filepath}\{video_2_name}.npy")
    print(f"Video 1 shape: {video_1.shape}")
    print(f"Video 2 shape: {video_2.shape}")

    # Convert arrays to tensors
    spikes = torch.from_numpy(spikes)
    video_1 = torch.from_numpy(video_1)
    video_2 = torch.from_numpy(video_2)

    # Reshape videos to 2D tensors
    video_1 = video_1.view(video_1.shape[0] * video_1.shape[1], video_1.shape[2])
    video_2 = video_2.view(video_2.shape[0] * video_2.shape[1], video_2.shape[2])

    # Switch rows and columns of tensors
    spikes = spikes.t()
    video_1 = video_1.t()
    video_2 = video_2.t()

    # Concatenate the video tensors
    video = torch.cat((video_1, video_2), dim=1)
    print(f"Video shape: {video.shape}")
    print(f"Spikes shape: {spikes.shape}")

    # Change data type to float for normalisation
    #video = video.to(torch.float32)

    # Normalise data
    #spikes = torch.nn.functional.normalize(spikes, dim=1)
    #video = torch.nn.functional.normalize(video, dim=1)

    # Save tensors
    # torch.save(spikes, os.path.join(filepath, "Spikes.pt"))
    # torch.save(video, os.path.join(filepath, "Video.pt"))

    # Get data shapes
    n_neurons = spikes.size(dim=1)
    n_pixels = video.size(dim=1)

    return spikes, video, n_neurons, n_pixels
