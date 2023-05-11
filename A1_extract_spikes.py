import numpy as np
import matplotlib.pyplot as plt
import os


# Set session name
filename = r"20230218_125525__CHS034_B3_APOP_control"

# Set filepath
filepath = fr"Z:\Data\Filipe\CHS\Processed\{filename}\suite2p"

# Load the iscell.npy file from the combined folder
indices = np.load(fr"{filepath}\combined\iscell.npy")

# Load the iscell.npy file from the plane 5 folder
remove_ROIs = np.load(fr"{filepath}\plane5\iscell.npy")

# Find the number of ROIs in plane 5
subtract = remove_ROIs.shape[0]

# Update the indices variable without the plane 5 ROIs
indices = indices[:-subtract, :]

# Create a new list of the indices with only the ROIs which are cells
list_indices = []
for i in range(0, len(indices)):
    if indices[i, 0] == 1:
        list_indices.append(i)
    else:
        continue

# Load the spks.npy (spikes of ROIs) and extract only the ROIs which are cells
spks = np.load(fr"{filepath}\combined\spks.npy")

# Cut the plane 5 neurons
spks = spks[:-subtract, :]

# Select only the cells
spikes = spks[list_indices, :]

# Check and create if necessary a folder
os.makedirs(fr"Z:\Behavioural_data\EyeCam\Preprocessed\{filename}", exist_ok=True)

# Save the array with only the cells
np.save(fr"Z:\Behavioural_data\EyeCam\Preprocessed\{filename}\spikes.npy", spikes)

"""
## To save the raw fluorescence of the neurons, but it is not necessary for the experiment
# Load the F.npy (raw fluorescence of ROIs) and extract only the ROIs which are cells
fluorescence = np.load(fr"{filepath}\combined\F.npy")

# Cut the plane 5 neurons
fluorescence = fluorescence[:-subtract, :]

# Select only the cells
neurons = fluorescence[list_indices, :]

# Save the array with only the cells
np.save(fr"{filepath}\neurons.npy", neurons)
"""

"""
## To visualise the data
# Plotting the results
# Set up the figure and axis objects
fig, ax = plt.subplots(figsize=(10, 5))

# Plot the neuron activity matrix
ax.imshow(neurons, cmap="viridis", aspect="auto")

# Add labels and title
ax.set_xlabel("Time")
ax.set_ylabel("Neuron Fluorescence")
ax.set_title("Neuron Activity")

# Show the plot
plt.show()
"""