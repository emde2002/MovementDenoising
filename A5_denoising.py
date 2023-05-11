import load_tensors
import training_loop
import test_model
from sklearn.model_selection import KFold
import regressor
import torch
import numpy as np


# Set session name
session = r"20230218_125525__CHS034_B3_APOP_control"

# Set spikes and video array names
spikes_name = r"spikes_trimmed"
video_1_name = r"CHS034_B3_2023-02-18-12-55-52_cam_1_trimmed"
video_2_name = r"CHS034_B3_2023-02-18-12-55-52_cam_2_trimmed"

# Set filepath
filepath = fr"Z:\Behavioural_data\EyeCam\Preprocessed\{session}"

# Load spikes and video tensors
spikes, video, n_neurons, n_pixels = load_tensors.load_data(spikes_name=spikes_name, video_1_name=video_1_name, video_2_name=video_2_name, filepath=filepath)

# Set factors
n_factors = 3

# Initialise percent variance explained list
percent_variance_explained_list = []

# Create 5 splits where each 4 chunks will be used for training and 1 chunk will be used for testing
kf = KFold(n_splits=5)

# 5 iterations of training and testing with the test split being different in each model
for i, (train_indices, test_indices) in enumerate(kf.split(video)):
    
    # Separate the data into training and testing
    train_spikes = spikes[train_indices]
    test_spikes = spikes[test_indices]
    train_video = video[train_indices]
    test_video = video[test_indices]

    # Move tensors to GPU if available
    if torch.cuda.is_available():
        train_spikes = train_spikes.cuda()
        train_video = train_video.cuda()
        test_spikes = test_spikes.cuda()
        test_video = test_video.cuda()

    # Load model
    model = regressor.Regressor(n_inputs=n_pixels, n_factors=n_factors, n_outputs=n_neurons)

    # Set Training Parameters
    criterion = torch.nn.MSELoss()
    learning_rate = 0.001
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    acceptable_loss = 0.001
    
    # Train model
    model = training_loop.training_loop(model=model, video=train_video, spikes=train_spikes, criterion=criterion, optimiser=optimiser, acceptable_loss=acceptable_loss)

    # Test model
    percent_variance_explained = test_model.test_model(model=model, test_input_data=test_video, test_output_data=test_spikes)
    
    # Save percent variance explained in list
    percent_variance_explained_list.append(percent_variance_explained)
    
    # Save model
    torch.save(model, fr"{filepath}\RRR_Model_Number_{i}.pt")

# Print percent variance explained and save as text file
print(f"Percent Variance Explained: {percent_variance_explained_list}")
np.savetxt(fr"{filepath}\percent_variance_explained_list.txt", percent_variance_explained_list)

# Initialise the parameter lists
input_encoding_weights_list = []
hidden_layer_list = []
output_decoding_weights_list = []
intercept_list = []

# Create lists of the parameters of each model
for i in range(5):
    
    # Load model
    loaded_model = torch.load(fr"{filepath}\RRR_Model_Number_{i}.pt")
    loaded_model.eval()
    
    # Get parameters of model
    input_encoding_weights = loaded_model.input_encoding_weights.cpu().detach().numpy()
    hidden_layer = loaded_model.hidden_layer.cpu().detach().numpy()
    output_decoding_weights = loaded_model.output_decoding_weights.cpu().detach().numpy()
    intercept = loaded_model.intercept.cpu().detach().numpy()

    # Append parameters to lists
    input_encoding_weights_list.append(input_encoding_weights)
    hidden_layer_list.append(hidden_layer)
    output_decoding_weights_list.append(output_decoding_weights)
    intercept_list.append(intercept)

# Calculate mean of each parameter
mean_input_encoding_weights = np.mean(input_encoding_weights_list, axis=0)
mean_hidden_layer = np.mean(hidden_layer_list, axis=0)
mean_output_decoding_weights = np.mean(output_decoding_weights_list, axis=0)
mean_intercept = np.mean(intercept_list, axis=0)

# Create super model with the mean parameters
super_model = regressor.Regressor(n_inputs=n_pixels, n_factors=n_factors, n_outputs=n_neurons)

# Update parameters with the mean parameters
mean_input_encoding_weights = torch.from_numpy(mean_input_encoding_weights)
super_model.input_encoding_weights = torch.nn.Parameter(mean_input_encoding_weights)
mean_hidden_layer = torch.from_numpy(mean_hidden_layer)
super_model.hidden_layer = torch.nn.Parameter(mean_hidden_layer)
mean_output_decoding_weights = torch.from_numpy(mean_output_decoding_weights)
super_model.output_decoding_weights = torch.nn.Parameter(mean_output_decoding_weights)
mean_intercept = torch.from_numpy(mean_intercept)
super_model.intercept = torch.nn.Parameter(mean_intercept)

# Calculate percent variance explained of super model
mean_percent_variance_explained_list = sum(percent_variance_explained_list) / len(percent_variance_explained_list)
mean_percent_variance_explained_list = np.array([mean_percent_variance_explained_list])

# Print percent variance explained and save as text file
print(f"Percent Variance Explained of super model: {mean_percent_variance_explained_list}")
np.savetxt(fr"{filepath}\percent_variance_explained_Super_Model.txt", mean_percent_variance_explained_list)

# Save super model
torch.save(super_model, fr"{filepath}\Super_Model.pt")

# Predict spikes using the super model
with torch.no_grad():
        
    # Predict (Y_pred)
    prediction = super_model(video)
    
# Detach prediction
prediction = prediction.detach().numpy()    

# Subtract prediction from spikes
predicted_spikes = np.subtract(spikes, prediction)

# Save
np.save(fr"{filepath}\Predicted_spikes.npy", predicted_spikes)
