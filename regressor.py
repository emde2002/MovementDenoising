import torch
import numpy as np


# Check if GPU available
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Device: {device}")


class Regressor(torch.nn.Module):

    def __init__(self, n_inputs, n_factors, n_outputs):
        # Constructor
        super(Regressor, self).__init__()

        # Initialise Weights
        self.n_inputs = n_inputs
        self.n_factors = n_factors
        self.n_outputs = n_outputs

        # Define layers
        self.input_encoding_weights = torch.nn.Parameter(torch.tensor(np.random.uniform(low=-1, high=1, size=(self.n_inputs, self.n_factors)), dtype=torch.float, device=device))
        self.hidden_layer = torch.nn.Parameter(torch.tensor(np.random.uniform(low=-1, high=1, size=(self.n_factors, self.n_factors)), dtype=torch.float, device=device))
        self.output_decoding_weights = torch.nn.Parameter(torch.tensor(np.random.uniform(low=-1, high=1, size=(self.n_factors, self.n_outputs)), dtype=torch.float, device=device))
        self.intercept = torch.nn.Parameter(torch.tensor(np.ones(self.n_outputs), dtype=torch.float, device=device))

    def forward(self, input_matrix):
        # Encode Input Into Factors
        encoded_input = torch.matmul(input_matrix, self.input_encoding_weights)

        # Pass through hidden layer
        hidden_output = torch.matmul(encoded_input, self.hidden_layer)

        # Decode Factors Into Output
        decoded_output = torch.matmul(hidden_output, self.output_decoding_weights)

        # Add Intercept
        output = decoded_output + self.intercept

        return output
