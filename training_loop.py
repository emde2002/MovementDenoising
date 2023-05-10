import numpy as np


def train_model(model, input_matrix, output_matrix, criterion, optimiser):
    # Clear Gradients
    optimiser.zero_grad()

    # Get Model Prediction
    prediction = model(input_matrix)

    # Get Loss
    loss = criterion(prediction, output_matrix)

    # Get Gradients
    loss.backward()

    # Update Weights
    optimiser.step()

    # Return Loss
    loss = loss.cpu().detach().numpy()

    return loss


def training_loop(model, video, spikes, criterion, optimiser, acceptable_loss):
    # Initialise epoch counter
    epoch = 0

    # Training loop
    while True:

        # Train Network
        loss = train_model(model, video, spikes, criterion, optimiser)

        # Print iteration and loss value
        if epoch % 50 == 0:
            print(f"Epoch: {str(epoch).zfill(4)} Loss: {np.round(loss, decimals=4)}")

        # Stop loop if converged
        if loss < acceptable_loss:
            print("Model converged")
            
            break

        # Stop loop if reached maximum number of iterations
        elif epoch == 1500:
            print("Model did not converge")
            
            break

        # Increase epoch and repeat loop
        else:
            epoch += 1

    return model
