import torch
from sklearn.metrics import r2_score


def test_model(model, test_input_data, test_output_data):
    # Remove gradient calculations
    with torch.no_grad():
        
        # Predict (Y_pred)
        prediction = model(test_input_data)

    # Detach
    prediction = prediction.cpu().detach().numpy()

    # Move test output data to CPU
    test_output_data = test_output_data.cpu()

    # Score (Percentage variance explained)
    percent_variance_explained = r2_score(y_true=test_output_data, y_pred=prediction)  
    
    return percent_variance_explained
