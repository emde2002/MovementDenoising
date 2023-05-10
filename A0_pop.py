# Load the tensors
# 5(or3)K fold, cross validation
# Basically split 5 chunks of data, train on 4, test on 1, repeat 5 times (each time on a differnt chunk)

# Then take the mean of the errors of each iteration
# So you run A6 5 times and each time create a different model

# Thne take  the mean of all the parameters across each model to create a super model
# Even if the model overfits it is okay since we are looking at the stuff which remain
# Then I substract the prediction from the super model to get the real stuff

# sickit learn KFold import, 

# Loss is substractino of the prredicted minus real and then take the square of the values and then get the mean (5% eroor difference)
# A Rsquare of 0.3 is very good

# Use roster map to visualise neurons afterwards

# Clarify that is not motion correction but it is motor activity denoising (artifact insinuates that there is something present which is not true
# 
# #reduced rank multivariant regression)

# How good the model is (30% is very good)
#Explain Rsqueare score
"""  
Matrix Multiplication Rules 
(A x S) Dot (S x B) = (A x B) 

Variables 
n_timepoints 
n_pixels 100 
n_neurons 
n_factors 

Input Matrix = (N_Timepoints x N_Pixels) 
Pixel Encoding Weights = (N_Pixels x N_Factors) 
Latent Factors = Input_Matrix x Encoding Matrix = (N_Timepoints x N_Factors) 
Neuron Decoding Weights = (N_Factors x N_Neurons) 
Output Matrix = Latent Factors X Neuron_Decoding_Weights = (N_Timepoints x N_Neurons) 

"""