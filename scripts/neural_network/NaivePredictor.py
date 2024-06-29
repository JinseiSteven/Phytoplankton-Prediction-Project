import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class NaivePredictor(nn.Module):
    def __init__(self, lstm_input_size, lstm_hidden_size, lstm_num_layers, num_abiotic_variables, dense_size):
        super(NaivePredictor, self).__init__()
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=lstm_input_size, 
                            hidden_size=lstm_hidden_size, 
                            num_layers=lstm_num_layers)
        
        # abiotic processing layers
        self.abiotic_linear1 = nn.Linear(num_abiotic_variables, 
                                         dense_size)
        
        # final combinatory layer
        self.prediction_layer = nn.Linear(lstm_hidden_size + dense_size, lstm_input_size)
        
    def forward(self, abiotic_variables, past_concentrations):
        # encoding the history
        lstm_out, _ = self.lstm(past_concentrations)
        
        # getting the last time step output from LSTM (lstm_hidden_size)
        lstm_last_output = lstm_out[-1, :]
        
        # processing the abiotic data
        abiotic_processed = torch.relu(self.abiotic_linear1(abiotic_variables))
        
        # concatenating the two outputs
        combined_features = torch.cat((lstm_last_output, abiotic_processed), dim=-1)
        
        # applying the final prediction
        predictions = self.prediction_layer(combined_features)
        
        return predictions

    def loss(self, input_data, expected_concentrations, metric="MSE"):

        abio_data, phyto_data = input_data

        # predicting phytoplankton concentrations
        predicted_concentrations = self.forward(abio_data, phyto_data)

        # calculating the loss based on the chosen metric
        if metric == "MSE":
            loss = F.mse_loss(predicted_concentrations, expected_concentrations)
        elif metric == "MAE":
            loss = F.l1_loss(predicted_concentrations, expected_concentrations)
        elif metric == "Huber":
            loss = F.smooth_l1_loss(predicted_concentrations, expected_concentrations)
        else:

            # just defaulting to MSE if a specified loss metric is not supported
            print(f'Loss metric {metric} was not found, please choose one from ["MSE", "MAE", "Huber", "CosSim"]')
            print('Defaulted to "MSE"')
            loss = F.mse_loss(predicted_concentrations, expected_concentrations)

        return loss




# Example usage
if __name__ == "__main__":

    lstm_input_size = 4  # Input size to LSTM (log concentration)
    lstm_hidden_size = 64  # LSTM hidden size
    lstm_num_layers = 1  # Number of LSTM layers
    num_abiotic_variables = 5  # Number of abiotic variables
    dense_size = 32  # Size of dense layers for abiotic variables
    num_phytoplankton_types = 4  # Number of phytoplankton types
    
    # Create model instance
    model = NaivePredictor(lstm_input_size, lstm_hidden_size, lstm_num_layers, num_abiotic_variables, dense_size, num_phytoplankton_types)
    
    # Example input tensors (batch_size = 1 for demonstration)
    sequence_length = 10  # Example sequence length
    batch_size = 1
    
    # Example input data (random for illustration)
    past_concentrations = torch.randn(batch_size, sequence_length, lstm_input_size)  # (batch_size, sequence_length, lstm_input_size)
    abiotic_variables = torch.randn(batch_size, num_abiotic_variables)  # (batch_size, num_abiotic_variables)
    
    print(past_concentrations.shape)
    print(abiotic_variables.shape)

    # Forward pass
    outputs = model(past_concentrations, abiotic_variables)
    
    # Print output shape (should be batch_size x num_phytoplankton_types)
    print("Output shape:", outputs.shape)
    
    # Print model summary
    print(model)
