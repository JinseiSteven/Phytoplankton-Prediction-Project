import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.functional as F
from tqdm.auto import tqdm


class PhytoPredictor(nn.Module):
    """
    Neural network model for predicting phytoplankton concentrations using abiotic data and phytoplankton history.

    Parameters:
    input_size_phyto (int): Input size for phytoplankton data.
    lstm_hidden_size (int): Hidden size of the LSTM layer for phytoplankton data.
    input_size_abio (int): Input size for abiotic data.
    ffnn_hidden_size (int): Hidden size of the feedforward neural network (FFNN) layer.
    output_size (int): Output size, number of classes or predicted values.
    p_drop (float): Dropout probability for regularization.
    bidirectional (bool, optional): Whether the LSTM is bidirectional. Defaults to True.

    Methods:
    forward(abio_data, phyto_input):
        Performs a forward pass through the model to predict phytoplankton concentrations.
    _predict_logits(abio_input, phyto_input):
        Predicts logits (raw output) before activation using the model's components.

    Attributes:
    history_encoder (nn.LSTM): LSTM layer for encoding phytoplankton history.
    FFNN (nn.Sequential): Feedforward neural network for final prediction.
    bidirectional (bool): Indicates if the LSTM is bidirectional.

    Example:
    >>> model = PhytoPredictor(input_size_phyto=10, lstm_hidden_size=64, input_size_abio=5, ffnn_hidden_size=32, output_size=1, p_drop=0.2)
    >>> abio_data = torch.randn(1, 5)  # Example abiotic data tensor
    >>> phyto_data = torch.randn(1, 1, 10)  # Example phytoplankton data tensor
    >>> predictions = model(abio_data, phyto_data)
    >>> predictions.shape
    torch.Size([1, 1])  # Example output shape, depends on output_size
    """

    def __init__(self, input_size_phyto, lstm_hidden_size, input_size_abio, ffnn_hidden_size, output_size, p_drop, bidirectional=True):
        super(PhytoPredictor, self).__init__()

        self.bidirectional = bidirectional

        # setting up the (Bi)LSTM for the phytoplankton data, input size will probably be the same as the hidden size
        self.history_encoder = nn.LSTM(
            input_size=input_size_phyto,
            hidden_size=lstm_hidden_size,
            bidirectional=bidirectional,
        )
        
        # the (Bi)LSTM will have concatenated the forward and backward pass parameters
        if bidirectional:
            ffnn_input_size = 2 * lstm_hidden_size + input_size_abio
        else:
            ffnn_input_size = lstm_hidden_size + input_size_abio

        # lastly we use a two layer FFNN to predict the concentrations
        self.FFNN = nn.Sequential(
            nn.Dropout(p_drop),
            nn.Linear(ffnn_input_size, ffnn_hidden_size),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(ffnn_hidden_size, output_size)
        )

    def forward(self, abio_data, phyto_input):
        """
        Forward pass of the model to predict phytoplankton concentrations.

        Parameters:
        abio_data (torch.Tensor): Abiotic data tensor of shape (batch_size, input_size_abio).
        phyto_input (torch.Tensor): Phytoplankton history matrix tensor of shape (batch_size, seq_len, input_size_phyto).

        Returns:
        torch.Tensor: Predicted phytoplankton concentrations tensor of shape (batch_size, output_size).
        """

        # first we predict using the neural network to get the logits
        logits = self._predict_logits(abio_data, phyto_input)

        # after acquiring the logits from the model, we use a ReLU activation function to hopefully get results
        output = F.ReLu(logits)

        return output
    
    def loss(self, input_data, expected_concentrations):
        """
        TODO TODO TODO TODO TODO TODO
        """

        abio_data, phyto_data = input_data

        raise(NotImplementedError)


    def _predict_logits(self, abio_input, phyto_input):
        """
        Predicts logits (raw output) before activation using the model's components.

        Parameters:
        abio_input (torch.Tensor): Abiotic data tensor of shape (batch_size, input_size_abio).
        phyto_input (torch.Tensor): Phytoplankton history matrix tensor of shape (batch_size, seq_len, input_size_phyto).

        Returns:
        torch.Tensor: Logits tensor before activation of shape (batch_size, output_size).
        """
        
        # pretty sure we should be using hx, but we can use the others if I am wrong
        u, (hx, cx) = self.history_encoder(phyto_input)
        
        # size is [1, 2 * clusters] if bidirectional else [1, clusters]
        encoded_history = hx.reshape(-1)

        # we concatenate the (Bi)LSTM output with the abio input
        combined_input = torch.cat((encoded_history, abio_input), dim=1)
        
        # then we will put this input through the FFNN to hopefully get some results
        logits_output = self.FFNN(combined_input)
        
        return logits_output


def predict(model, data, calc_loss=False):
    """
    Predicts phytoplankton concentrations using the provided neural network model and optionally calculates the loss.

    Parameters:
    model (object): The trained neural network model used for prediction.
    data (list): List of tuples, each containing:
        - abio_data (np.ndarray): Abiotic data array for a sample.
        - phyto_data (np.ndarray): Phytoplankton history matrix for a sample.
        - actual_concentrations (np.ndarray): Actual phytoplankton concentrations for a sample.
    calc_loss (bool, optional): Whether to calculate and return the average loss along with predictions. Defaults to False.

    Returns:
    tuple or np.ndarray: Depending on calc_loss parameter:
        - If calc_loss=True: Returns a tuple (concentrations, average_loss), where:
            - concentrations (np.ndarray): Predicted phytoplankton concentrations.
            - average_loss (float): Average loss computed over all samples in data.
        - If calc_loss=False: Returns concentrations (np.ndarray) containing only predicted phytoplankton concentrations.

    Notes:
    - The function sets the model to evaluation mode (eval()) to disable dropout and batch normalization.
    - It iterates over the input data to predict phytoplankton concentrations using model.forward().
    - If calc_loss=True, it calculates the loss between predicted concentrations and actual concentrations using model.loss().
    - If calc_loss=True, it returns both predicted concentrations and the average loss across all samples.
    - If calc_loss=False, it returns only the predicted concentrations.

    Example:
    >>> model = MyNeuralNetwork()
    >>> data = [([abio_data1, phyto_data1], actual_concentrations1), ([abio_data2, phyto_data2], actual_concentrations2)]
    >>> predictions = predict(model, data)
    >>> predictions.shape
    (2, 1)  # Example shape, depends on model output
    """

    # set the model to evaluation mode, to prevent dropout etc
    model.eval()

    all_losses = []
    # since we are not training wd can trun off the gradient tracking
    with torch.no_grad():

        # iterate over our data
        for (abio_data, phyto_data), actual_concentrations in data:
            
            # predicting the phytoplankton concentrations
            concentrations = model.forward(abio_data, phyto_data)

            # if we want to calculate the loss and keep track of it, we do the following
            if calc_loss:
                loss = model.loss((abio_data, phyto_data), actual_concentrations)
                all_losses.append(loss)

    # if we keep track of the loss, we return the average loss along with our concentrations
    if calc_loss:
        return concentrations, np.sum(all_losses) / len(data)

    return concentrations


def train_neural_model(
    model, 
    optimiser,
    data,
    trial_name,
    epochs=5, 
    check_every=10):
    """
    Trains a neural network model using the specified optimizer on the provided data, monitoring training progress
    and saving the best model based on evaluation loss.

    Parameters:
    model (object): The neural network model to train.
    optimiser (torch.optim.Optimizer): The optimizer used for training the model.
    data (pd.DataFrame): The input data containing both abiotic and phytoplankton data.
    trial_name (str): Name of the trial or model being trained, used for saving the best model checkpoint.
    epochs (int, optional): Number of epochs (iterations over the entire dataset) for training. Defaults to 5.
    check_every (int, optional): Frequency of evaluation checks (in number of steps) to save the best model based on evaluation loss. Defaults to 10.

    Returns:
    tuple: A tuple containing:
        - model (object): The trained neural network model.
        - training_loss_log (list): List of training losses recorded during training.
        - evaluation_loss_log (list): List of evaluation losses recorded during training.

    Notes:
    - The function splits the input data into training and evaluation sets using the data_splitter function.
    - It initializes logs for training and evaluation losses.
    - During training, it updates the model parameters based on backpropagation of loss computed using the model's loss function.
    - The best model checkpoint (lowest evaluation loss) is saved periodically based on check_every parameter.
    - After training, the function evaluates the model one last time on the evaluation set and records the final evaluation loss.
    - Training progress is visualized using a tqdm progress bar.
    - The trained model, training loss log, and evaluation loss log are returned as outputs.

    Example:
    >>> model = MyNeuralNetwork()
    >>> optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
    >>> data = pd.read_csv("data.csv")
    >>> trial_name = "trial_1"
    >>> trained_model, train_losses, eval_losses = train_neural_model(model, optimiser, data, trial_name, epochs=10)
    """
    
    # splitting the data for a single location into a training and evaluation split
    training_split, evaluation_split = data_splitter(data)

    # initializing the logs for the loss of both the training and evaluation set
    training_loss_log = []
    evaluation_loss_log = []

    # we do a first pass of the evaluation set and calculate the current loss (as a base thingy)
    _, eval_loss = model.predict(
        evaluation_split,
        calc_loss=True
    )

    # and record the loss ofcourse
    evaluation_loss_log.append(eval_loss)

    best_devloss = eval_loss

    # calculate number of steps the model will be srunning for
    total_steps = epochs * len(training_split)
    step = 0

    # and then train the model (with a fancy schmancy loading bar ofcourse)
    with tqdm(range(total_steps)) as bar:        
        for epoch in range(epochs):
            for (abio_data, phyto_data), actual_concentrations in training_split:

                #[-----------------------Start Training Loop----------------------------]
                # setting the mode to "training" so we use dropout etc
                model.train()  

                # reset gradients
                optimiser.zero_grad()

                # calculate loss
                loss = model.loss((abio_data, phyto_data), actual_concentrations)
                
                # backpropogate through the model
                loss.backward()

                # update parameters with optimal direction according to gradient descent
                optimiser.step()

                #[-------------------------End Training Loop----------------------------]
    
                # updating progress bar ;)
                bar.set_postfix({
                    'loss': f"{loss:.2f}"
                })
                bar.update()
                training_loss_log.append(loss)

                # every so often we check whether the model has improved or not (so we can save the best version)
                if step % check_every == 0: 

                    # we do a first pass of the evaluation set and calculate the current loss (as a base thingy)
                    _, eval_loss = model.predict(
                        evaluation_split,
                        calc_loss=True
                    )

                    # and record the loss ofcourse
                    evaluation_loss_log.append(eval_loss)

                    if eval_loss <= best_devloss:
                        torch.save(model.state_dict(), f"models/{trial_name}.pt")
                        best_devloss = eval_loss                    
    
                step += 1

    # once we are done with training we evaluate one last time
    _, eval_loss = model.predict(
        evaluation_split,
        calc_loss=True
    )

    # and record the loss ofcourse
    evaluation_loss_log.append(eval_loss)

    return model, training_loss_log, evaluation_loss_log


def data_splitter(data, abio_columns, phyto_columns, shuffled_rows=True, random_seed=None, train_ratio=0.7, lookback=-1):
    """
    Splits the input data into training and evaluation sets for machine learning tasks involving abiotic and phytoplankton data.

    Parameters:
    data (pd.DataFrame): The input DataFrame containing both abiotic and phytoplankton data.
    abio_columns (list of str): List of column names for abiotic data.
    phyto_columns (list of str): List of column names for phytoplankton data.
    shuffled_rows (bool, optional): Whether to shuffle the rows of the dataset before splitting. Defaults to True.
    random_seed (int, optional): Random seed for reproducibility. Defaults to None.
    train_ratio (float, optional): Ratio of data to use for training (between 0 and 1). Defaults to 0.7.
    lookback (int, optional): Number of previous time steps to consider for each sample. Defaults to -1 (uses all previous data).

    Returns:
    tuple: A tuple containing two lists:
        - training_data (list): List of tuples, each containing:
            - abio_data (np.ndarray): Abiotic data array for a sample.
            - phyto_data (np.ndarray): Phytoplankton history matrix for a sample.
            - phyto_concentration (np.ndarray): Actual phytoplankton concentrations for a sample.
        - evaluation_data (list): List of tuples, similar to training_data, for evaluation purposes.

    Notes:
    - The function ensures that required columns ('DATUM' for location and date) are present in the input data.
    - It splits the data into training and evaluation sets based on the specified train_ratio.
    - For each sample, it extracts abiotic data, phytoplankton concentrations, and optionally a history of previous phytoplankton data.
    - If lookback is negative, all previous data is considered; otherwise, only the specified number of previous time steps is used.
    - The function handles cases where lookback might lead to accessing rows before the start of the dataset.

    Example:
    >>> data = pd.DataFrame({
    ...     'DATUM': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'],
    ...     'Abio1': [1.0, 2.0, 3.0, 4.0],
    ...     'Abio2': [5.0, 6.0, 7.0, 8.0],
    ...     'Phyto1': [0.1, 0.2, 0.3, 0.4],
    ...     'Phyto2': [0.5, 0.6, 0.7, 0.8]
    ... })
    >>> abio_columns = ['Abio1', 'Abio2']
    >>> phyto_columns = ['Phyto1', 'Phyto2']
    >>> train_data, eval_data = data_splitter(data, abio_columns, phyto_columns)
    >>> len(train_data), len(eval_data)
    (3, 1)
    >>> len(train_data[0]), len(eval_data[0])
    (2, 2)
    """

    indices_list = list(range(len(data)))

    # initializing the random seed in case we want to compare runs with different settings
    if random_seed is not None:
        random.seed(random_seed)

    # shuffling the indices so we get splits that have intermixed indices
    if shuffled_rows:
        random.shuffle(indices_list)

    # assert some important columns are present (currently just Datum but maybe future rewrite for handling all locations at the same time)
    loc_date_columns = ["DATUM"]

    for column in loc_date_columns:
        if column not in data.columns:
            print(f"The expected column {column} is not present in the given dataset, while it is required")
            return

    train_row_count = int(train_ratio * len(data))

    # dividing the indices up
    train_indices = sorted(indices_list[:train_row_count])
    evaluation_indices = sorted(indices_list[train_row_count:])

    training_data = []
    evaluation_data = []

    # we want to prepare the data for both the testing and evaluation splits
    for index_split, data_list in [(train_indices, training_data), (evaluation_indices, evaluation_data)]:
        for index in index_split:

            # first grabbing the abiotic data and the actual phytoplankton concentrations at the timestamp and location
            abio_data = data.loc(index, abio_columns).to_numpy()
            phyto_concentration = data.loc(index, phyto_columns).to_numpy()

            # if no lookback has been decided, the lstm will use all previous data
            if lookback < 0:
                start_index = 0
            
            # else we will only look at the specified amount of previous values
            else:  
                # gotta be careful not to get an index < 0 since it will grab the last rows
                start_index = max(0, index - lookback)

            # creating the phytoplankton history matrix
            phyto_data = data.loc[start_index:index - 1, phyto_columns].to_numpy()

            # lastly we add the processed data to the data lists
            data_list.append(((abio_data, phyto_data), phyto_concentration))
    
    return training_data, evaluation_data


def aggregate_phyto_data(data, clustered_phyto_types, keep_original_columns=False, fill_methods=["forward_fill", "backward_fill"], labels=None):
    """
    Aggregates phytoplankton data by filling missing values and summing specified groups of phytoplankton types.

    Parameters:
    data (pd.DataFrame): The input DataFrame containing phytoplankton data (possibly with missing values and must have a DATUM column).
    clustered_phyto_types (list of list of str): A list where each sublist contains the names of phytoplankton types to be grouped together.
    keep_original_columns (bool, optional): If True, retains the original phytoplankton columns in the DataFrame. Defaults to False.
    fill_methods (list of str, optional): A list of methods to fill missing values. Supported methods are "forward_fill", "backward_fill", and "linear_interpolate". Defaults to ["forward_fill", "backward_fill"].
    labels (list of str, optional): A list of labels for the new aggregated columns. If None, default labels "group_{index}" will be used. Defaults to None.

    Returns:
    pd.DataFrame: A DataFrame with filled missing values and aggregated phytoplankton data. Original columns are dropped unless keep_original_columns is set to True.

    Notes:
    - Missing values in the DataFrame are filled based on the specified fill methods. If a method is not recognized, forward fill is used as the default.
    - The aggregation is done by summing the values of the specified groups of phytoplankton types, and the results are added as new columns to the DataFrame.
    - The original phytoplankton columns are dropped by default unless keep_original_columns is set to True.

    Example:
    >>> data = pd.DataFrame({
    ...     'LOC_CODE': ['A', 'A', 'B', 'B'],
    ...     'Acg': [1, np.nan, 3, 4],
    ...     'Bfg': [5, 6, np.nan, 8],
    ...     'Dng': [9, 10, 11, np.nan]
    ... })
    >>> clustered_phyto_types = [['Acg', 'Bfg'], ['Dng']]
    >>> aggregate_phyto_data(data, clustered_phyto_types)
    LOC_CODE  group_0  group_1
    0        A      6.0     9.0
    1        A     12.0    10.0
    2        B      3.0    11.0
    3        B     12.0     NaN
    """

    # first we have to fill in the missing values (or at least most of them)
    # applying the different specified filling methods to the data within a single location
    for method in fill_methods:

        if method == "forward_fill":
            fill_func = lambda group: group.ffill()
        elif method == "backward_fill":
            fill_func = lambda group: group.bfill()
        elif method == "linear_interpolate":
            fill_func = lambda group: group.interpolate('linear')

        # just defaulting to forward fill if a specified fill method is not supported
        else:
            print(f'Fill method {method} was not found, please choose one from ["forward_fill", "backward_fill", "linear_interpolate"]')
            print('Defaulted to "forward_fill"')
            fill_func = lambda group: group.ffill()

        data = data.groupby("LOC_CODE").apply(fill_func).reset_index(drop=True)

    # if no labels are given, we just set the labels to "group_{index}"
    if labels is None:
        labels = [f"group_{i}" for i in range(len(clustered_phyto_types))]

    # summing the different types of phytoplankton inside a group
    # adding the summed result as a new columns, in place
    for label, cluster in zip(labels, clustered_phyto_types):
        data[label] = data[cluster].sum(axis=1)
    
    # sometimes we would want to keep the original phytoplankton columns
    if keep_original_columns:

        # so we just return before dropping these columns
        return data

    # dropping the columns
    phyto_columns = {phyto_type for cluster in clustered_phyto_types for phyto_type in cluster}

    return data.drop(phyto_columns, axis=1)


if __name__ == "__main__":

    # splitting the phytoplankton randomly
    # random_split = np.split(np.asarray(phyto_columns), 5)

    df = pd.read_excel('../../data/MERGED_DATA_180624.xlsx', sheet_name='MERGE_FINAL')

    columns = list(df.columns)

    non_phyto_columns = ['TIJD', 'ZS [mg/l]', 'ZICHT [dm]', 'T [oC]', 'SiO2 [umol/L]', 'SALNTT [DIMSLS]', 'PO4 [umol/L]', 'pH [DIMSLS]', 'NO3 [umol/L]', 'NO2 [umol/L]', 'NH4 [umol/L]', 'E [/m]', 'E_method', 'CHLFa [ug/l]', '    Q', 'PAR [J/m2d]', 'PAR [kJ/m2d]', 'kPAR_7d', 'kPAR_14d', 'DIN', 'DIN:SRP', 'DIN:SI', 'SRP:SI', 'IM [Jm2d]', 'interpolated_columns']

    loc_date_columns = ["LOC_CODE", "DATUM"]
