import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from module import Module

class DeepLog(Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        """DeepLog model used for training and predicting logs.

            Parameters
            ----------
            input_size : int
                Dimension of input layer.

            hidden_size : int
                Dimension of hidden layer.

            output_size : int
                Dimension of output layer.

            num_layers : int, default=2
                Number of hidden layers, i.e. stacked LSTM modules.
            """
        # Initialise nn.Module
        super(DeepLog, self).__init__()

        # Store input parameters
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers  = num_layers

        # Initialise model layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.out  = nn.Linear(hidden_size, output_size)

    ########################################################################
    #                       Forward through network                        #
    ########################################################################

    def forward(self, X):
        """Forward sample through DeepLog.

            Parameters
            ----------
            X : tensor
                Input to forward through DeepLog network.

            Returns
            -------
            result : tensor

            """
        # One hot encode X
        X = F.one_hot(X, self.input_size).to(torch.float)

        # Set initial hidden states
        hidden = self._get_initial_state(X)
        state  = self._get_initial_state(X)

        # Perform LSTM layer
        out, hidden = self.lstm(X, (hidden, state))
        # Perform output layer
        out = self.out(out[:, -1, :])

        # Return result
        return out


    ########################################################################
    #                            Predict method                            #
    ########################################################################

    def predict(self, X, y=None, k=1, variable=False, verbose=True):
        """Predict the k most likely output values

            Parameters
            ----------
            X : torch.Tensor of shape=(n_samples, seq_len)
                Input of sequences, these will be one-hot encoded to an array of
                shape=(n_samples, seq_len, input_size)

            k : int, default=1
                Number of output items to generate

            variable : boolean, default=False
                If True, predict inputs of different sequence lengths

            verbose : boolean, default=True
                If True, print output

            Returns
            -------
            result : torch.Tensor of shape=(n_samples, k)
                k most likely outputs

            confidence : torch.Tensor of shape=(n_samples, k)
                Confidence levels for each output
            """
        # Get the predictions
        result = super().predict(X, variable=variable, verbose=verbose)
        # Get the probabilities from the log probabilities
        result = result.exp()
        # Compute k most likely outputs
        confidence, result = result.topk(k)
        # Return result
        return result, confidence

    ########################################################################
    #                             I/O methods                              #
    ########################################################################

    def save(self, path):
        """Save trained model to path

            Parameters
            ----------
            path : string
                Path to output trained model
            """
        torch.save(self.state_dict(), path)

    def load(self, path):
        """Load trained model from path

            Parameters
            ----------
            path : string
                Path from which to load model.
            """
        # Load from path
        self.load_state_dict(torch.load(path))
        # Return self
        return self

    ########################################################################
    #                         Auxiliary functions                          #
    ########################################################################

    def _get_initial_state(self, X):
        """Return a given hidden state for X."""
        # Return tensor of correct shape as device
        return torch.zeros(
            self.num_layers ,
            X.size(0)       ,
            self.hidden_size
        ).to(X.device)
