import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtrain import Module

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
        self.lstm    = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.out     = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

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
        X = F.one_hot(X.to(torch.int64), self.input_size).to(torch.float)

        # Set initial hidden states
        hidden = self._get_initial_state(X)
        state  = self._get_initial_state(X)

        # Perform LSTM layer
        out, hidden = self.lstm(X, (hidden, state))
        # Perform output layer
        out = self.out(out[:, -1, :])
        # Create probability
        out = self.softmax(out)

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

            y : Ignored
                Ignored

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

    def save(self, outfile):
        """Save model to output file.

            Parameters
            ----------
            outfile : string
                File to output model.
            """
        # Save to output file
        torch.save(self.state_dict(), outfile)

    @classmethod
    def load(cls, infile, device=None):
        """Load model from input file.

            Parameters
            ----------
            infile : string
                File from which to load model.
            """
        # Load state dictionary
        state_dict = torch.load(infile, map_location=device)

        print(state_dict.keys())

        # Get input variables from state_dict
        input_size  = state_dict.get('lstm.weight_ih_l0').shape[1]
        hidden_size = state_dict.get('lstm.weight_hh_l0').shape[1]
        output_size = input_size
        num_layers  = (len(state_dict) - 2) // 4

        # Create ContextBuilder
        result = cls(
            input_size  = input_size,
            hidden_size = hidden_size,
            output_size = output_size,
            num_layers  = num_layers,
        )

        # Cast to device if necessary
        if device is not None: result = result.to(device)

        # Set trained parameters
        result.load_state_dict(state_dict)

        # Return result
        return result

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
