import argparse
import torch
import torch.nn as nn
import torch.optim as optim

class DeepLog(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers=2, window_size=10):
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

            window_size : int, default=10
                Size of input windows
            """
        # Initialise nn.Module
        super(DeepLog, self).__init__()

        # Store input parameters
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers  = num_layers
        self.window_size = window_size

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    #                             Train model                              #
    ########################################################################

    def train(self, data, epochs=100, criterion=nn.CrossEntropyLoss(), verbose=False):
        """Train model with given data.

            Parameters
            ----------
            data : iterable
                Iterable of input, output labels.

            epochs : int, default=100
                Number of epochs to train with.

            criterion : torch.nn.modules.loss, default=nn.CrossEntropyLoss()
                Loss function

            verbose : boolean, default=False
                If true, output progress
            """
        # Initialise Adam optimizer
        optimizer = optim.Adam(self.parameters())

        # Perform each epoch
        for epoch in range(epochs):
            # Reset total loss
            loss_total = 0

            # Loop over entire dataset
            for i, (X, y) in enumerate(data):
                # Transform the input to proper size
                X = X.detach().view(-1, self.window_size, self.input_size).to(self.device)
                # Perform forward pass
                output = self(X)
                # Compute loss
                loss = criterion(output, y.to(self.device))
                # Increment total loss
                loss_total += loss

                # Perform backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Print output if verbose
                if verbose:
                    # Show current epoch
                    print("Epoch [{:{}}/{}]".format(epoch+1, len(str(epochs)), epochs), end=' ')
                    # Show average loss
                    print("loss = {:.4f}".format(loss_total/(i+1)), end=' ')
                    # Show progress bar
                    print("{}{} ({:6.2f}%)".format(
                        "#"*round( 50 * min(1,    (i+1)/len(data) )),
                        "."*round( 50 * max(0, (1-(i+1)/len(data)))),
                        100*(i+1)/len(data)), end='\r')

            # Print output if verbose
            if verbose: print()


    ########################################################################
    #                            Predict method                            #
    ########################################################################

    def predict(self, X, y, n, batch_size=4096):
        """Predict data

            Parameters
            ----------
            X : iterable
                Iterable of input data to predict anomalous

            y : iterable
                Iterable of real label to predict anomalous

            n : int
                Number of predictions to test for

            Returns
            -------
            result : torch.tensor
                Predictions of anomalies (-1) or benign samples (+1)
            """
        # Initialise result
        result = torch.zeros(y.shape[0])
        # Prepare data
        X = X.detach().view(-1, self.window_size, self.input_size).to(self.device)

        # Predict data in batches
        for i in range(0, X.shape[0], batch_size):
            # Get real y
            y_real = y[i:min(i+batch_size, X.shape[0])]

            # Predict data
            y_pred = self(X[i:min(i+batch_size, X.shape[0])])
            # Get top N arguments
            y_pred = torch.argsort(y_pred, 1)[:, -n:]

            # Check if y is in predicted
            result_ = (y_real.view(-1, 1) == y_pred).any(1)
            # Add result
            result[i:min(i+batch_size, X.shape[0])] = result_

            print("{}/{}".format(i, X.shape[0]), end='\r')

        # Return result
        return result

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
                self.num_layers, X.size(0), self.hidden_size
            ).to(self.device)
