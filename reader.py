import torch

class Reader(object):

    def __init__(self):
        """Log reader object for parsing given data."""
        pass

    def read(self, path, window_size=10):
        """Read log file.

            Parameters
            ----------
            path : string
                Path to log file that must be read.
                Log file must be of format:

                key_a1 key_a2 key_a3 key_a4 key_a5
                key_b1 key_b2 key_b3
                key_c1 key_c2 key_c3 ... key_ci
                ...

            window_size : int, default=10
                Size of window for which to read X.

            Returns
            -------
            X : Tensor of shape=(n_samples, window_size)
                Input samples

            y : Tensor of shape=(n_samples,)
                Corresponding labels
            """
        # Initialise input and labels
        X = list()
        y = list()

        # Open input file
        with open(path, 'r') as infile:
            # Read lines
            for line in infile.readlines():
                # Get line as integer array
                data = list(map(int, line.strip().split()))
                # Compute labels as next in sequence
                y.extend(data[window_size:])
                # Split line in sequences of size <window>
                for i in range(len(data) - window_size):
                    X.append(data[i:i+window_size])

        # Transform to torch tensor
        X = torch.tensor(X, dtype=torch.float) - 1
        y = torch.tensor(y) - 1

        # Return as dataset
        return X, y
