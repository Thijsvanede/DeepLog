import random
import torch
from torch.utils.data import DataLoader, TensorDataset

class VariableDataLoader(object):
    """Load data from variable length inputs

        Attributes
        ----------
        lengths : dict()
            Dictionary of input-length -> input samples

        index : boolean, default=False
            If True, also returns original index

        batch_size : int, default=1
            Size of each batch to output

        shuffle : boolean, default=True
            If True, shuffle the data randomly, each yielded batch contains
            only input items of the same length
        """

    def __init__(self, X, y, index=False, batch_size=1, shuffle=True):
        """Load data from variable length inputs

            Parameters
            ----------
            X : iterable of shape=(n_samples,)
                Input sequences
                Each item in iterable should be a sequence (of variable length)

            y : iterable of shape=(n_samples,)
                Labels corresponding to X

            index : boolean, default=False
                If True, also returns original index

            batch_size : int, default=1
                Size of each batch to output

            shuffle : boolean, default=True
                If True, shuffle the data randomly, each yielded batch contains
                only input items of the same length
            """
        # Get inputs by length
        self.lengths = dict()
        # Loop over inputs
        for i, (X_, y_) in enumerate(zip(X, y)):
            X_length, y_length, i_length = self.lengths.get(len(X_), (list(), list(), list()))
            X_length.append(X_)
            y_length.append(y_)
            i_length.append(i)
            self.lengths[len(X_)] = (X_length, y_length, i_length)

        # Transform to tensors
        for k, v in self.lengths.items():
            self.lengths[k] = (
                torch.as_tensor(v[0]),
                torch.as_tensor(v[1]),
                torch.as_tensor(v[2])
            )

        # Set index
        self.index = index
        # Set batch_size
        self.batch_size = batch_size
        # Set shuffle
        self.shuffle = shuffle
        # Reset
        self.reset()

        # Get keys
        self.keys = set(self.data.keys())

    def reset(self):
        """Reset the VariableDataLoader"""
        # Reset done
        self.done = set()
        # Reset DataLoaders
        self.data = { k: iter(DataLoader(
            TensorDataset(v[0], v[1], v[2]),
            batch_size = self.batch_size,
            shuffle    = self.shuffle))
            for k, v in self.lengths.items()
        }

    def __iter__(self):
        """Returns iterable of VariableDataLoader"""
        # Reset
        self.reset()
        # Return self
        return self

    def __next__(self):
        """Get next item of VariableDataLoader"""
        # Check if we finished the iteration
        if self.done == self.keys:
            self.reset()
            # Stop iterating
            raise StopIteration

        # Select key
        if self.shuffle:
            key = random.choice(list(self.keys - self.done))
        else:
            key = sorted(self.keys - self.done)[0]

        # Yield next item in batch
        try:
            X_, y_, i = next(self.data.get(key))
            if self.index:
                item = (X_, y_, i)
            else:
                item = (X_, y_)
        except StopIteration:
            # Add key
            self.done.add(key)
            # Get item iteratively
            item = next(self)

        # Return next item
        return item
