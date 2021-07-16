import csv
import itertools
import json
import numpy as np
import torch
import warnings
from collections import deque

class PreprocessLoader(object):

    def __init__(self):
        """Load preprocessed data"""
        # Create loader object
        self.loader = Loader()
        # Create filter for preprocessed data
        self.filter = Filter()

    def load(self, infile, dim_in, dim_out=1, max=float('inf'),
        key=lambda x: x.get('src'), extract=[], train_ratio=0.5, random=False):
        """Load sequences from input file

            Parameters
            ----------
            infile : string
                Path to input file

            dim_in : int
                Dimension of input sequence

            dim_out : int, default=1
                Dimension of output sequence

            max : float, default=inf
                Maximum number of items to extract

            key : func, default=lambda x: x.get('src')
                Group input by this input key, default is 'src'.

            extract : list, default=list()
                Fields to extract

            train_ratio : float, default=0.5
                Ratio to train with

            random : boolean, default=False
                Whether to split randomly
            """
        # Load data
        data, encodings = self.load_sequences(infile, dim_in, dim_out, max,
                                              key=key, extract=extract)
        # Split data
        data = self.split_train_test(data, train_ratio, random)
        # Split data on input and output
        for k, v in data.items():
            for k2, v2 in v.items():
                if k == 'key':
                    data[k][k2] = {'X': v2, 'y': v2}
                else:
                    data[k][k2] = {'X': v2[:, :-dim_out ],
                                   'y': v2[:,  -dim_out:]}

        # Return result
        return data, encodings

    def load_sequences(self, infile, dim_in, dim_out=1, max=float('inf'),
        key = lambda x: x.get('src'), extract=[]):
        """Load sequences from input file

            Parameters
            ----------
            infile : string
                Path to input file

            dim_in : int
                Dimension of input sequence

            dim_out : int, default=1
                Dimension of output sequence

            max : float, default=inf
                Maximum number of items to extract

            key : func, default=lambda x: x.get('src')
                Group input by this input key, default is 'src'.

            extract : list, default=[]
                Fields to extract

            Returns
            -------
            data : dict()
                Dictionary of key -> data

            encodings : dict()
                Dictionary of key -> mapping
            """
        # Initialise encodings
        encodings = {k: dict() for k in ['key'] + extract}
        # Initialise output
        result = {k: list() for k in ['key'] + extract}

        # Read data
        data = self.loader.load(infile, max=max, decode=True)

        # Read sequences from data
        for key, datapoint in self.filter.ngrams(data, dim_in+dim_out,
            group = key,
            key   = lambda x: tuple(x.get(item) for item in extract)):

            # Unpack data
            datapoint = {k: v for k, v in zip(extract, zip(*datapoint))}
            datapoint['key'] = key

            # Store data
            for k, v in datapoint.items():
                # Transform data if necessary
                if k in extract:
                    for x in v:
                        if x not in encodings[k]: encodings[k][x] = len(encodings[k])
                    v = [encodings[k][x] for x in v]
                elif k == 'key':
                    if v not in encodings[k]: encodings[k][v] = len(encodings[k])
                    v = encodings[k][v]

                # Update datapoint
                data_ = result.get(k, list())
                data_.append(v)
                result[k] = data_

        # Get data as tensors
        result = {k: torch.Tensor(v).to(torch.int64) for k, v in result.items()}

        # Return result
        return result, encodings

    def split_train_test(self, data, train_ratio=0.5, random=False):
        """Split data in train and test sets

            Parameters
            ----------
            data : dict()
                Dictionary of identifier -> array-like of data

            train_ratio : float, default=0.5
                Ratio of training samples

            random : boolean, default=False
                Whether to split randomly
            """
        # Get number of samples
        n_samples = next(iter(data.values())).shape[0]

        # Select training and testing data
        # Initialise training
        i_train = np.zeros(n_samples, dtype=bool)
        if random:
            # Set training data to randomly selected
            i_train[np.random.choice(
                    np.arange(i_train.shape[0]),
                    round(0.5*i_train.shape[0]),
                    replace=False
            )] = True
        else:
            # Set training data to first half
            i_train[:int(n_samples*train_ratio)] = True
        # Testing is everything not in training
        i_test  = ~i_train

        # Split into train and test data
        for k, v in data.items():
            data[k] = {'train': v[i_train], 'test': v[i_test]}

        # Return result
        return data



################################################################################
#                    Object for filtering and grouping data                    #
################################################################################
class Filter(object):
    """Filter object for filtering and grouping json data."""

    def __init__(self):
        """Filter object for filtering and grouping json data."""
        pass

    def groupby(self, data, key):
        """Split data by key

            Parameters
            ----------
            data : iterable
                Iterable to split

            key : func
                Function by which to split data

            Yields
            ------
            key : Object
                Key value of item

            item : Object
                Datapoint of data
            """
        for k, v in itertools.groupby(data, key=key):
            for x in v:
                yield k, x

    def aggregate(self, data, group, key=lambda x: x):
        """Aggregate data by key

            Parameters
            ----------
            data : iterable
                Iterable to aggregate

            group : func
                Function by which to split data

            key : func
                Function by which to aggregate data

            Returns
            -------
            result : dict()
                Dictionary of key -> list of datapoints
            """
        # Initialise result
        result = dict()

        # Loop over datapoints split by key
        for k, v in self.groupby(data, group):
            # Add datapoint
            buffer = result.get(k, [])
            buffer.append(key(v))
            result[k] = buffer

        # Return result
        return result

    def ngrams(self, data, n, group, key=lambda x: x):
        """Aggregate data by key

            Parameters
            ----------
            data : iterable
                Iterable to aggregate

            n : int
                Length of n-gram

            group : func
                Function by which to split data

            key : func
                Function by which to aggregate data

            Returns
            -------
            result : dict()
                Dictionary of key -> list of datapoints
            """
        # Initialise result
        result = dict()

        # Loop over datapoints split by key
        for k, v in self.groupby(data, group):
            # Add datapoint
            buffer = result.get(k, deque())
            buffer.append(key(v))
            # Yield if we find n-gram
            if len(buffer) >= n:
                # Yield buffergroup=lambda x: (x.get('source'), x.get('src_ip')), key=lambda x: x.get('detector_name')
                yield k, tuple(buffer)
                # Remove last item
                buffer.popleft()
            # Store buffer
            result[k] = buffer



################################################################################
#                         Object for loading csv files                         #
################################################################################
class Loader(object):
    """Loader for data from preprocessed files"""

    def load(self, infile, max=float('inf'), decode=False):
        """Load data from given input file

            Parameters
            ----------
            infile : string
                Path to input file from which to load data

            max : float, default='inf'
                Maximum number of events to load from input file

            decode : boolean, default=False
                If True, it decodes data from input file
            """
        # Initialise encoding
        encoding = {}

        # Read encoding file
        if decode:
            try:
                with open("{}.encoding.json".format(infile)) as file:
                    # Read encoding as json
                    encoding = json.load(file)
                    # Transform
                    for k, v in encoding.items():
                        encoding[k] = {str(i): item for i, item in enumerate(v)}
            except FileNotFoundError as e:
                warnings.warn("Could not decode: '{}'".format(e))

        # Read input file
        with open(infile) as infile:
            # Create csv reader
            reader = csv.DictReader(infile)

            # Read data
            for i, data in enumerate(reader):
                # Break on max
                if i >= max: break

                # Decode data
                if decode:
                    yield {k: encoding.get(k, {}).get(v, v) for k, v in data.items()}
                # Or yield data
                else:
                    # Yield result as ints where possible
                    result = dict()
                    for k, v in data.items():
                        try:
                            result[k] = int(v)
                        except ValueError:
                            result[k] = v
                    yield result
