import numpy  as np
import pandas as pd
import torch
from tqdm import tqdm

class Preprocessor(object):
    """Preprocessor for loading data from standard data formats."""

    def __init__(self, length, timeout, NO_EVENT=-1337):
        """Preprocessor for loading data from standard data formats.

            Parameters
            ----------
            length : int
                Number of events in context.

            timeout : float
                Maximum time between context event and the actual event in
                seconds.

            NO_EVENT : int, default=-1337
                ID of NO_EVENT event, i.e., event returned for context when no
                event was present. This happens in case of timeout or if an
                event simply does not have enough preceding context events.
            """
        # Set context length
        self.context_length = length
        self.timeout        = timeout

        # Set no-event event
        self.NO_EVENT = NO_EVENT

        # Set required columns
        self.REQUIRED_COLUMNS = {'timestamp', 'event', 'machine'}


    ########################################################################
    #                      General data preprocessing                      #
    ########################################################################

    def sequence(self, data, labels=None, verbose=False, mapping=None):
        """Transform pandas DataFrame into DeepCASE sequences.

            Parameters
            ----------
            data : pd.DataFrame
                Dataframe to preprocess.

            labels : int or array-like of shape=(n_samples,), optional
                If a int is given, label all sequences with given int. If an
                array-like is given, use the given labels for the data in file.
                Note: will overwrite any 'label' data in input file.

            verbose : boolean, default=False
                If True, prints progress in transforming input to sequences.

            mapping : dict(), default=None, optional
                If mapping is provided, use given mapping. Any additional unique values
                will be appended to the given mapping. NO_EVENT may be present in the 
                provided mapping, but it is not required.

            Returns
            -------
            context : torch.Tensor of shape=(n_samples, context_length)
                Context events for each event in events.

            events : torch.Tensor of shape=(n_samples,)
                Events in data.

            labels : torch.Tensor of shape=(n_samples,)
                Labels will be None if no labels parameter is given, and if data
                does not contain any 'labels' column.

            mapping : dict()
                Mapping from new event_id to original event_id.
                Sequencing will map all events to a range from 0 to n_events.
                This is because event IDs may have large values, which is
                difficult for a one-hot encoding to deal with. Therefore, we map
                all Event ID values to a new value in that range and provide
                this mapping to translate back.
            """
        ################################################################
        #                  Transformations and checks                  #
        ################################################################

        # Case where a single label is given
        if isinstance(labels, int):
            # Set given label to all labels
            labels = np.full(data.shape[0], labels, dtype=int)

        # Transform labels to numpy array
        labels = np.asarray(labels)

        # Check if data contains required columns
        if set(data.columns) & self.REQUIRED_COLUMNS != self.REQUIRED_COLUMNS:
            raise ValueError(
                ".csv file must contain columns: {}"
                .format(list(sorted(self.REQUIRED_COLUMNS)))
            )

        # Check if labels is same shape as data
        if labels.ndim and labels.shape[0] != data.shape[0]:
            raise ValueError(
                "Number of labels: '{}' does not correspond with number of "
                "samples: '{}'".format(labels.shape[0], data.shape[0])
            )

        ################################################################
        #                          Map events                          #
        ################################################################

        # Create mapping of events
        if mapping is None:
            mapping = {
                i: event for i, event in enumerate(np.unique(data['event'].values))
            }
            
            # Check that NO_EVENT is not in events
            if self.NO_EVENT in mapping.values():
                raise ValueError(
                    "NO_EVENT ('{}') is also a valid Event ID".format(self.NO_EVENT)
                )
            
            mapping[len(mapping)] = self.NO_EVENT
        
        # Use given mapping and add missing events
        else:
            mapping = mapping.copy()
            difference = list(set(data['event'].values) - set(mapping.values()))
            max_key = len(mapping)
            for i, val in enumerate(difference):
                mapping[i+max_key] = val

            # Check that NO_EVENT is not in unseen events
            if self.NO_EVENT in difference:
                raise ValueError(
                    "NO_EVENT ('{}') is also a valid Event ID".format(self.NO_EVENT)
                )
            
            # Add NO_EVENT if it is not present in mapping
            if self.NO_EVENT not in mapping.values():
                mapping[len(mapping)] = self.NO_EVENT

        mapping_inverse = {v: k for k, v in mapping.items()}

        # Apply mapping
        data['event'] = data['event'].map(mapping_inverse)

        ################################################################
        #                      Initialise results                      #
        ################################################################

        # Set events as events
        events = torch.Tensor(data['event'].values).to(torch.long)

        # Set context full of NO_EVENTs
        context = torch.full(
            size       = (data.shape[0], self.context_length),
            fill_value = mapping_inverse[self.NO_EVENT],
        ).to(torch.long)

        # Set labels if given
        if labels.ndim:
            labels = torch.Tensor(labels).to(torch.long)
        # Set labels if contained in data
        elif 'label' in data.columns:
            labels = torch.Tensor(data['label'].values).to(torch.long)
        # Otherwise set labels to None
        else:
            labels = None

        ################################################################
        #                        Create context                        #
        ################################################################

        # Sort data by timestamp
        data = data.sort_values(by='timestamp')

        # Group by machines
        machine_grouped = data.groupby('machine')
        # Add verbosity
        if verbose: machine_grouped = tqdm(machine_grouped, desc='Loading')

        # Group by machine
        for machine, events_ in machine_grouped:
            # Get indices, timestamps and events
            indices    = events_.index.values
            timestamps = events_['timestamp'].values
            events_    = events_['event'].values

            # Initialise context for single machine
            machine_context = np.full(
                (events_.shape[0], self.context_length),
                mapping_inverse[self.NO_EVENT],
                dtype = int,
            )

            # Loop over all parts of the context
            for i in range(self.context_length):

                # Compute time difference between context and event
                time_diff = timestamps[i+1:] - timestamps[:-i-1]
                # Check if time difference is larger than threshold
                timeout_mask = time_diff > self.timeout

                # Set mask to NO_EVENT
                machine_context[i+1:, self.context_length-i-1] = np.where(
                    timeout_mask,
                    mapping_inverse[self.NO_EVENT],
                    events_[:-i-1],
                )

            # Convert to torch Tensor
            machine_context = torch.Tensor(machine_context).to(torch.long)
            # Add machine_context to context
            context[indices] = machine_context

        ################################################################
        #                        Return results                        #
        ################################################################

        # Return result
        return context, events, labels, mapping


    ########################################################################
    #                     Preprocess different formats                     #
    ########################################################################

    def csv(self, path, nrows=None, labels=None, verbose=False, mapping=None):
        """Preprocess data from csv file.

            Note
            ----
            **Format**: The assumed format of a .csv file is that the first line
            of the file contains the headers, which should include
            ``timestamp``, ``machine``, ``event`` (and *optionally* ``label``).
            The remaining lines of the .csv file will be interpreted as data.

            Parameters
            ----------
            path : string
                Path to input file from which to read data.

            nrows : int, default=None
                If given, limit the number of rows to read to nrows.

            labels : int or array-like of shape=(n_samples,), optional
                If a int is given, label all sequences with given int. If an
                array-like is given, use the given labels for the data in file.
                Note: will overwrite any 'label' data in input file.

            verbose : boolean, default=False
                If True, prints progress in transforming input to sequences.

            mapping : dict(), default=None, optional
                If mapping is provided, use given mapping. Any additional unique values
                will be appended to the given mapping. NO_EVENT may be present in the 
                provided mapping, but it is not required.

            Returns
            -------
            events : torch.Tensor of shape=(n_samples,)
                Events in data.

            context : torch.Tensor of shape=(n_samples, context_length)
                Context events for each event in events.

            labels : torch.Tensor of shape=(n_samples,)
                Labels will be None if no labels parameter is given, and if data
                does not contain any 'labels' column.
            """
        # Read data from csv file into pandas dataframe
        data = pd.read_csv(path, nrows=nrows)

        # Transform to sequences and return
        return self.sequence(data, labels=labels, verbose=verbose, mapping=mapping)


    def json(self, path, labels=None, verbose=False, mapping=None):
        """Preprocess data from json file.

            Note
            ----
            json preprocessing will become available in a future version.

            Parameters
            ----------
            path : string
                Path to input file from which to read data.

            labels : int or array-like of shape=(n_samples,), optional
                If a int is given, label all sequences with given int. If an
                array-like is given, use the given labels for the data in file.
                Note: will overwrite any 'label' data in input file.

            verbose : boolean, default=False
                If True, prints progress in transforming input to sequences.

            mapping : dict(), default=None, optional
                If mapping is provided, use given mapping. Any additional unique values
                will be appended to the given mapping. NO_EVENT may be present in the 
                provided mapping, but it is not required.

            Returns
            -------
            events : torch.Tensor of shape=(n_samples,)
                Events in data.

            context : torch.Tensor of shape=(n_samples, context_length)
                Context events for each event in events.

            labels : torch.Tensor of shape=(n_samples,)
                Labels will be None if no labels parameter is given, and if data
                does not contain any 'labels' column.
            """
        raise NotImplementedError("Parsing '.json' not yet implemented.")


    def ndjson(self, path, labels=None, verbose=False, mapping=None):
        """Preprocess data from ndjson file.

            Note
            ----
            ndjson preprocessing will become available in a future version.

            Parameters
            ----------
            path : string
                Path to input file from which to read data.

            labels : int or array-like of shape=(n_samples,), optional
                If a int is given, label all sequences with given int. If an
                array-like is given, use the given labels for the data in file.
                Note: will overwrite any 'label' data in input file.

            verbose : boolean, default=False
                If True, prints progress in transforming input to sequences.

            mapping : dict(), default=None, optional
                If mapping is provided, use given mapping. Any additional unique values
                will be appended to the given mapping. NO_EVENT may be present in the 
                provided mapping, but it is not required.

            Returns
            -------
            events : torch.Tensor of shape=(n_samples,)
                Events in data.

            context : torch.Tensor of shape=(n_samples, context_length)
                Context events for each event in events.

            labels : torch.Tensor of shape=(n_samples,)
                Labels will be None if no labels parameter is given, and if data
                does not contain any 'labels' column.
            """
        raise NotImplementedError("Parsing '.ndjson' not yet implemented.")


    def text(self, path, nrows=None, labels=None, verbose=False, mapping=None):
        """Preprocess data from text file.

            Note
            ----
            **Format**: The assumed format of a text file is that each line in
            the text file contains a space-separated sequence of event IDs for a
            machine. I.e. for *n* machines, there will be *n* lines in the file.

            Parameters
            ----------
            path : string
                Path to input file from which to read data.

            nrows : int, default=None
                If given, limit the number of rows to read to nrows.

            labels : int or array-like of shape=(n_samples,), optional
                If a int is given, label all sequences with given int. If an
                array-like is given, use the given labels for the data in file.
                Note: will overwrite any 'label' data in input file.

            verbose : boolean, default=False
                If True, prints progress in transforming input to sequences.

            mapping : dict(), default=None, optional
                If mapping is provided, use given mapping. Any additional unique values
                will be appended to the given mapping. NO_EVENT may be present in the 
                provided mapping, but it is not required.

            Returns
            -------
            events : torch.Tensor of shape=(n_samples,)
                Events in data.

            context : torch.Tensor of shape=(n_samples, context_length)
                Context events for each event in events.

            labels : torch.Tensor of shape=(n_samples,)
                Labels will be None if no labels parameter is given, and if data
                does not contain any 'labels' column.
            """
        # Initialise data
        events     = list()
        machines   = list()

        # Open text file
        with open(path) as infile:

            # Loop over each line, i.e. machine
            for machine, line in enumerate(infile):

                # Break if machine >= nrows
                if nrows is not None and machine >= nrows: break

                # Extract events for each machine
                for event in map(int, line.split()):

                    # Add data
                    events  .append(event)
                    machines.append(machine)

        # Transform to pandas DataFrame
        data = pd.DataFrame({
            'timestamp': np.arange(len(events)), # Increasing order
            'event'    : events,
            'machine'  : machines,
        })

        # Transform to sequences and return
        return self.sequence(data, labels=labels, verbose=verbose, mapping=mapping)


if __name__ == "__main__":
    ########################################################################
    #                               Imports                                #
    ########################################################################

    import argformat
    import argparse
    import os

    ########################################################################
    #                           Parse arguments                            #
    ########################################################################

    # Create Argument parser
    parser = argparse.ArgumentParser(
        description     = "Preprocessor: processes data from standard formats into DeepCASE sequences.",
        formatter_class = argformat.StructuredFormatter
    )

    # Add arguments
    parser.add_argument('file',                                  help='file      to preprocess')
    parser.add_argument('--write',                               help='file      to write output')
    parser.add_argument('--type',              default='auto'  , help="file type to preprocess (auto|csv|json|ndjson|t(e)xt)")
    parser.add_argument('--context', type=int, default=10      , help="size of context")
    parser.add_argument('--timeout', type=int, default=60*60*24, help="maximum time between context and event")

    # Parse arguments
    args = parser.parse_args()

    ########################################################################
    #                              Parse type                              #
    ########################################################################

    # Allowed extensions
    ALLOWED_EXTENSIONS = {'csv', 'json', 'ndjson', 'txt', 'text'}

    # Infer type
    if args.type == 'auto':
        # Get file by extension
        args.type = os.path.splitext(args.file)[1][1:]
        # Check if recovered extension is allowed
        if args.type not in ALLOWED_EXTENSIONS:
            raise ValueError(
                "Automatically parsed extension not supported: '.{}'. "
                "Please manually specify --type (csv|json|ndjson|t(e)xt)"
                .format(args.type)
            )

    ########################################################################
    #                              Preprocess                              #
    ########################################################################

    # Create preprocessor
    preprocessor = Preprocessor(
        context = args.context,
        timeout = args.timeout,
    )

    # Preprocess file
    if args.type == 'csv':
        events, context, labels = preprocessor.csv(args.file)
    elif args.type == 'json':
        events, context, labels = preprocessor.json(args.file)
    elif args.type == 'ndjson':
        events, context, labels = preprocessor.ndjson(args.file)
    elif args.type == 'txt' or args.type == 'text':
        events, context, labels = preprocessor.text(args.file)
    else:
        raise ValueError("Unsupported file type: '{}'".format(args.type))

    ########################################################################
    #                             Write output                             #
    ########################################################################

    # Write output if necessary
    if args.write:

        # Open output file
        with open(args.write, 'wb') as outfile:
            # Write output
            torch.save({
                'events' : events,
                'context': context,
                'labels' : labels,
            }, outfile)

        ####################################################################
        #                           Load output                            #
        ####################################################################

        # Open output file
        with open(args.write, 'rb') as infile:
            # Load output
            data = torch.load(infile)
            # Load variables
            events  = data.get('events')
            context = data.get('context')
            labels  = data.get('labels')

    ########################################################################
    #                             Show output                              #
    ########################################################################

    print("Events : {}".format(events))
    print("Context: {}".format(context))
    print("Labels : {}".format(labels))
