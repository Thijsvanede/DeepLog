# Imports
import argformat
import argparse
import torch
import torch.nn as nn
import warnings

# DeepLog imports
from deeplog              import DeepLog
from deeplog.preprocessor import Preprocessor

if __name__ == "__main__":
    ########################################################################
    #                           Parse arguments                            #
    ########################################################################

    # Parse arguments
    parser = argparse.ArgumentParser(
        prog            = "deeplog.py",
        description     = "Deeplog: Anomaly detection and diagnosis from system logs through deep learning",
        formatter_class = argformat.StructuredFormatter,
    )

    # Add DeepLog mode arguments, run in different modes
    parser.add_argument('mode', help="mode in which to run DeepLog", choices=(
        'train',
        'predict',
    ))

    # Add arguments
    group_input = parser.add_argument_group("Input parameters")
    group_input.add_argument('--csv'      , help="CSV events file to process")
    group_input.add_argument('--txt'      , help="TXT events file to process")
    group_input.add_argument('--length'   , type=int  , default=20          , help="sequence LENGTH           ")
    group_input.add_argument('--timeout'  , type=float, default=float('inf'), help="sequence TIMEOUT (seconds)")

    # Deeplog parameters
    group_deeplog = parser.add_argument_group("DeepLog parameters")
    group_deeplog.add_argument(      '--hidden', type=int, default=64 , help='hidden dimension')
    group_deeplog.add_argument('-i', '--input' , type=int, default=300, help='input  dimension')
    group_deeplog.add_argument('-l', '--layers', type=int, default=2  , help='number of lstm layers to use')
    group_deeplog.add_argument('-k', '--top'   , type=int, default=1  , help='accept any of the TOP predictions')
    group_deeplog.add_argument('--save', help="save DeepLog to   specified file")
    group_deeplog.add_argument('--load', help="load DeepLog from specified file")

    # Training
    group_training = parser.add_argument_group("Training parameters")
    group_training.add_argument('-b', '--batch-size', type=int, default=128,   help="batch size")
    group_training.add_argument('-d', '--device'    , default='auto'     ,     help="train using given device (cpu|cuda|auto)")
    group_training.add_argument('-e', '--epochs'    , type=int, default=10,    help="number of epochs to train with")

    # Parse given arguments
    args = parser.parse_args()

    ########################################################################
    #                              Load data                               #
    ########################################################################

    # Set device
    if args.device is None or args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create preprocessor
    preprocessor = Preprocessor(
        length  = args.length,
        timeout = args.timeout,
    )

    # Load files
    if args.csv is not None and args.txt is not None:
        # Raise an error if both csv and txt are specified
        raise ValueError("Please specify EITHER --csv OR --txt.")
    if args.csv:
        # Load csv file
        X, y, label, mapping = preprocessor.csv(args.csv)
    elif args.txt:
        # Load txt file
        X, y, label, mapping = preprocessor.txt(args.txt)

    X = X.to(args.device)
    y = y.to(args.device)

    ########################################################################
    #                            Create DeepLog                            #
    ########################################################################

    # Load DeepLog from file, if necessary
    if args.load:
        deeplog = DeepLog.load(args.load).to(args.device)

    # Otherwise create new DeepLog instance
    else:
        deeplog = DeepLog(
            input_size  = args.input,
            hidden_size = args.hidden,
            output_size = args.input,
            num_layers  = args.layers,
        ).to(args.device)

    # Train DeepLog
    if args.mode == "train":

        # Print warning if training DeepLog without saving it
        if args.save is None:
            warnings.warn("Training DeepLog without saving it to output.")

        # Train DeepLog
        deeplog.fit(
            X             = X,
            y             = y,
            epochs        = args.epochs,
            batch_size    = args.batch_size,
            criterion     = nn.CrossEntropyLoss(),
        )

        # Save DeepLog to file
        if args.save:
            deeplog.save(args.save)

    # Predict with DeepLog
    if args.mode == "predict":

        # Predict using DeepLog
        y_pred, confidence = deeplog.predict(
            X = X,
            k = args.top,
        )

        ####################################################################
        #                         Predict DeepLog                          #
        ####################################################################

        # Initialise predictions
        y_pred_top = y_pred[:, 0]
        # Compute top TOP predictions
        for top in range(1, args.top):
            # Get mask
            mask = y == y_pred[:, top]
            # Set top values
            y_pred_top[mask] = y[mask]

        from sklearn.metrics import classification_report
        print(classification_report(y.cpu(), y_pred_top.cpu(), digits=4))
