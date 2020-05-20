import argparse
import torch
from   torch.utils.data import DataLoader, TensorDataset

from deeplog      import DeepLog
from labelencoder import LabelEncoder
from reader       import Reader
from utils        import TextHelpFormatter

if __name__ == "__main__":
    ########################################################################
    #                           Parse arguments                            #
    ########################################################################

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="DeepLog: anomaly detection using deep learning.",
        formatter_class=TextHelpFormatter
    )

    # Add arguments
    parser.add_argument('test' , type=str, nargs='+', help="file(s) from which to read test data.")

    # Add DeepLog arguments
    parse_group_deeplog = parser.add_argument_group("Optional DeepLog parameters")
    parse_group_deeplog.add_argument('-b', '--batch-size', type=int, default=128, help="batch size for training")
    parse_group_deeplog.add_argument('-d', '--hidden-dim', type=int, default= 64, help="dimension of hidden layer")
    parse_group_deeplog.add_argument('-e', '--epochs'    , type=int, default= 10, help="number of epochs to train with")
    parse_group_deeplog.add_argument('-l', '--layers'    , type=int, default=  2, help="number of layers used by DeepLog")
    parse_group_deeplog.add_argument('-o', '--output'    , type=int, default=  0, help="required output shape")
    parse_group_deeplog.add_argument('-w', '--window'    , type=int, default= 10, help="window size used by DeepLog")

    # Add input/output arguments
    parse_group_io = parser.add_argument_group("Optional I/O parameters")
    parse_input    = parse_group_io.add_mutually_exclusive_group(required=True)
    parse_input   .add_argument('--train', type=str, nargs='+', help="path to train data")
    parse_input   .add_argument('--load' , type=str,            help="path to load trained model")
    parse_group_io.add_argument('--save' , type=str,            help="path to save trained model")

    # Parse given arguments
    args = parser.parse_args()

    ########################################################################
    #                              Load data                               #
    ########################################################################

    # Test data
    X_test, y_test = list(), list()
    for i, test in enumerate(args.test):
        X_test_ , y_test_ = Reader().read(test, args.window)
        X_test.append(X_test_)
        y_test.append(y_test_)
    X_test = torch.cat(X_test)
    y_test = torch.cat(y_test)
    # Store labels
    labels = y_test

    # Train from data
    if args.train:
        # Read train data from given files
        X_train, y_train = list(), list()
        for train in args.train:
            X_train_ , y_train_ = Reader().read(train, args.window)
            X_train.append(X_train_)
            y_train.append(y_train_)
        X_train = torch.cat(X_train)
        y_train = torch.cat(y_train)
        # Get all labels
        labels = torch.cat((y_train, labels))

    # Encode labels
    le = LabelEncoder()
    le = le.fit(labels)
    from time import time
    start = time()
    y_test  = torch.tensor(le.transform(y_test))

    if args.train:
        y_train = torch.tensor(le.transform(y_train))
        # Get data as dataset
        data = DataLoader(
            TensorDataset(X_train, y_train),
            args.batch_size,
            shuffle=False,
            pin_memory=True
        )

    ########################################################################
    #                            Create DeepLog                            #
    ########################################################################

    # Compute required output shape
    output_size = args.output or labels.unique().shape[0]
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Hyperparameters - TODO remove
    input_size = 1
    # Create DeepLog instance
    deeplog = DeepLog(input_size, args.hidden_dim, output_size, args.layers, args.window).to(device)

    # Train from data
    if args.train:
        deeplog.train(data, epochs=args.epochs, verbose=True)
    # Load pretrained model
    elif args.load:
        deeplog = deeplog.load(args.load)
    # Save model if necessary
    if args.save:
        deeplog.save(args.save)

    ########################################################################
    #                           Predict DeepLog                            #
    ########################################################################

    deeplog.predict(X_test, y_test, 5)
