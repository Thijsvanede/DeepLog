import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from   torch.utils.data import DataLoader, TensorDataset

from argformat     import StructuredFormatter
from deeplog       import DeepLog
from labelencoder  import LabelEncoder
from preprocessing import PreprocessLoader
from reader        import Reader

if __name__ == "__main__":
    ########################################################################
    #                           Parse arguments                            #
    ########################################################################

    # Parse arguments
    parser = argparse.ArgumentParser(
        prog            = "deeplog.py",
        description     = "DeepLog: anomaly detection using deep learning.",
        formatter_class = StructuredFormatter
    )

    # Add arguments
    group_input = parser.add_argument_group("Input parameters")
    group_input.add_argument('file', help='file to read as input')
    group_input.add_argument('-f', '--field' , default='threat_name', help='FIELD to extract from input FILE')
    group_input.add_argument('-m', '--max'   , type=float, default=float('inf'), help='maximum number of items to read from input')
    group_input.add_argument('-w', '--window', type=int  , default=10          , help="length of input sequence")

    # Deeplog parameters
    group_deeplog = parser.add_argument_group("Tiresias parameters")
    group_deeplog.add_argument(      '--hidden', type=int, default=64 , help='hidden dimension')
    group_deeplog.add_argument('-i', '--input' , type=int, default=300, help='input  dimension')
    group_deeplog.add_argument('-l', '--layers', type=int, default=2  , help='number of lstm layers to use')
    group_deeplog.add_argument('-k', '--top'   , type=int, default=1  , help='accept any of the TOP predictions')

    # Training
    group_training = parser.add_argument_group("Training parameters")
    group_training.add_argument('-b', '--batch-size', type=int, default=128,   help="batch size")
    group_training.add_argument('-d', '--device'    , default='auto'     ,     help="train using given device (cpu|cuda|auto)")
    group_training.add_argument('-e', '--epochs'    , type=int, default=10,    help="number of epochs to train with")
    group_training.add_argument('-r', '--random'    , action='store_true',     help="train with random selection")
    group_training.add_argument(      '--ratio'     , type=float, default=0.5, help="proportion of data to use for training")

    # Parse given arguments
    args = parser.parse_args()

    ########################################################################
    #                              Load data                               #
    ########################################################################

    # Set device
    if args.device is None or args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    # Create loader for preprocessed data
    loader = PreprocessLoader()
    # Load data
    data, encodings = loader.load(args.file, args.window, 1, args.max,
                            train_ratio=args.ratio,
                            key=lambda x: (x.get('source'), x.get('src_ip')),
                            extract=[args.field],
                            random=args.random)

    # Get short handles
    X_train = data.get(args.field).get('train').get('X').to(device)
    y_train = data.get(args.field).get('train').get('y').to(device).reshape(-1)
    X_test  = data.get(args.field).get('test' ).get('X').to(device)
    y_test  = data.get(args.field).get('test' ).get('y').to(device).reshape(-1)

    # # Test data
    # X_test, y_test= Reader().read('data/hdfs_test_normal', args.window)
    # # Select maximum of test data
    # X_test = X_test[:int(args.max)]
    # y_test = y_test[:int(args.max)]
    #
    # # Read train data from given files
    # X_train, y_train = Reader().read('data/hdfs_train', args.window)
    #
    # # Encode labels
    # le = LabelEncoder()
    # le = le.fit(torch.cat((y_test, y_train)))
    # y_test  = torch.tensor(le.transform(y_test))
    # y_train = torch.tensor(le.transform(y_train))

    ########################################################################
    #                            Create DeepLog                            #
    ########################################################################

    # Create DeepLog instance
    deeplog = DeepLog(args.input, args.hidden, args.input, args.layers).to(device)
    # Train DeepLog
    deeplog.fit(X_train, y_train,
        epochs        = args.epochs,
        batch_size    = args.batch_size,
        learning_rate = 0.01,
        criterion     = nn.CrossEntropyLoss,
        optimizer     = optim.SGD,
        variable      = False,
        verbose       = True,
    )
    # Predict using DeepLog
    y_pred, confidence = deeplog.predict(X_test, y_test, k=args.top)

    ########################################################################
    #                           Predict DeepLog                            #
    ########################################################################

    # Initialise predictions
    y_pred_top = y_pred[:, 0]
    # Compute top TOP predictions
    for top in range(1, args.top):
        # Get mask
        mask = y_test == y_pred[:, top]
        # Set top values
        y_pred_top[mask] = y_test[mask]

    from sklearn.metrics import classification_report
    print(classification_report(y_test.cpu(), y_pred_top.cpu(), digits=4))
