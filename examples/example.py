# import DeepLog and Preprocessor
from deeplog              import DeepLog
from deeplog.preprocessor import Preprocessor

##############################################################################
#                                 Load data                                  #
##############################################################################

# Create preprocessor for loading data
preprocessor = Preprocessor(
    length  = 20,           # Extract sequences of 20 items
    timeout = float('inf'), # Do not include a maximum allowed time between events
)

# Load data from csv file
X, y, label, mapping = preprocessor.csv("<path/to/file.csv>")
# Load data from txt file
X, y, label, mapping = preprocessor.txt("<path/to/file.txt>")

##############################################################################
#                                  DeepLog                                  #
##############################################################################

# Create DeepLog object
deeplog = DeepLog(
    input_size  = 300, # Number of different events to expect
    hidden_size = 64 , # Hidden dimension, we suggest 64
    output_size = 300, # Number of different events to expect
)

# Optionally cast data and DeepLog to cuda, if available
deeplog = deeplog.to("cuda")
X       = X      .to("cuda")
y       = y      .to("cuda")

# Train deeplog
deeplog.fit(
    X          = X,
    y          = y,
    epochs     = 10,
    batch_size = 128,
)

# Predict using deeplog
y_pred, confidence = deeplog.predict(
    X = X,
    y = y,
    k = 3,
)
