# Import pytorch
import torch

# Import DeepLog and Preprocessor
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

# Load data from HDFS dataset
X_train       , y_train       , label_train       , mapping_train        = preprocessor.text("./data/hdfs_train"        , verbose=True)
X_test        , y_test        , label_test        , mapping_test         = preprocessor.text("./data/hdfs_test_normal"  , verbose=True)
X_test_anomaly, y_test_anomaly, label_test_anomaly, mapping_test_anomaly = preprocessor.text("./data/hdfs_test_abnormal", verbose=True)

##############################################################################
#                                  DeepLog                                   #
##############################################################################

# Create DeepLog object
deeplog = DeepLog(
    input_size  = 300, # Number of different events to expect
    hidden_size = 64 , # Hidden dimension, we suggest 64
    output_size = 300, # Number of different events to expect
)

# Optionally cast data and DeepLog to cuda, if available
if torch.cuda.is_available():
    # Set deeplog to device
    deeplog = deeplog.to("cuda")

    # Set data to device
    X_train        = X_train       .to("cuda")
    y_train        = y_train       .to("cuda")
    X_test         = X_test        .to("cuda")
    y_test         = y_test        .to("cuda")
    X_test_anomaly = X_test_anomaly.to("cuda")
    y_test_anomaly = y_test_anomaly.to("cuda")

# Train deeplog
deeplog.fit(
    X          = X_train,
    y          = y_train,
    epochs     = 100,
    batch_size = 128,
)

# Predict normal data using deeplog
y_pred_normal, confidence = deeplog.predict(
    X = X_test,
    k = 3, # Change this value to get the top k predictions (called 'g' in DeepLog paper, see Figure 6)
)

# Predict anomalous data using deeplog
y_pred_anomaly, confidence = deeplog.predict(
    X = X_test_anomaly,
    k = 3, # Change this value to get the top k predictions (called 'g' in DeepLog paper, see Figure 6)
)

################################################################################
#                             Check for anomalies                              #
################################################################################

# Check if the actual value matches any of the predictions
# If any prediction matches, it is not an anomaly, so to get the anomalies, we
# invert our answer using ~

# Check for anomalies in normal data (ideally, we should not find any)
anomalies_normal = ~torch.any(
    y_test == y_pred_normal.T,
    dim = 0,
)

# Check for anomalies in abnormal data (ideally, we should not find all)
anomalies_abnormal = ~torch.any(
    y_test_anomaly == y_pred_anomaly.T,
    dim = 0,
)

# Print the fraction of anomalies in normal data (False positives)
print(f"False positives: {anomalies_normal.sum() / anomalies_normal.shape[0]}")

# Print the fraction of anomalies in abnormal data (True positives)
print(f"True  positives: {anomalies_abnormal.sum() / anomalies_abnormal.shape[0]}")
