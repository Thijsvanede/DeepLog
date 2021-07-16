Code
====
To use DeepLog into your own project, you can use it as a standalone module.
Here we show some simple examples on how to use the DeepLog package in your own python code.
For a complete documentation we refer to the :ref:`Reference` guide.

Import
^^^^^^
To import components from DeepLog simply use the following format

.. code:: python

  from deeplog          import <Object>
  from deeplog.<module> import <Object>

For example, the following code imports the DeepLog neural network as found in the :ref:`Reference`.

.. code:: python

  # Imports
  from deeplog import DeepLog

Working example
^^^^^^^^^^^^^^^

In this example, we load data from either a ``.csv`` or ``.txt`` file and use that data to train and predict with DeepLog.

.. code:: python

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
  y, X, label, mapping = preprocessor.csv("<path/to/file.csv>")
  # Load data from txt file
  y, X, label, mapping = preprocessor.txt("<path/to/file.txt>")

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
