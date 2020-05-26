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

In this example, we import all different LSTM implementations and use it to predict the next item in a sequence.
First we import the necessary torch modules and different LSTMs that we want to use.

.. code:: python

  # import Tiresias and PreprocessLoader
  from deeplog import DeepLog
  from deeplog.preprocessing import PreprocessLoader
  import torch.nn as nn

  ##############################################################################
  #                                 Load data                                  #
  ##############################################################################
  # Create loader for preprocessed data
  loader = PreprocessLoader()
  # Load data
  data, encodings = loader.load(
      <infile>,
      dim_in      = 20,
      dim_out     = 1,
      train_ratio = 0.5,
      key         = lambda x: (x.get(<groupby_key>),),
      extract     = [<event_key>],
      random      = False
  )

  # Get short handles
  device = 'cpu' # Optionally set to 'cuda'
  X_train = data.get(<event_key>).get('train').get('X').to(device)
  y_train = data.get(<event_key>).get('train').get('y').to(device).reshape(-1)
  X_test  = data.get(<event_key>).get('test' ).get('X').to(device)
  y_test  = data.get(<event_key>).get('test' ).get('y').to(device).reshape(-1)

  ##############################################################################
  #                                  Tiresias                                  #
  ##############################################################################
  deeplog = DeepLog(<dim_input>, <dim_hidden>, <dim_output>).to(device)
  # Train deeplog
  deeplog.fit(X_train, y_train, epochs=10, batch_size=128, criterion=nn.CrossEntropyLoss())
  # Predict using deeplog
  y_pred, confidence = deeplog.predict(X_test, k=5)
