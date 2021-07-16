.. _DeepLog:

DeepLog
=======

The DeepLog class uses the `torch-train`_ library for training and prediction.
This class implements the neural network as described in the paper `Deeplog: Anomaly detection and diagnosis from system logs through deep learning`_.

.. _`Deeplog: Anomaly detection and diagnosis from system logs through deep learning`: https://doi.org/10.1145/3133956.3134015
.. _`torch-train`: https://github.com/Thijsvanede/torch-train


.. autoclass:: deeplog.DeepLog

Initialization
^^^^^^^^^^^^^^

.. automethod:: deeplog.DeepLog.__init__

Forward
^^^^^^^

As DeepLog is a Neural Network, it implements the :py:meth:`forward` method which passes input through the entire network.

.. automethod:: deeplog.DeepLog.forward

Fit
^^^

DeepLog inherits its fit method from the `torch-train`_ module. See the `documentation`_ for a complete reference.

.. automethod:: deeplog.DeepLog.fit

.. _`documentation`: https://torch-train.readthedocs.io/en/latest/reference/module.html#fit

Predict
^^^^^^^

The regular network gives a probability distribution over all possible output values.
However, DeepLog outputs the `k` most likely outputs, therefore it overwrites the :py:meth:`predict` method of the :py:class:`Module` class from `torch-train`_.

.. automethod:: deeplog.DeepLog.predict
