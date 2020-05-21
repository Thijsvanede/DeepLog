.. _DeepLog:

DeepLog
=======

The DeepLog class is an extension of the :ref:`Module` object.
This class implements the neural network as described in the paper `Deeplog: Anomaly detection and diagnosis from system logs through deep learning`_.

.. _`Deeplog: Anomaly detection and diagnosis from system logs through deep learning`: https://doi.org/10.1145/3133956.3134015


.. autoclass:: deeplog.DeepLog

Initialization
^^^^^^^^^^^^^^

.. automethod:: deeplog.DeepLog.__init__

Forward
^^^^^^^

As DeepLog is a Neural Network, it implements the :py:meth:`forward` method which passes input through the entire network.

.. automethod:: deeplog.DeepLog.forward

Predict
^^^^^^^

The regular network gives a probability distribution over all possible output values.
However, DeepLog outputs the `k` most likely outputs, therefore it overwrites the :py:meth:`predict` method of the :ref:`Module` class.

.. automethod:: deeplog.DeepLog.predict
