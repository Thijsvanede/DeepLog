Overview
========
This section explains the design of DeepLog on a high level.
DeepLog is a network that is implemented as a `torch-train`_ :py:class:`Module`, which is an extension of :py:class:`torch.nn.Module` including automatic methods to :py:meth:`fit` and :py:meth:`predict` data.
This means it can be trained and used as any neural network module in the pytorch library.

In addition, we provide automatic methods to train and predict events given previous event sequences using the `torch-train`_ library.
This follows a ``scikit-learn`` approach with :py:meth:`fit`, :py:meth:`predict` and :py:meth:`fit_predict` methods.
We refer to its documentation for a detailed description.

.. _`torch-train`: https://github.com/anonymized/torch-train
