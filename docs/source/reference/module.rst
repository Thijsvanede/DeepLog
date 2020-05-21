.. _Module:

Module
======

The Module class is an extension of the `torch.nn.Module`_ object.
This class implements scikit-learn-like :py:meth:`fit` and :py:meth:`predict` methods to automatically use :py:class:`nn.Module` objects for training and predicting labels.
This module also automatically keeps track of the progress during fitting and predicting.

.. _`torch.nn.Module`: https://pytorch.org/docs/stable/nn.html#module

.. autoclass:: module.Module

Initialization
^^^^^^^^^^^^^^

.. automethod:: module.Module.__init__

Fit
^^^

To train a Module, we use the :py:meth:`fit` method.

.. automethod:: module.Module.fit

Predict
^^^^^^^

A module can also :py:meth:`predict` outputs for given inputs.
Usually this method is called after fitting.

.. automethod:: module.Module.predict

Fit-Predict
^^^^^^^^^^^

Sometimes we want to :py:meth:`fit` and :py:meth:`predict` on the same data.
This can easily be achieved using the :py:meth:`fit_predict` method.

.. automethod:: module.Module.fit_predict
