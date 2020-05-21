.. _Progress:

Progress
========

The :py:class:`Progress` class is used to track the progress of training and prediction.

.. autoclass:: module.Progress

Initialization
^^^^^^^^^^^^^^

.. automethod:: module.Progress.__init__

Reset
^^^^^

To restart the Progress, we use the :py:meth:`reset` method.
This sets the amount of items we expect to train with and the number of epochs we use for training.

.. automethod:: module.Progress.reset

Update
^^^^^^

A module will update using the :py:meth:`update` method, which automatically prints the progress.

.. automethod:: module.Progress.update

Second, when we move to the next epoch we use the :py:meth:`update_epoch` method.

.. automethod:: module.Progress.update_epoch
