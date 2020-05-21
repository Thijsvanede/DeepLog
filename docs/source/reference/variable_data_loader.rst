.. _Variable_Data_Loader:

Variable Data Loader
====================

The Variable Data Loader class is a different implementation of the iterable `torch.utils.data.DataLoader`_ class that can handle inputs of variable lengths.

.. _`torch.utils.data.DataLoader`: https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader


.. autoclass:: variable_data_loader.VariableDataLoader

Initialization
^^^^^^^^^^^^^^

.. automethod:: variable_data_loader.VariableDataLoader.__init__

Iterable
^^^^^^^^

The VariableDataLoader is an iterable object that iterates through the entire dataset.
The same object can be called multiple times due to the reset method, which automatically resets the iterable after a complete iteration.
Note that it can also be set manually using the :py:meth:`reset` method.

.. automethod:: variable_data_loader.VariableDataLoader.reset
