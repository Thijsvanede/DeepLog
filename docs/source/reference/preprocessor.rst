.. _PreprocessLoader:

PreprocessLoader
================

The PreprocessLoader loads preprocessed data stored as (encoded) ``.csv`` files.

.. autoclass:: preprocessing.PreprocessLoader

Initialization
^^^^^^^^^^^^^^

.. automethod:: preprocessing.PreprocessLoader.__init__


Load data
^^^^^^^^^

In order to load preprocessed data, we use the :py:meth:`load` method.
This method assumes the data is stored as a ``.csv`` file.
It groups inputs using the ``key`` function and extracts all fields specified by the ``extract`` parameter.

.. automethod:: preprocessing.PreprocessLoader.load
