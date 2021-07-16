.. _Preprocessor:

Preprocessor
============
The Preprocessor class provides methods to automatically extract event sequences from various common data formats.
To start sequencing, first create the Preprocessor object.

.. autoclass:: preprocessor.Preprocessor

.. automethod:: preprocessor.Preprocessor.__init__

Formats
^^^^^^^
We currently support the following formats:
 * ``.csv`` files containing a header row that specifies the columns 'timestamp', 'event' and 'machine'.
 * ``.txt`` files containing a line for each machine and a sequence of events (integers) separated by spaces.

Transforming ``.csv`` files into sequences is the quickest method and is done by the following method call:

.. automethod:: preprocessor.Preprocessor.csv

Transforming ``.txt`` files into sequences is slower, but still possible using the following method call:

.. automethod:: preprocessor.Preprocessor.text

Future supported formats
------------------------

.. note::

   These formats already have an API entrance, but are currently **NOT** supported.

* ``.json`` files containing values for 'timestamp', 'event' and 'machine'.
* ``.ndjson`` where each line contains a json file with keys 'timestamp', 'event' and 'machine'.

.. automethod:: preprocessor.Preprocessor.json

.. automethod:: preprocessor.Preprocessor.ndjson
