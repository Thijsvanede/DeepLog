Installation
============
The most straigtforward way of installing DeepLog is via pip

.. note::

  Installation via pip is currently only supported from the command line ``pip install -e <path/to/directory/containing/setup.py>``

.. code::

  pip install deeplog

If you wish to stay up to date with the latest development version, you can instead download the `source code`_.
In this case, make sure that you have all the required `dependencies`_ installed.

.. _source code: https://github.com/anonymized/DeepLog

.. _dependencies:

Dependencies
^^^^^^^^^^^^
DeepLog requires the following python packages to be installed:

- argformat: https://github.com/anonymized/argformat
- numpy: https://numpy.org/
- scikit-learn: https://scikit-learn.org/
- pytorch: https://pytorch.org/

All dependencies should be automatically downloaded if you install DeepLog via pip. However, should you want to install these libraries manually, you can install the dependencies using the requirements.txt file

.. code::

  pip install -r requirements.txt

Or you can install these libraries yourself

.. code::

  pip install -U argformat numpy scikit-learn torch
