Command line tool
=================
When DeepLog is installed, it can be used from the command line.
The :code:`__main__.py` file in the :code:`deeplog` module implements this command line tool.
The command line tool provides a quick and easy interface to predict sequences from :code:`.csv` files.
The full command line usage is given in its :code:`help` page:

.. Note::

  Note that when handling very large inputs, DeepLog can be slow.
  In order to more quickly test on smaller inputs we provide the ``--max`` flag, which specifies the maximum amount of samples to read from the input file.
  E.g., to use only the first 100k samples, one may invoke DeepLog using ``--max 1e5`` flag.

.. code:: text

  usage: deeplog.py [-h] [-f FIELD] [-m MAX] [-w WINDOW] [--hidden HIDDEN] [-i INPUT] [-l LAYERS] [-k TOP]
                  [-b BATCH_SIZE] [-d DEVICE] [-e EPOCHS] [-r] [--ratio RATIO]
                  file

  DeepLog: anomaly detection using deep learning.

  optional arguments:
  -h, --help                   show this help message and exit

  Input parameters:
  file                         file to read as input
  -f, --field      FIELD       FIELD to extract from input FILE           (default = threat_name)
  -m, --max        MAX         maximum number of items to read from input (default =         inf)
  -w, --window     WINDOW      length of input sequence                   (default =          10)

  Tiresias parameters:
  --hidden         HIDDEN      hidden dimension                           (default =          64)
  -i, --input      INPUT       input  dimension                           (default =         300)
  -l, --layers     LAYERS      number of lstm layers to use               (default =           2)
  -k, --top        TOP         accept any of the TOP predictions          (default =           1)

  Training parameters:
  -b, --batch-size BATCH_SIZE  batch size                                 (default =         128)
  -d, --device     DEVICE      train using given device (cpu|cuda|auto)   (default =        auto)
  -e, --epochs     EPOCHS      number of epochs to train with             (default =          10)
  -r, --random                 train with random selection
  --ratio          RATIO       proportion of data to use for training     (default =         0.5)

Examples
^^^^^^^^
Use first half of ``<data.csv>`` to train DeepLog and use second half of ``<data.csv>`` to predict and test the prediction.

.. code::

  python3 -m deeplog <data.csv> --ratio 0.5
