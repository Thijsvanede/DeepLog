Command line tool
=================
When DeepLog is installed, it can be used from the command line.
The :code:`__main__.py` file in the :code:`deeplog` module implements this command line tool.
The command line tool provides a quick and easy interface to predict sequences from :code:`.csv` files.
The full command line usage is given in its :code:`help` page:

.. code:: text

  usage: deeplog.py [-h] [--csv CSV] [--txt TXT] [--length LENGTH] [--timeout TIMEOUT] [--hidden HIDDEN]
                    [-i INPUT] [-l LAYERS] [-k TOP] [--save SAVE] [--load LOAD] [-b BATCH_SIZE]
                    [-d DEVICE] [-e EPOCHS]
                    {train,predict}

  Deeplog: Anomaly detection and diagnosis from system logs through deep learning

  positional arguments:
    {train,predict}              mode in which to run DeepLog

  optional arguments:
    -h, --help                   show this help message and exit

  Input parameters:
    --csv       CSV              CSV events file to process
    --txt       TXT              TXT events file to process
    --length    LENGTH           sequence LENGTH                          (default =   20)
    --timeout   TIMEOUT          sequence TIMEOUT (seconds)               (default =  inf)

  DeepLog parameters:
    --hidden    HIDDEN           hidden dimension                         (default =   64)
    -i, --input INPUT            input  dimension                         (default =  300)
    -l, --layers LAYERS          number of lstm layers to use             (default =    2)
    -k, --top   TOP              accept any of the TOP predictions        (default =    1)
    --save      SAVE             save DeepLog to   specified file
    --load      LOAD             load DeepLog from specified file

  Training parameters:
    -b, --batch-size BATCH_SIZE  batch size                               (default =  128)
    -d, --device DEVICE          train using given device (cpu|cuda|auto) (default = auto)
    -e, --epochs EPOCHS          number of epochs to train with           (default =   10)

Examples
^^^^^^^^
Use first half of ``<data.csv>`` to train DeepLog and use second half of ``<data.csv>`` to predict and test the prediction.

.. code::

  python3 -m deeplog train   --csv <data.csv> --save deeplog.save # Training
  python3 -m deeplog predict --csv <data.csv> --load deeplog.save # Predicting
