# DeepLog: Anomaly detection and diagnosis from system logs through deep learning
Pytorch implementation of [DeepLog: Anomaly Detection and Diagnosis from System Logs through Deep Learning](https://doi.org/10.1145/3133956.3134015) (CCS'17). This code was implemented as part of the TODO paper. We ask people to [cite](#References) both works when using the software for academic research papers.

## Introduction
Anomaly detection is a critical step towards building a secure and trustworthy system. The primary purpose of a system log is to record system states and significant events at various critical points to help debug system failures and perform root cause analysis. Such log data is universally available in nearly all computer systems. Log data is an important and valuable resource for understanding system status and performance issues; therefore, the various system logs are naturally excellent source of information for online monitoring and anomaly detection. We propose DeepLog, a deep neural network model utilizing Long Short-Term Memory (LSTM), to model a system log as a natural language sequence. This allows DeepLog to automatically learn log patterns from normal execution, and detect anomalies when log patterns deviate from the model trained from log data under normal execution. In addition, we demonstrate how to incrementally update the DeepLog model in an online fashion so that it can adapt to new log patterns over time. Furthermore, DeepLog constructs workflows from the underlying system log so that once an anomaly is detected, users can diagnose the detected anomaly and perform root cause analysis effectively. Extensive experimental evaluations over large log data have shown that DeepLog has outperformed other existing log-based anomaly detection methods based on traditional data mining methodologies.

## Documentation
We provide an extensive documentation including installation instructions and reference at [deeplog.readthedocs.io](https://deeplog.readthedocs.io/en/latest)

Note, currently the readthedocs is not online.
The docs are available from the `/docs` directory. Build them by executing
```
make build
```
from the `/docs` directory. The documentation can then be found under `/docs/build/html/index.html`. **Important: to build the documentation yourself, you will also need to have `sphinx-rdt-theme` and `recommonmark` installed.**

## References
[1] `TODO`

[2] `Du, M., Li, F., Zheng, G., & Srikumar, V. (2017). Deeplog: Anomaly detection and diagnosis from system logs through deep learning. In Proceedings of the 2017 ACM SIGSAC Conference on Computer and Communications Security (CCS) (pp. 1285-1298).`

### Bibtex
```
TODO
```

```
@inproceedings{du2017deeplog,
  title={Deeplog: Anomaly detection and diagnosis from system logs through deep learning},
  author={Du, Min and Li, Feifei and Zheng, Guineng and Srikumar, Vivek},
  booktitle={Proceedings of the 2017 ACM SIGSAC Conference on Computer and Communications Security},
  pages={1285--1298},
  year={2017}
}
```
