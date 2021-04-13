import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deeplog",
    version="0.0.1",
    author="Anonymized",
    author_email="Anonymized",
    description="Pytorch implementation of Deeplog: Anomaly detection and diagnosis from system logs through deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/anonymized/DeepLog",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
