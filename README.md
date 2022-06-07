# smallSSD: The **small** robot company's **s**emi-**s**upervised **d**etection dataset

This repository returns a dataset, modelled off the [torchvision](https://pytorch.org/vision/stable/index.html) datasets:

```python
from torch.utils.data import DataLoader
from smallssd.data import LabelledData, UnlabelledData

labelled_loader = DataLoader(LabelledData())
```

This code expects the labelled data to be in the [`data`](data) folder.

### Installation

This package will be uploaded to PyPi. However, the current way to install it is to clone this repository, and to pip install the local repository:

```bash
pip install -e .
```
