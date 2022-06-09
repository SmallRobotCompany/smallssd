import torch
from pathlib import Path

from smallssd import data
from smallssd.keys import LabelKeys


def test_labelled_dataset():
    d = data.LabelledData(Path(__file__).parent / "test_data", eval=False)
    assert len(d) == 1

    x, y = d[0]

    assert isinstance(x, torch.Tensor)

    assert len(y) == 2
    assert y[LabelKeys.BOXES].shape[1] == 4
    assert y[LabelKeys.BOXES].shape[0] == y[LabelKeys.LABELS].shape[0]


def test_unlabelled_data():
    d = data.UnlabelledData(Path(__file__).parent / "test_data")
    assert len(d) == 1
    assert isinstance(d[0], torch.Tensor)


def test_data_downloads_and_unpacks_correctly(tmp_path):
    d = data.LabelledData(root=tmp_path, eval=True, download=True)
    assert len(d) == 156
