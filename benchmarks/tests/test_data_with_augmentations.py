import pytest

from src.data_with_augmentations import AugmentedDataset, train_val_augmentations
from smallssd.config import DATAFOLDER_PATH


@pytest.mark.integration
def test_data_with_augmentations():
    train_ds, val_ds = AugmentedDataset.split(
        root=DATAFOLDER_PATH, augmentations=train_val_augmentations(), eval=False
    )

    for ds in (train_ds, val_ds):
        for batch in ds:
            im, t = batch
            assert len(im.unique()) > 1
            for box in t["boxes"]:
                x1, y1, x2, y2 = box
                assert x2 > x1
                assert y2 > y1
