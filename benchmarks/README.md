# Benchmark models

Benchmark models to accompany the `smallSSD` dataset.

To run this (assuming you have the `smallSSD` package installed):
1. Install the additional requirements in the [`requirements.txt`](requirements.txt) file
2. Run [`main.py`](main.py)

Trained models will be stored (by default) in `benchmarks/lightning_logs`.
A [test script](test.py) is additionally provided to test models.

In addition, a [pseudo labelling script](pseudo_labels.py) is provided.
In this script, a trained model labels 2000 of the unlabelled examples, and is further trained on the pseudo-labelled examples (concatenated to the labelled examples). To run the psuedo labelling pipeline, first train a fully-supervised model using `main.py`. Then, use that fully trained model to label some instances and re-train that same model:

```bash
python pseudo_labels.py --version <PATH_TO_FULLY_TRAINED_MODEL> --model <FULLY_TRAINED_MODEL_TYPE>
```

Alternatively, the logic in both scripts is combined in the [end to end](end_to_end.py) script.

PyTorch torchvision detection models should be drop in replaceable to this pipeline; we currently train
Faster R-CNN, Retinanet, YOLO and SSD models.
