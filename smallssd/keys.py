from re import L


class RawLabelKeys:
    IMAGE_NAME = "image_name"
    HEIGHT = "image_height"
    WIDTH = "image_width"
    LABELS = "labels"


class RawBoxKeys:
    ID = "id"
    X1 = "x1"
    Y1 = "y1"
    X2 = "x2"
    Y2 = "y2"
    CLASSNAME = "class_name"


class LabelKeys:
    BOXES = "boxes"
    LABELS = "labels"


def key_attributes(key_class):
    return [key for key, _ in key_class.__dict__.items() if not key.startswith("__")]


CLASSNAME_TO_IDX = {"wheat": 1, "weed": 2}
