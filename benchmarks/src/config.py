from smallssd.keys import CLASSNAME_TO_IDX

# + 1 since the 0 index is for the background
NUM_OUTPUT_CLASSES = len(CLASSNAME_TO_IDX) + 1
BATCH_SIZE = 2

MAX_PSUEDO_LABELLED_IMAGES = 2000

TEST_MAP_KWARGS = {"max_detection_thresholds": [100]}
