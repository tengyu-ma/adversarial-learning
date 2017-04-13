# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 24
ORG_IMAGE_SIZE = 32

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

EVAL_DIR = '/tmp/cifar10_eval'
EVAL_DATA = 'test'  # 'train_eval'
CHECKPOINT_DIR = '/tmp/cifar10_train'
NUM_EXAMPLES = 10000
EPS = 50
NOISE_OUTPUT = True