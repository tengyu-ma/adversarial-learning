import tensorflow as tf

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

# EVAL_DATA = 'test'  # 'train_eval'
EVAL_DIR = '/tmp/cifar10_eval'
DATA_DIR = '/tmp/cifar10_data/cifar-10-batches-bin'
CHECKPOINT_DIR = '/tmp/cifar10_train'
NUM_EXAMPLES = 10000

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
NOISE_OUTPUT = False
tf.app.flags.DEFINE_integer('batch_size', 128, """Number of images to process in a batch.""")
# tf.app.flags.DEFINE_string('data_dir', '/tmp/cifar10_data', """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False, """Train the model using fp16.""")

EVAL_DATA = 'test_batch.bin'
ORG_IMAGE_SIZE = 32
EPS = 20
