import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('data_dir', '/tmp/cifar10_train/cifar-10-batches-bin',
                           """Directory where to get input """)
tf.app.flags.DEFINE_string('eval_dir', '/tmp/cifar10_eval',
                           """Directory for evaluation""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

IS_EVAL_DATA = 'test'  # 'train_eval'
# EVAL_DIR = '/tmp/cifar10_eval'
# DATA_DIR = '/tmp/cifar10_data/cifar-10-batches-bin'
# CHECKPOINT_DIR = '/tmp/cifar10_train'

# MODE = '32_to_24'
# MODE = '24_to_noise'
MODE = 'normal'
# MODE = 'show'
# MODE = 'vote_show'
# MODE = 'train'
EPS = int(0.06 * 255)
NUM_EXAMPLES = 2000
if MODE == 'normal':
    # EVAL_DATA = 'test_batch_15_f2_e2_after_cae.bin'
    EVAL_DATA = 'test_batch_noise_custom_25.bin'
    ORG_IMAGE_SIZE = 24
elif MODE == 'train':
    ORG_IMAGE_SIZE = 32
    EVAL_DATA = 'test_batch.bin'
else:
    EVAL_DATA = 'test_batch_org.bin'
    ORG_IMAGE_SIZE = 24

if MODE == 'normal' or MODE == 'train':
    tf.app.flags.DEFINE_integer('batch_size', 128, """Number of images to process in a batch.""")
else:
    tf.app.flags.DEFINE_integer('batch_size', 1, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/tmp/cifar10_data', """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False, """Train the model using fp16.""")
