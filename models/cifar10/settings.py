import tensorflow as tf

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/cifar10_train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_string('data_dir', '/tmp/cifar10_data',
                           """Directory where to get input """)
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('eval_data_set', 'test_batch.bin',
                           """Data set for evaluation""")
tf.app.flags.DEFINE_string('eval_dir', '/tmp/cifar10_eval',
                           """Directory for evaluation""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 10,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('image_size', 32, """The size of cifar images used.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_examples', 10000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', True, """Whether to run eval only once.""")
tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_boolean('use_fp16', False, """Train the model using fp16.""")
tf.app.flags.DEFINE_integer('batch_size', 128, """Number of images to process in a batch.""")

# MODE = '32_to_24'
# MODE = '24_to_noise'
MODE = 'normal'
# MODE = 'show'
# MODE = 'vote_show'
# MODE = 'train'
EPS = int(0.06 * 255)
NUM_EXAMPLES = 1000
if MODE == 'normal':
    # EVAL_DATA = 'test_batch_15_f2_e2_after_cae.bin'
    EVAL_DATA = 'test_batch_noise_custom_25.bin'
    # ORG_IMAGE_SIZE = 24
elif MODE == 'train':
    # ORG_IMAGE_SIZE = 32
    EVAL_DATA = 'test_batch.bin'
else:
    EVAL_DATA = 'test_batch_org.bin'
    # ORG_IMAGE_SIZE = 24
#
# if MODE == 'normal' or MODE == 'train':
#     tf.app.flags.DEFINE_integer('batch_size', 128, """Number of images to process in a batch.""")
# else:
#     tf.app.flags.DEFINE_integer('batch_size', 1, """Number of images to process in a batch.""")
# tf.app.flags.DEFINE_string('data_dir', '/tmp/cifar10_data', """Path to the CIFAR-10 data directory.""")
