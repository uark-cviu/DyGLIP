import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

""""
# Evaluation logging pipeline:
# 1: Create a model-name directory that contains log/ model/ output/ csv/.
# 2: For different datasets, the model, log and output files will include the dataset name. Also, create a flags.txt log file inside the folder.
# Model name convention -> Always starts with the base_model name (DyGLIP), i.e., "base_model"_"model"
"""

# General params for experiment setup - which need to be provided.
flags.DEFINE_string('base_model', 'DyGLIP', 'Base model string.')
flags.DEFINE_string('model', 'default', 'Model string.')

flags.DEFINE_string('dataset', 'Enron_new', 'Dataset string.')
flags.DEFINE_string('dataset_path', 'Enron_new', 'Dataset path.')
flags.DEFINE_string('dataset_cache_path', 'Enron_new', 'Dataset path.')
flags.DEFINE_integer('time_steps', 3, '# time steps to train (+1)') # Predict at next time step.
flags.DEFINE_integer('GPU_ID', 0, 'GPU_ID')
flags.DEFINE_integer('epochs', 10, 'Number of epochs to train.')
flags.DEFINE_integer('batch_size', 1, 'Batch size (# nodes)')
flags.DEFINE_boolean('featureless', False, 'Use 1-hot instead of features')
flags.DEFINE_boolean('use_attention', False, 'Use attention features')
flags.DEFINE_float('max_gradient_norm', 1.0, 'Clip gradients to this norm')
flags.DEFINE_string('features', 'reid', 'Node Embedding Features')

flags.DEFINE_float('edge_val_fraction', 0.2, 'Percentage of val')
flags.DEFINE_float('edge_test_fraction', 0.0, 'Percentage of test')

# Evaluation settings.
flags.DEFINE_integer('test_freq', 1, 'Testing frequency')
flags.DEFINE_integer('val_freq', 1, 'Validation frequency')

# Tunable hyper-parameters.
flags.DEFINE_integer('neg_sample_size', 10, 'number of negative samples')
flags.DEFINE_integer('walk_len', 40, 'Walk len')
flags.DEFINE_float('neg_weight', 1, 'Wt. for negative samples')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate for self-attention model.')

flags.DEFINE_float('spatial_drop', 0.1, 'attn Dropout (1 - keep probability).')
flags.DEFINE_float('temporal_drop', 0.5, 'ffd Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0.0005, 'Weight for L2 loss on embedding matrix.')

flags.DEFINE_boolean('use_residual', False, 'Residual connections')

# Architecture configuration parameters.
flags.DEFINE_string('structural_head_config', '16', 'Encoder layer config: # attention heads in each GAT layer')
flags.DEFINE_string('structural_layer_config', '128', 'Encoder layer config: # units in each GAT layer')

flags.DEFINE_string('temporal_head_config', '16', 'Encoder layer config: # attention heads in each GAT layer')
flags.DEFINE_string('temporal_layer_config', '128', 'Encoder layer config: # units in each GAT layer')

flags.DEFINE_boolean('position_ffn', True, 'Use position wise feedforward')

# Generally static parameters -> Will not be updated by the argparse parameters.
flags.DEFINE_string('optimizer', 'adam', 'Optimizer for training: (adadelta, adam, rmsprop)')
flags.DEFINE_integer('seed', 7, 'Random seed')

# Directory structure.
flags.DEFINE_string('save_dir', "output", 'Save dir defaults to output/ within the base directory')
flags.DEFINE_string('log_output_dir', "log", 'Log dir defaults to log/ within the base directory')
flags.DEFINE_string('csv_dir', "csv", 'CSV dir defaults to csv/ within the base directory')
flags.DEFINE_string('model_dir', "model", 'Model dir defaults to model/ within the base directory')
flags.DEFINE_string('best_model_dir', "best_model", 'Model dir defaults to model/ within the base directory')
flags.DEFINE_string('output_mot_dir', "output_features", 'Directory for MOT output')

flags.DEFINE_integer('window', 5, 'Window for temporal attention (default : -1 => full)')
