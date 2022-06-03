
"""
    Purpose             : To convert AVC TensorFlow classifier model (.h5 file) to frozen graph (.pb format).
    Device Requirment   : Python Interpreter with Tensorflow-2.x
    Author              : Shivam Dixit
    Last Edited         : 15 March 2022
    Edited By           : Shivam Dixit
    File                : freezTF.py

"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

#path of the directory where you want to save your model
frozen_out_path = './frozen_model'

# name of the .pb file
frozen_graph_filename = "vehicalClassFrozen"
model = tf.keras.models.load_model('./savedModel/resnet50V2_fineTuned_orig_data_tf2_6.h5')
# Convert Keras model to ConcreteFunction
full_model = tf.function(lambda x: model(x))
full_model = full_model.get_concrete_function(
    tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
# Get frozen ConcreteFunction
frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()
layers = [op.name for op in frozen_func.graph.get_operations()]
print("-" * 60)
print("Frozen model layers: ")
for layer in layers:
    print(layer)
print("-" * 60)
print("Frozen model inputs: ")
print(frozen_func.inputs)
print("Frozen model outputs: ")
print(frozen_func.outputs)
# Save frozen graph to disk
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir=frozen_out_path,
                  name=f"{frozen_graph_filename}.pb",
                  as_text=False)
# Save its text representation
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir=frozen_out_path,
                  name=f"{frozen_graph_filename}.pbtxt",
                  as_text=True)