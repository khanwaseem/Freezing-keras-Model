
"""
    Purpose             : To do inference using freezed model in tensorflow
    Device Requirment   : Python Interpreter with Tensorflow-2.x
    Author              : Waseem Khan
    Last Edited         : 6 June 2022
    Edited By           : Waseem Khan
    File                : infer_copy.py

"""

import numpy as np
import os
import cv2
import time
import tensorflow as tf

def wrap_frozen_graph(graph_def, inputs, outputs, print_graph=False):
    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph

    if print_graph == True:
        print("-" * 50)
        print("Frozen model layers: ")
        layers = [op.name for op in import_graph.get_operations()]
        for layer in layers:
            print(layer)
        print("-" * 50)

    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))

frozen_model = './frozen_model/tf26/vehicalClassFrozen.pb' ## .pb file

# Load frozen graph using TensorFlow 1.x functions
with tf.io.gfile.GFile(frozen_model, "rb") as f:
    graph_def = tf.compat.v1.GraphDef()
    print(graph_def)
    loaded = graph_def.ParseFromString(f.read())

# Wrap frozen graph to ConcreteFunctions
frozen_func = wrap_frozen_graph(graph_def=graph_def,
                                inputs=["x:0"],
                                outputs=["Identity:0"],
                                print_graph=False)

# print("-" * 50)
# print("Frozen model inputs: ")
# print(frozen_func.inputs)
# print("Frozen model outputs: ")
# print(frozen_func.outputs)

path = 'source'

labels_name = {0:'37-MAV 6 Axle', 1:'09-Car Ecco', 2:'26-MAV 3 Axle Truck', 3:'34-MAV 5 lifted axle', 4:'07-Jeep Scorpio Bolero', 5:'36-MAV 5 AXLE trailer Lifted', 6:'03-Car Hatchback',
                    7:'38-MAV 6 Axle Lifted', 8:'33-MAV 5 Axle', 9:'31-MAV 4 lifted axle', 10:'23-Bus 2 Axle Big', 11:'08-PickUp', 12:'30-MAV 4 Axle', 13:'32-MAV 4 Axle trailer', 14:'05-Car SUV', 
                    15:'11-Tata Ace', 16:'17-Truck 2 Axle  Big', 17:'13-LCV 2 Axle Truck', 18:'22-Truck 2 Axle loading', 19:'35-MAV 5 Axle Trailer', 20:'04-Car Sedan'}

IMAGES = [os.path.join(path, img) for img in os.listdir(path) if img.endswith('.jpg')]
total_time = 0
img_count = 0
for image in IMAGES:
    im = cv2.imread(image)
    img_count += 1
    start = time.time()
    im = cv2.resize(im, (224, 224)).astype('float32')
    img = np.expand_dims(im, axis=0)

    # Get predictions for test images
    frozen_graph_predictions = frozen_func(x=tf.constant(img))[0]
    end = time.time()
    class_name = labels_name[tf.argmax(frozen_graph_predictions[0].numpy()).numpy()]
    key_value = tf.argmax(frozen_graph_predictions[0].numpy()).numpy()
    class_confidence = round((max(max(frozen_graph_predictions)) * 100).numpy())
    return_values_tuple = class_name, key_value, str(class_confidence)
    print(return_values_tuple)
    time_ = end-start

    total_time += time_

    #print(f"{labels_name[tf.argmax(frozen_graph_predictions[0].numpy()).numpy()]}, Classification time: {time_}")
    # Print the prediction for the first image
    # print("-" * 50)
    # print("Example TensorFlow frozen graph prediction reference:")

    # print(labels_name[tf.argmax(frozen_graph_predictions[0].numpy()).numpy()])

print('\nTotal image count:', img_count)
print(f'Avg time per image: {round(total_time/img_count, 4)}')