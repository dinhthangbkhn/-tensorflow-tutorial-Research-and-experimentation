import tensorflow as tf

tf.enable_eager_execution()

# -------------------------------Tensors-----------------------------------------------
print(tf.add(1, 2))
print(tf.add([1, 2], [3, 4]))
print(tf.square(5))
print(tf.reduce_sum([1, 2, 3]))
print(tf.encode_base64("hello world"))

print(tf.square(2) + tf.square(3))
print(tf.add(tf.square(2), tf.square(3)))

x = tf.matmul([[1]], [[2, 3]])
print(x.shape)
print(x.dtype)

# -------------------------------NumPy Compatibility---------------------------------
import numpy as np

ndarray = np.ones([3, 3])

print("Tensorflow operations convert numpy arrays to Tensors automatically")
tensor = tf.multiply(ndarray, 42)
print(tensor)

print("Add Numpy operations convert Tensors to numpy arrays automatically")
print(np.add(tensor, 1))

print("The .numpy() method explicitly converts a Tensor to a numpy array")
print(tensor.numpy())

# --------------------------------GPU acceleration----------------------------------------
x = tf.random_uniform([3, 3])
print("Is there a GPU available:")
print(tf.test.is_gpu_available())

print("Is the Tensor on GPU #0")
print(x.device.endswith("GPU:0"))

# Create a source Dataset
# Create a source dataset using one of the factory functions
# like Dataset.from_tensors, Dataset.from_tensor_slices or
# using objects that read from files like TextLineDataset or TFRecordDataset.
# See the TensorFlow Guide for more information.
ds_tensors = tf.data.Dataset.from_tensor_slices([1,2,3,4,5,6])

#----------------------------------Make dataset from csv file------------------------
import tempfile
_, filename = tempfile.mkstemp()

with open(filename, "w") as f:
    f.write("""Line1
    Line 2
    Line3""")
ds_file = tf.data.TextLineDataset(filename)
print(ds_file)

#-----------------------------------Apply transformations------------------------------
print("Application transformation")
# print(ds_tensors)
ds_tensors = ds_tensors.map(tf.square).shuffle(2).batch(2)
ds_file = ds_file.batch(2)

print(ds_tensors)
print(ds_file)

#-----------------------------------------Iterate---------------------------------------
print("\nElements of ds_tensors:")
for x in ds_tensors:
    print(x)
print("\nElements in ds_file")
for x in ds_file:
    print(x)
