from __future__ import absolute_import, division, print_function
import os
import matplotlib.pyplot as plt
import tensorflow as tf
tfe = tf.contrib.eager
tf.enable_eager_execution()

print("Tensorflow version: {}".format(tf.VERSION))
print("Eager execution: {}".format(tf.executing_eagerly()))

train_dataset_url = "http://download.tensorflow.org/data/iris_training.csv"

train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
                                           origin=train_dataset_url)
print("Local copy of the dataset file: {}".format(train_dataset_fp))
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width','species']
feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
label_name = 'species'
class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']
#---------------------------CREATE tf.data.Dataset --------------------------------------
batch_size = 32

train_dataset = tf.contrib.data.make_csv_dataset(
    train_dataset_fp,
    batch_size,
    column_names = column_names,
    label_name = label_name,
    num_epochs = 1
)
# print(next(iter(train_dataset))) #get a batch
features, labels = next(iter(train_dataset))
# print(list(features))
# print(features.keys())
# print(list(features.values()))
# print(labels)

def pack_features_vector(features, labels):
    features = tf.stack(list(features.values()), axis=1) #4 features, 32 values (4,32) -> 32 example, 4 features (32,4)
    return features, labels

train_dataset = train_dataset.map(pack_features_vector)
# features, labels = next(iter(train_dataset))

# print(features)



#---------------------------CREATE MODEL---------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation = tf.nn.relu, input_shape = (4,)),
    tf.keras.layers.Dense(10, activation = tf.nn.relu),
    tf.keras.layers.Dense(3, activation=tf.nn.softmax)
])
# prediction = model(features)
# print(prediction)
# print("Prediction {}".format(tf.argmax(prediction, axis=1)))



#---------------------------TRAIN THE MODEL ---------------------------
#---------------------------Define the loss and gradient function ---------------------------
def loss(model, x, y):
    y_  = model(x)
    return tf.losses.sparse_softmax_cross_entropy(labels = y, logits = y_)
# l = loss(model, features, labels) #tensor
# print("Loss test: {}".format(l))

def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

#---------------------------Create an optimizer ---------------------------
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
global_step = tf.train.get_or_create_global_step()

# loss_value, grads = grad(model, features, labels)
# print("Step:{}, Initial Loss:{}".format(global_step.numpy(), loss_value.numpy()))

#Apply gradients to variables.
# apply_gradients(
#     grads_and_vars,
#     global_step=None,
#     name=None
# )
# Args:
# grads_and_vars: List of (gradient, variable) pairs as returned by compute_gradients().
# global_step: Optional Variable to increment by one after the variables have been updated.
# name: Optional name for the returned operation. Default to the name passed to the Optimizer constructor.
# optimizer.apply_gradients(zip(grads, model.trainable_variables), global_step = global_step)
# print("Step:{}, Initial Loss:{}".format(global_step.numpy(), loss(model, features, labels)))



#---------------------------TRAIN LOOP---------------------------
train_loss_results = []
train_accuracy_results = []

num_epochs = 201
for epoch in range(num_epochs):
    epoch_loss_avg = tfe.metrics.Mean()
    epoch_accuracy = tfe.metrics.Accuracy()
    #Train loop using batches 32
    for x, y in train_dataset:
        #optimize the model
        loss_value, grads = grad(model, x, y)
        optimizer.apply_gradients(zip(grads,model.trainable_variables), global_step= global_step)

        #track progress
        epoch_loss_avg(loss_value)
        epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)
    #end epoch
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())

    if epoch%50 == 0:
        print("Epoch:{} Loss:{} Accuracy:{}".format(epoch, epoch_loss_avg.result(), epoch_accuracy.result()))


#---------------------------VISUALIZE LOSS FUNCTION ---------------------------
fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')

axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(train_loss_results)

axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(train_accuracy_results);

#---------------------------EVALUATE MODEL ---------------------------
test_url = "http://download.tensorflow.org/data/iris_test.csv"

test_fp = tf.keras.utils.get_file(fname=os.path.basename(test_url),
                                  origin=test_url)
test_dataset = tf.contrib.data.make_csv_dataset(
    train_dataset_fp,
    batch_size,
    column_names=column_names,
    label_name='species',
    num_epochs=1,
    shuffle=False)

test_dataset = test_dataset.map(pack_features_vector)
test_accuracy = tfe.metrics.Accuracy()
for (x,y) in test_dataset:
    logits = model(x)
    prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
    test_accuracy(prediction, y)
print("Test set accuracy: {:.3}".format(test_accuracy.result()))
