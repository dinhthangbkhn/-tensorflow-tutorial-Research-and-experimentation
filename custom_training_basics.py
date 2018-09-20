import tensorflow as tf
tf.enable_eager_execution()
tfe = tf.contrib.eager
#----------------------Variable-----------------------------
# x= tf.zeros([10,10])
# x+=2
# print(x)
#
# v=tf.Variable(1.0)
# assert v.numpy() == 1.0
# v.assign(3)
# assert v.numpy() == 3.0
# v.assgin(tf.square(v))
# assert v.numpy() == 9.0

#----------------------------DEFINE THE MODEL--------------------------
class Model(object):
    def __init__(self):
        self.W = tfe.Variable(5.0)
        self.b = tfe.Variable(0.0)
    def __call__(self, x):
        return self.W*x + self.b
model = Model()

#----------------------------DEFINE A LOSS FUNCTION --------------------
def loss(predicted_y, desired_y):
    return tf.reduce_mean(tf.square(predicted_y-desired_y))

#----------------------------OBTAIN TRAINING DATA--------------------
TRUE_W = 3.0
TRUE_b = 2.0
NUM_EXAMPLES = 1000

inputs = tf.random_normal(shape=[NUM_EXAMPLES])
noise = tf.random_normal(shape=[NUM_EXAMPLES])
outputs = inputs*TRUE_W + TRUE_b + noise

import matplotlib.pyplot as plt
plt.scatter(inputs, outputs, c="b")
plt.scatter(inputs, model(inputs), c="r")
plt.show()

print("Current loss:")
print(loss(model(inputs), outputs).numpy())


def train(model, inputs, outputs, learning_rate):
    with tf.GradientTape() as t:
        current_loss = loss(model(inputs), outputs)
    dW, db = t.gradient(current_loss, [model.W, model.b])
    model.W.assign_sub(learning_rate*dW)
    model.b.assign_sub(learning_rate*db)

model = Model()
Ws,bs=[],[]
epochs = range(10)
for epoch in epochs:
    Ws.append(model.W.numpy())
    bs.append(model.b.numpy())
    current_loss = loss(model(inputs), outputs)
    train(model, inputs, outputs, learning_rate=0.1)
    print("Epoch %2d: W=%1.2f b=%1.2f loss=%2.5f" % (epoch, Ws[-1], bs[-1], current_loss))

#Plot it all
plt.plot(epochs, Ws, "r", epochs, bs, "b")
plt.plot([TRUE_W]*len(epochs), "r--", [TRUE_b]*len(epochs), "b--")
plt.legend(["W", "b", "true W", "true_b"])
plt.show()