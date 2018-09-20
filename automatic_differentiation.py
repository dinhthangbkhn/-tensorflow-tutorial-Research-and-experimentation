import tensorflow as tf

tf.enable_eager_execution()
tfe = tf.contrib.eager
# print(tfe)

# ---------------------Derivatives of a function----------------------
from math import pi


def f(x):
    return tf.square(tf.sin(x))


assert f(pi / 2).numpy() == 1.0

grad_f = tfe.gradients_function(f)  # function

assert tf.abs(grad_f(pi / 2)[0]).numpy() < 1e-7


# ---------------------Higher order gradients--------------------------
def f(x):
    return tf.square(tf.sin(x))


def grad(f):
    return lambda x: tfe.gradients_function(f)(x)[0]


x = tf.lin_space(-2 * pi, 2 * pi, 100)
import matplotlib.pyplot as plt

plt.plot(x, f(x), label="f")  # tensor
plt.plot(x, grad(f)(x), label="first derivative")  # tensor
plt.plot(x, grad(grad(f))(x), label="second derivative")  # tensor
plt.plot(x, grad(grad(grad(f)))(x), label="third derivative")  # tensor
plt.legend()
plt.show()


# ---------------------------Gradient tapes-----------------------------
# TensorFlow first "records" all the operations applied to compute the output of the function.
# We call this record a "tape"
# It then uses that tape and the gradients functions associated with each primitive operation
# to compute the gradients of the user-defined function using
def f(x, y):
    output = 1
    for i in range(int(y)):
        output = tf.multiply(output, x)
    return output

def g(x, y):
    return tfe.gradients_function(f)(x, y)[0]

assert f(3, 2).numpy() == 9.0
assert f(4, 3).numpy() == 64
assert g(3.0, 2).numpy() == 6
assert g(4.0, 3).numpy() == 48

#tinh dao ham tuong ung voi tham so tuong ung
x = tf.ones((2,2))
with tf.GradientTape(persistent=True) as t:
    t.watch(x)
    y=tf.reduce_sum(x)
    z=tf.multiply(y,y)

dz_dy = t.gradient(z,y)
assert dz_dy.numpy() == 8

dz_dx=t.gradient(z,x)
for i in [0,1]:
    for j in [0,1]:
        assert dz_dx[i][j].numpy() == 8



#----------------------------------Higher-order gradients-------------------------------------
x=tf.constant(1.0)

with tf.GradientTape() as t:
    with tf.GradientTape() as t2:
        t2.watch(x)
        y=x**3
    dy_dx = t2.gradient(y,x)
d2y_d2x = t.gradient(dy_dx, x)
assert dy_dx.numpy() == 3
assert d2y_d2x.numpy() == 6
