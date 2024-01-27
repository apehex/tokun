import math
import random
import tensorflow as tf

# CONSTANTS ###################################################################

N_BATCH = 64
N_STEPS = 2**10

R_TRAINING = 0.1

# SGD #########################################################################

def step(model: tf.Module, loss: callable, x: tf.Tensor, y: tf.Tensor, rate: float=R_TRAINING) -> tuple:
    with tf.GradientTape() as __tape:
        # grad / data
        __ratios = []
        # compute gradient on these
        __tape.watch(model.trainable_variables)
        # loss
        __loss = loss(target_y=y, predicted_y=model(x, training=True))
        # backward
        __grad = __tape.gradient(__loss, model.trainable_variables)
        # update the model
        for __i in range(len(model.trainable_variables)):
            model.trainable_variables[__i].assign_sub(rate * __grad[__i])
            __ratios.append(math.log10(tf.math.reduce_std(rate * __grad[__i]) / tf.math.reduce_std(model.trainable_variables[__i])))
    return __loss, __ratios

def train(model: tf.Module, loss: callable, x_train: tf.Tensor, y_train: tf.Tensor, x_test: tf.Tensor, y_test:tf.Tensor, steps: int=N_STEPS, batch: int=N_BATCH, rate: float=R_TRAINING) -> tuple:
    __train_loss = []
    __test_loss = []
    __ratios = []
    for __i in range(steps + 1):
        # random batch
        __indices = random.sample(population=range(x_train.shape[0]), k=batch)
        __x = tf.gather(params=x_train, indices=__indices, axis=0)
        __y = tf.gather(params=y_train, indices=__indices, axis=0)
        # update the model
        __loss, __r = step(model=model, x=__x, y=__y, loss=loss, rate=rate)
        # save loss
        __train_loss.append((__i, __loss))
        # save ratios grad / data
        __ratios.append((__i, __r))
        # log the progress
        if __i % int(0.1 * steps) == 0:
            __test_loss.append((__i, loss(target_y=y_test, predicted_y=model(x_test, training=False))))
            print('[epoch {epoch}] train loss: {train} test loss: {test}'.format(epoch=__i, train=__train_loss[-1][1], test=__test_loss[-1][1]))
    return __train_loss, __test_loss, __ratios
