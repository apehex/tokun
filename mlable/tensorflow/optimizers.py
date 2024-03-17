import math
import random
import tensorflow as tf

# CONSTANTS ###################################################################

N_BATCH = 64
N_EPOCHS = 1

R_TRAINING = 0.1

# LEARNING RATE ###############################################################

def learning_rate_hokusai(epoch: int, lr_min: float, lr_max: float, lr_exp: float, rampup: int, sustain: int) -> float:
    __lr = lr_min
    if epoch < rampup:
        __lr = lr_min + (epoch * (lr_max - lr_min) / rampup)
    elif epoch < rampup + sustain:
        __lr = lr_max
    else:
        __lr = lr_min + (lr_max - lr_min) * lr_exp ** (epoch - rampup - sustain)
    return __lr

# SGD #########################################################################

def step(model: tf.Module, loss: callable, x: tf.Tensor, y: tf.Tensor, rate: float=R_TRAINING) -> tuple:
    with tf.GradientTape() as __tape:
        # grad / data
        __ratios = []
        # compute gradient on these
        __tape.watch(model.trainable_variables)
        # loss
        __loss = loss(y_true=y, y_pred=model(x, training=True))
        # backward
        __grad = __tape.gradient(__loss, model.trainable_variables)
        # update the model
        for __i in range(len(model.trainable_variables)):
            model.trainable_variables[__i].assign_sub(rate * __grad[__i])
            __ratios.append(math.log10(tf.math.reduce_std(rate * __grad[__i]) / tf.math.reduce_std(model.trainable_variables[__i])))
    return __loss, __ratios

def train(model: tf.Module, loss: callable, x_train: tf.Tensor, y_train: tf.Tensor, x_test: tf.Tensor, y_test:tf.Tensor, epochs: int=N_EPOCHS, batch: int=N_BATCH, rate: float=R_TRAINING) -> tuple:
    # debugging data
    __train_loss = []
    __test_loss = []
    __ratios = []
    # steps per epoch
    __steps = int(x_train.shape[0]) // batch
    for __i in range(epochs):
        for __j in range(__steps):
            # iteration number
            __k = __i * __steps + __j
            # random batch
            __indices = random.sample(population=range(x_train.shape[0]), k=batch)
            __x = tf.gather(params=x_train, indices=__indices, axis=0)
            __y = tf.gather(params=y_train, indices=__indices, axis=0)
            # update the model
            __loss, __r = step(model=model, x=__x, y=__y, loss=loss, rate=rate)
            # save loss
            __train_loss.append((__k, __loss))
            # save ratios grad / data
            __ratios.append((__k, __r))
            if __j % __steps == 0:
                # log the progress
                __test_loss.append((__k, loss(y_true=y_test, y_pred=model(x_test, training=False))))
                print('[epoch {epoch}] train loss: {train} test loss: {test}'.format(epoch=__i, train=__train_loss[-1][1], test=__test_loss[-1][1]))
    return __train_loss, __test_loss, __ratios
