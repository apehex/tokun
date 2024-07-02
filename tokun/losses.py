import tensorflow as tf

# CCE #########################################################################

class CategoricalCrossentropyFromEmbeddings(tf.keras.losses.CategoricalCrossentropy):
    def __init__(self, decoder: callable, name='categorical_crossentropy_from_embeddings', **kwargs) -> None:
        # init
        super(CategoricalCrossentropyFromEmbeddings, self).__init__(from_logits=False, label_smoothing=0.0, axis=-1, reduction='sum_over_batch_size', **kwargs)
        # decoder
        self._decoder = decoder

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        # decode
        __yt = self._decoder(y_true)
        __yp = self._decoder(y_pred)
        # decompression factor
        __dim = int(tf.size(__yt) / tf.size(y_true))
        # expanded loss
        __loss = super(CategoricalCrossentropyFromEmbeddings, self).call(y_true=__yt, y_pred=__yp)
        # expanded shape
        __shape = list(__loss.shape)
        # group by token
        __shape[-1] = __shape[-1] // __dim
        __shape.append(__dim)
        # reshape
        __loss = tf.reshape(__loss, shape=__shape)
        # reduce so that the loss shape matches the input shapes
        return tf.reduce_mean(__loss, axis=-1)

    def get_config(self) -> dict:
        __config = super(CategoricalCrossentropyFromEmbeddings, self).get_config()
        __config.update({'decoder': self._decoder})
        return __config

    @classmethod
    def from_config(cls, config: dict) -> tf.keras.losses.Loss:
        __decoder = config.pop('decoder')
        return cls(decoder=__decoder, **config)
