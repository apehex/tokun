import tensorflow as tf

import mlable.metrics

# GROUP ACCURACY ##############################################################

class CategoricalGroupAccuracyFromEmbeddings(mlable.metrics.CategoricalGroupAccuracy):
    def __init__(self, decoder: callable, group: int=4, name: str='categorical_group_accuracy_from_embeddings', dtype: tf.dtypes.DType=tf.dtypes.float32, **kwargs) -> None:
        # init
        super(CategoricalGroupAccuracyFromEmbeddings, self).__init__(group=group, name=name, dtype=dtype, **kwargs)
        # decoder
        self._decoder = decoder

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight: tf.Tensor=None) -> None:
        __yt = self._decoder(y_true)
        __yp = self._decoder(y_pred)
        super(CategoricalGroupAccuracyFromEmbeddings, self).update_state(y_true=__yt, y_pred=__yp, sample_weight=sample_weight)

    def get_config(self) -> dict:
        __config = super(CategoricalGroupAccuracyFromEmbeddings, self).get_config()
        __config.update({'decoder': self._decoder})
        return __config

    @classmethod
    def from_config(cls, config: dict) -> tf.keras.metrics.Metric:
        __decoder = config.pop('decoder')
        return cls(decoder=__decoder, **config)
