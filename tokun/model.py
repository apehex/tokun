import tensorflow as tf

import tokun.layers

# ENCODER #####################################################################

class Encoder(tf.keras.models.Model):
    def __init__(self, depth: int, token_dim: int, encoding_dim: int, embedding_dim: int, latent_dim: int, batch_dim: int=None, attention: bool=False, normalization: bool=False, **kwargs) -> None:
        super(Encoder, self).__init__(**kwargs)
        self._encoder = tf.keras.Sequential([
            tf.keras.Input(shape=(encoding_dim,), batch_size=batch_dim, name='input'), # (B * G ^ D, U)
            tf.keras.layers.Dense(units=embedding_dim, activation=None, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer=None, name='embed-1'),] # (B * G ^ D, U) => (B * G ^ D, E)
            + [tokun.layers.TokenizeBlock(left_axis=-2, right_axis=-1, token_dim=token_dim, latent_dim=latent_dim, attention=attention, normalization=normalization, name='tokenize' + (__i + 1) * '-4') for __i in range(depth)]) # (B * G ^ i, E) => (B * G ^ (i-1), E)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self._encoder(x)

# DECODER #####################################################################

class Decoder(tf.keras.models.Model):
    def __init__(self, depth: int, token_dim: int, encoding_dim: int, embedding_dim: int, latent_dim: int, batch_dim: int=None, attention: bool=False, normalization: bool=False, **kwargs) -> None:
        super(Decoder, self).__init__(**kwargs)
        self._decoder = tf.keras.Sequential(
            [tf.keras.Input(shape=(latent_dim,), batch_size=batch_dim, name='input')] # (B, E)
            + [tokun.layers.DetokenizeBlock(token_dim=token_dim, embedding_dim=embedding_dim, attention=attention, normalization=normalization, name='detokenize' + (depth - __i) * '-4') for __i in range(depth)] # (B * G ^ i, E) => (B * G ^ (i+1), E)
            + [tokun.layers.HeadBlock(encoding_dim=encoding_dim, name='project-head')]) # (B * G ^ D, E) => (B * G ^ D, U)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self._decoder(x)

# VAE #########################################################################

class AutoEncoder(tf.keras.models.Model):
    def __init__(self, depth: int, token_dim: int, encoding_dim: int, embedding_dim: int, latent_dim: int, batch_dim: int=None, attention: bool=False, normalization: bool=False, **kwargs) -> None:
        super(AutoEncoder, self).__init__(**kwargs)
        self._encoder = Encoder(depth=depth, token_dim=token_dim, encoding_dim=encoding_dim, embedding_dim=embedding_dim, latent_dim=latent_dim, batch_dim=batch_dim, attention=attention, normalization=normalization)
        self._decoder = Decoder(depth=depth, token_dim=token_dim, encoding_dim=encoding_dim, embedding_dim=embedding_dim, latent_dim=latent_dim, batch_dim=batch_dim, attention=attention, normalization=normalization)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self._decoder(self._encoder(x))
