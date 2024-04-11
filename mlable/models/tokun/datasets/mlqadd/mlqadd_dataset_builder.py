"""Wrapper around the MLQA dataset."""

import itertools

import tensorflow as tf
import tensorflow_datasets as tfds

import mlable.models.tokun.datasets.pipeline as _mmad

# GENERIC #####################################################################

def _cycle(iterators: list) -> iter:
    return itertools.chain(*zip(*iterators))

# LOAD DATA ###################################################################

def _load(lang: str='en', **kwargs) -> iter:
    __name = 'mlqa/{language}'.format(language=lang)
    __shuffle_files = kwargs.get('shuffle_files', True)
    __data_dir = kwargs.get('data_dir', '~/.cache/tensorflow/')
    __split = kwargs.get('split', 'test')
    __dataset = tfds.load(name=__name, split=__split, shuffle_files=__shuffle_files, data_dir=__data_dir)
    return iter(__dataset.batch(1))

# PREPROCESS ##################################################################

def _merge(batch: dict) -> tf.Tensor:
    return batch['title'] + b'\n' + batch['context'] + b'\n' + batch['question'] + b'\n' + batch['answers']['text']

def _preprocess(batch: tf.Tensor) -> str:
    __flat = tf.reshape(tensor=batch, shape=(-1,))
    __concat = b''.join(__flat.numpy().tolist())
    return __concat.decode('utf-8')

# BUILD #######################################################################

class Builder(tfds.core.GeneratorBasedBuilder):
    """Builder for MLQAdd dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {'1.0.0': 'Initial release.',}

    def __init__(self, train_lang: list=['en'], test_lang: list=['es'], **kwargs) -> None:
        super(Builder, self).__init__(**kwargs)
        # track sample ids, used when saving to disk
        self._train_id = -1
        self._test_id = -1
        # batch iterators, one sample per query, one iterator per language
        self._train_iter = _cycle(iterators=[_load(lang=__l, split='test', **kwargs) for __l in train_lang])
        self._test_iter = _cycle(iterators=[_load(lang=__l, split='validation', **kwargs) for __l in test_lang])

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return self.dataset_info_from_configs(
            homepage='https://github.com/apehex/mlable/',
            supervised_keys=('utf-32', 'utf-32'),
            disable_shuffling=False,
            features=tfds.features.FeaturesDict({
                # 'bytes': tfds.features.Text(),
                'utf-32': tfds.features.Tensor(shape=(4, 256), dtype=tf.dtypes.float32)}))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Generates the data splits."""
        return {'train': self._generate_examples(train=True), 'test': self._generate_examples(train=False)}

    def _generate_examples(self, train: bool=True) -> iter:
        """Produces long text samples with mixed languages."""
        if train:
            for __batch in self._train_iter: # single sample in batch
                __utf8 = tf.reshape(tensor=_merge(batch=__batch), shape=()).numpy()
                __utf32 = _mmad.tokenize(text=__utf8.decode('utf-8'))
                for __i in range(__utf32.shape[0]):
                    self._train_id += 1
                    yield self._train_id, {'utf-32': tf.one_hot(indices=__utf32[__i], depth=256, dtype=tf.dtypes.float32).numpy()}
        else:
            for __b in self._test_iter: # single sample in batch
                self._test_id += 1
                __utf8 = tf.reshape(tensor=_merge(batch=__b), shape=()).numpy()
                __utf32 = _mmad.tokenize(text=__utf8.decode('utf-8'))
                for __i in range(__utf32.shape[0]):
                    self._test_id += 1
                    yield self._test_id, {'utf-32': tf.one_hot(indices=__utf32[__i], depth=256, dtype=tf.dtypes.float32).numpy()}
