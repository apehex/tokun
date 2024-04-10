"""MLQA dataset."""

import tensorflow_datasets as tfds

import mlable.models.tokun.datasets.mlqa.mlqa_dataset_builder as mmtdmm

class MlqaTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for MLQA dataset."""
  DATASET_CLASS = mmtdmm.mlqa_dataset_builder.Builder
  SPLITS = {'train': 3,}

if __name__ == '__main__':
  tfds.testing.test_main()
