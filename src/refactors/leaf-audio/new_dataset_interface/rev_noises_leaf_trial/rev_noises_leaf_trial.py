"""rev_noises_leaf_trial dataset."""

import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as p


# being leaf-audio a different project
# import framework.experimental.nn.dataset_utilities as dsu
# cannot be performed. --> code replication :(
def split_tvt(df: p.DataFrame):
    import numpy as np
    train, test, val = np.array_split(df, [int(df.shape[0] * 0.7), int(df.shape[0] * 0.8)])
    return train, val, test


def floating_to_pcm(signal):
    import numpy as np
    return (signal * np.iinfo(np.int16).max).astype(dtype=np.int16)

# https://www.tensorflow.org/datasets/add_dataset

# TODO(rev_noises_leaf_trial): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(rev_noises_leaf_trial): BibTeX citation
_CITATION = """
"""


class RevNoisesLeafTrial(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for rev_noises_leaf_trial dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(rev_noises_leaf_trial): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'audio': tfds.features.Audio(shape=(None,), dtype=tf.int16, sample_rate=16000),
            'label': tfds.features.ClassLabel(names=['RectangleRoomSample', 'LRoomSample', 'HouseRoomSample']),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('audio', 'label'),  # Set to `None` to disable
        homepage='https://dataset-homepage/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(rev_noises_leaf_trial): Downloads the data and defines the splits
    # caricare il dataset pkl, fare split ed un po'di pre-processing
    # HATEFUL ABS PATH
    df = p.read_pickle("/nas/home/gantonacci/Thesis/Project/pythonProject/datasets/nn/final_datasets/main_rlh_wn_frmp0_nofdr_for_leaf.pkl")
    dftrain, dfval, dftest = split_tvt(df)

    # TODO(rev_noises_leaf_trial): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(dftrain),
        'validation': self._generate_examples(dfval),
        'test': self._generate_examples(dftest)
    }

  def _generate_examples(self, split: p.DataFrame):
    """Yields examples."""
    # TODO(rev_noises_leaf_trial): Yields (key, example) tuples from the dataset
    for t in split.itertuples():
      yield t.Index, {
          # data.prepare si aspetta vettori wave PCM16
          'audio': floating_to_pcm(t.X),
          'label': t.class_label,
      }
