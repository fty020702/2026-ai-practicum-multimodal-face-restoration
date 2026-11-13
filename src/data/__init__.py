"""Data: synthetic + real old-photo datasets, degradations, prompt templates.

Use make_dataset(cfg, ...) to get the dataset selected by `data.source`:
  - default (source absent)  -> RestorationDataset (toy, unchanged)
  - source == "old_photo"    -> OldPhotoDataset (real DIV2K + paper-style aging)
"""

from .dataset import RestorationDataset, make_dataset

__all__ = ["RestorationDataset", "make_dataset"]
