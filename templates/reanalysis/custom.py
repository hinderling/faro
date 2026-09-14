"""Components specific to this re-analysis.

A re-analysis usually reuses the classes of the experiment it re-processes:
copy them from that experiment's ``custom.py`` (or import that file by path)
and import them in the pipeline cell:

    from custom import MySegmentator, MyFeatureExtractor

Keep them out of notebook cells so they can be tested. Once a class is
needed by a second experiment, move it into faro and pin the commit that
has it.
"""

import numpy as np
import pandas as pd
from skimage.measure import regionprops_table

from faro.feature_extraction.base import FeatureExtractor
from faro.segmentation.base import Segmentator


class MySegmentator(Segmentator):
    """TODO: return a label image, 0 = background, one integer per cell."""

    def segment(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TODO: segment the image")


class MyFeatureExtractor(FeatureExtractor):
    """One row per cell. ``labels`` is a dict keyed by segmentation name."""

    def __init__(self, used_mask: str = "labels", channel: int = 0):
        self.used_mask = used_mask
        self.channel = channel
        super().__init__()

    def extract_features(self, labels, image, df_tracked=None, metadata=None):
        table = regionprops_table(
            labels[self.used_mask],
            intensity_image=image[self.channel],
            properties=["label", "area", "mean_intensity"],
        )
        return pd.DataFrame(table), None
