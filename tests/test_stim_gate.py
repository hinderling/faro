"""Regression test for the feed-loop stim gate (previous-mode acquisition order).

The Controller feed loop runs several events ahead of the camera (qsize<3
backpressure). Without a gate it would build a stim event's SLM -- blocking
``get_stim_mask`` -- for a predecessor frame that has not been acquired yet;
on real hardware (minute-scale intervals + slow segmentation) the mask wait
then times out before the frame even exists, and the DMD fires its stale
pattern (the "first stim is all-on" bug's deeper cause).

``Controller._wait_for_frame_acquired`` gates the build on the predecessor
frame's acquisition, so ``get_stim_mask`` only ever waits out segmentation of
an already-acquired frame. This test drives the *real* feed loop through the
fake microscope with a deliberately slow camera (so acquisition lags the feed
loop) and asserts every stim SLM is built only after its predecessor frame is
acquired, with no background errors.

Self-contained on purpose: it builds its own scene/pipeline (OtsuSegmentator +
TrackerTrackpy + CenterCircle) instead of importing ``tests.fixtures``, so it
does not pull in the optional ``motile`` tracker.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from faro.core.controller import Controller
from faro.core.data_structures import Channel, RTMEvent, SegmentationMethod
from faro.core.pipeline import ImageProcessingPipeline
from faro.feature_extraction.simple import SimpleFE
from faro.segmentation.base import OtsuSegmentator
from faro.stimulation.center_circle import CenterCircle
from faro.tracking.trackpy import TrackerTrackpy

from tests.fake_microscope import FakeMicroscope

IMG = 256
N_T = 6
STIM_FRAMES = (2, 3, 4, 5)
CAM_DELAY = 0.20   # slow camera -> acquisition lags the feed loop
SEG_DELAY = 0.15   # segmentation latency (stardist-warmup stand-in)


def _circle_image() -> np.ndarray:
    img = np.zeros((IMG, IMG), np.uint16)
    y, x = np.ogrid[:IMG, :IMG]
    img[(y - 64) ** 2 + (x - 64) ** 2 <= 20**2] = 50_000
    img[(y - 192) ** 2 + (x - 192) ** 2 <= 15**2] = 50_000
    return img


class _SlowCircleScene:
    image_height = IMG
    image_width = IMG
    channels = ("phase-contrast", "stim-405")
    slm_name = "SLM"
    slm_shape = (IMG, IMG)

    def __init__(self):
        self.slm_events: list[tuple[int, np.ndarray]] = []

    def render(self, event):
        time.sleep(CAM_DELAY)
        return _circle_image()

    def on_slm_displayed(self, image, event):
        self.slm_events.append((event.index.get("t", 0), np.asarray(image)))


class _SlowOtsu(OtsuSegmentator):
    def segment(self, *a, **k):
        time.sleep(SEG_DELAY)
        return super().segment(*a, **k)


def _make_events():
    stim_ch = (Channel(config="stim-405", exposure=100),)
    return [
        RTMEvent(
            index={"t": t, "p": 0},
            channels=(Channel(config="phase-contrast", exposure=50),),
            stim_channels=stim_ch if t in STIM_FRAMES else (),
            metadata={},
        )
        for t in range(N_T)
    ]


def test_stim_slm_built_only_after_predecessor_acquired(tmp_path):
    pipeline = ImageProcessingPipeline(
        storage_path=str(tmp_path),
        segmentators=[SegmentationMethod("labels", _SlowOtsu(), 0, False)],
        tracker=TrackerTrackpy(search_range=50, memory=3),
        feature_extractor=SimpleFE("labels"),
        stimulator=CenterCircle(),
    )
    mic = FakeMicroscope(_SlowCircleScene())
    ctrl = Controller(mic, pipeline)

    # Record, at each stim-SLM build, whether the predecessor (t-1, p) is
    # already acquired -- the exact contract the gate must enforce.
    orig = ctrl._build_stim_slm
    records: list[tuple[int, int, bool]] = []

    def _wrapped(rtm_event, *, stim_mode="current"):
        t = rtm_event.index.get("t", 0)
        p = rtm_event.index.get("p", 0)
        with ctrl._acquired_lock:
            pred_acquired = (t - 1, p) in ctrl._acquired_frames
        records.append((t, p, pred_acquired))
        return orig(rtm_event, stim_mode=stim_mode)

    ctrl._build_stim_slm = _wrapped

    handle = ctrl.run_experiment(_make_events(), stim_mode="previous", validate=False)
    handle.wait()
    ctrl._analyzer.wait_idle(timeout=120)
    ctrl._analyzer.shutdown(wait=True)

    assert not ctrl.background_errors, (
        "background errors during acquisition: "
        + "; ".join(f"[{e.source}] {e.exc_type}: {e.message}" for e in ctrl.background_errors)
    )
    assert [r[0] for r in records] == list(STIM_FRAMES), (
        f"expected a stim build per stim frame, got {records}"
    )
    early = [(t, p) for t, p, ok in records if not ok]
    assert not early, f"stim SLM built before predecessor acquired for {early}"
    assert len(mic.scene.slm_events) >= len(STIM_FRAMES)
