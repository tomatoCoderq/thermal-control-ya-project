from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fourier import fourier_phase_features  # noqa: E402


class FourierFeatureTests(unittest.TestCase):
    def test_known_cosine_phase_is_recovered(self):
        frames = 128
        fft_bin = 5
        expected_phase = 0.73
        time = np.arange(frames, dtype=np.float32)
        curve = np.cos(2 * np.pi * fft_bin * time / frames + expected_phase)
        video = np.broadcast_to(curve, (3, 4, frames)).copy()

        features, bins, start = fourier_phase_features(
            video, first_bin=fft_bin, n_frequencies=1, row_chunk=2
        )

        self.assertEqual(start, 0)
        np.testing.assert_array_equal(bins, [fft_bin])
        self.assertEqual(features.shape, (1, 3, 4))
        phase_error = np.angle(np.exp(1j * (features[0] - expected_phase)))
        np.testing.assert_allclose(phase_error, 0.0, atol=1e-5)

    def test_sin_cos_encoding_has_two_channels_per_frequency(self):
        rng = np.random.default_rng(4)
        video = rng.normal(size=(2, 3, 64)).astype(np.float32)
        raw, _, _ = fourier_phase_features(video, n_frequencies=3)
        encoded, _, _ = fourier_phase_features(
            video, n_frequencies=3, phase_encoding="sin_cos"
        )
        self.assertEqual(encoded.shape, (6, 2, 3))
        np.testing.assert_allclose(encoded[0::2], np.sin(raw), atol=1e-6)
        np.testing.assert_allclose(encoded[1::2], np.cos(raw), atol=1e-6)

    def test_dc_bin_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "DC"):
            fourier_phase_features(np.zeros((2, 2, 16), np.float32), first_bin=0)


if __name__ == "__main__":
    unittest.main()





