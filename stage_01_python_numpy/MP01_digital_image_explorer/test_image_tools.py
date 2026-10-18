"""Lightweight independent numerical and IO checks; run with unittest."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from image_tools import (make_synthetic_image, normalize_rgb, to_grayscale,
                         histogram_u8, brighten, load_rgb_image)


class ImageToolsTests(unittest.TestCase):
    def test_fixture_known_pixels_and_shape(self):
        image = make_synthetic_image()
        self.assertEqual(image.shape, (64, 96, 3))
        self.assertEqual(image.dtype, np.uint8)
        self.assertEqual(image[0, 0].tolist(), [255, 0, 0])
        self.assertEqual(image[0, 32].tolist(), [0, 0, 255])
        self.assertEqual(image[-1, -1].tolist(), [255, 255, 64])
        np.testing.assert_array_equal(image, make_synthetic_image())

    def test_normalization_endpoints_and_midpoint(self):
        values = np.array([[[0, 128, 255]]], dtype=np.uint8)
        actual = normalize_rgb(values)
        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_allclose(actual, [[[0, 128 / 255, 1]]], atol=1e-7)

    def test_grayscale_primaries_and_neutral(self):
        image = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255], [128, 128, 128]]], dtype=np.uint8)
        actual = to_grayscale(image)
        self.assertEqual(actual.shape, (1, 4))
        np.testing.assert_allclose(actual, [[76.245, 149.685, 29.07, 128]], atol=0.001)

    def test_histogram_known_counts_and_empty_bins(self):
        actual = histogram_u8(np.array([[0, 0], [128, 255]], dtype=np.uint8))
        self.assertEqual(actual.shape, (256,))
        self.assertEqual(actual[[0, 128, 255]].tolist(), [2, 1, 1])
        self.assertEqual(actual[1], 0)
        self.assertEqual(actual.sum(), 4)

    def test_brightness_saturates_both_ends_without_mutation(self):
        image = np.array([[[250, 10, 0]]], dtype=np.uint8)
        self.assertEqual(brighten(image, 20)[0, 0].tolist(), [255, 30, 20])
        self.assertEqual(brighten(image, -20)[0, 0].tolist(), [230, 0, 0])
        self.assertEqual(image[0, 0].tolist(), [250, 10, 0])

    def test_invalid_rgb_shapes_and_dtypes_rejected(self):
        for image in [np.zeros((2, 2), dtype=np.uint8), np.zeros((2, 2, 4), dtype=np.uint8),
                      np.zeros((0, 2, 3), dtype=np.uint8), np.zeros((2, 2, 3), dtype=float)]:
            with self.subTest(shape=image.shape, dtype=image.dtype):
                with self.assertRaises(ValueError):
                    to_grayscale(image)
                with self.assertRaises(ValueError):
                    normalize_rgb(image)

    def test_invalid_histograms_and_brightness_rejected(self):
        for image in [np.array([], dtype=np.uint8).reshape(0, 1), np.zeros((2, 2)),
                      np.zeros((1, 1, 3), dtype=np.uint8)]:
            with self.assertRaises(ValueError):
                histogram_u8(image)
        for amount in [256, -256, 1.5, True]:
            with self.assertRaises(ValueError):
                brighten(make_synthetic_image(), amount)

    def test_png_roundtrip_and_grayscale_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'fixture.png'
            expected = make_synthetic_image()
            Image.fromarray(expected).save(path)
            np.testing.assert_array_equal(load_rgb_image(path), expected)
            Image.fromarray(np.array([[0, 128, 255]], dtype=np.uint8)).save(path)
            loaded = load_rgb_image(path)
            self.assertEqual(loaded.shape, (1, 3, 3))
            np.testing.assert_array_equal(loaded[0], [[0, 0, 0], [128, 128, 128], [255, 255, 255]])

    def test_palette_and_exif_orientation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'palette.png'
            palette = Image.new('P', (1, 1))
            palette.putpalette([255, 0, 0] + [0] * 765)
            palette.save(path)
            self.assertEqual(load_rgb_image(path)[0, 0].tolist(), [255, 0, 0])
            rgb = Image.fromarray(np.array([[[255, 0, 0], [0, 255, 0]]], dtype=np.uint8))
            exif = Image.Exif(); exif[274] = 6  # Rotate 90 degrees clockwise.
            rgb.save(path, exif=exif)
            loaded = load_rgb_image(path)
            self.assertEqual(loaded.shape, (2, 1, 3))
            self.assertEqual(loaded[0, 0].tolist(), [255, 0, 0])

    def test_high_depth_transparency_and_missing_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'fixture.png'
            for image in [Image.fromarray(np.array([[1024]], dtype=np.uint16)),
                          Image.new('RGBA', (2, 2), (1, 2, 3, 128))]:
                image.save(path)
                with self.assertRaises(ValueError):
                    load_rgb_image(path)
            with self.assertRaises(FileNotFoundError):
                load_rgb_image(Path(folder) / 'missing.png')


if __name__ == '__main__':
    unittest.main()
