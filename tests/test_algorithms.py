"""
Automated tests for the Smart Image Enhancer algorithms.
Run from the project folder with:   python -m unittest discover -s tests -v
(Uses only the standard library, so no extra install is needed.)
"""
import io
import os
import sys
import unittest

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import image_processing as ip      # noqa: E402
import frequency_filters as ff     # noqa: E402
import utils                       # noqa: E402

RNG = np.random.default_rng(0)


def random_gray(h=40, w=50):
    return RNG.integers(0, 256, size=(h, w), dtype=np.uint8)


def interior(a, m=3):
    return a[m:-m, m:-m]


class TestPointTransforms(unittest.TestCase):
    def test_negative_formula(self):
        img = np.array([[0, 100, 255]], dtype=np.uint8)
        np.testing.assert_array_equal(ip.negative(img), [[255, 155, 0]])

    def test_negative_twice_is_identity(self):
        img = random_gray()
        np.testing.assert_array_equal(ip.negative(ip.negative(img)), img)

    def test_gamma_one_is_identity(self):
        img = random_gray()
        np.testing.assert_array_equal(ip.gamma_transform(img, 1.0), img)

    def test_gamma_known_value(self):
        # 255 * (128/255)^2 = 64.25 -> 64
        self.assertEqual(ip.gamma_transform(np.array([[128]], dtype=np.uint8), 2.0)[0, 0], 64)

    def test_gamma_direction(self):
        mid = np.array([[100]], dtype=np.uint8)
        self.assertGreater(ip.gamma_transform(mid, 0.5)[0, 0], 100)   # brightens
        self.assertLess(ip.gamma_transform(mid, 2.0)[0, 0], 100)      # darkens

    def test_gamma_invalid(self):
        for g in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                ip.gamma_transform(random_gray(), g)

    def test_log_endpoints_and_monotonic(self):
        img = np.arange(256, dtype=np.uint8).reshape(16, 16)
        out = ip.log_transform(img).ravel()
        self.assertEqual(out[0], 0)
        self.assertEqual(out[255], 255)
        self.assertTrue(np.all(np.diff(out.astype(int)) >= 0))

    def test_contrast_stretch_range(self):
        img = RNG.integers(50, 101, size=(30, 30), dtype=np.uint8)
        img[0, 0], img[0, 1] = 50, 100
        out = ip.contrast_stretch(img)
        self.assertEqual(out.min(), 0)
        self.assertEqual(out.max(), 255)

    def test_contrast_stretch_constant_unchanged(self):
        img = np.full((10, 10), 77, dtype=np.uint8)
        np.testing.assert_array_equal(ip.contrast_stretch(img), img)

    def test_contrast_stretch_bad_percentiles(self):
        with self.assertRaises(ValueError):
            ip.contrast_stretch(random_gray(), 60, 40)

    def test_hist_eq_known_case(self):
        img = np.array([[0, 0], [255, 255]], dtype=np.uint8)
        np.testing.assert_array_equal(ip.histogram_equalization(img), img)

    def test_hist_eq_widens_low_contrast_and_keeps_order(self):
        img = RNG.integers(100, 121, size=(40, 40), dtype=np.uint8)
        out = ip.histogram_equalization(img)
        self.assertEqual(out.min(), 0)
        self.assertEqual(out.max(), 255)
        # the mapping must be non-decreasing: brighter input never becomes darker
        mapping = {int(v): int(out[img == v][0]) for v in np.unique(img)}
        vals = [mapping[k] for k in sorted(mapping)]
        self.assertEqual(vals, sorted(vals))

    def test_hist_eq_constant_unchanged(self):
        img = np.full((8, 8), 30, dtype=np.uint8)
        np.testing.assert_array_equal(ip.histogram_equalization(img), img)

    def test_hist_eq_colour_shape(self):
        img = RNG.integers(0, 256, size=(20, 20, 3), dtype=np.uint8)
        self.assertEqual(ip.histogram_equalization(img).shape, img.shape)

    def test_original_not_modified(self):
        img = random_gray()
        copy = img.copy()
        for f in (ip.negative, ip.histogram_equalization, ip.contrast_stretch, ip.mean_filter, ip.median_filter):
            f(img)
        np.testing.assert_array_equal(img, copy)

    def test_slicing(self):
        img = np.array([[10, 50, 100, 200]], dtype=np.uint8)
        np.testing.assert_array_equal(ip.gray_level_slicing(img, 40, 120, 255, True), [[10, 255, 255, 200]])
        np.testing.assert_array_equal(ip.gray_level_slicing(img, 40, 120, 255, False), [[0, 255, 255, 0]])
        with self.assertRaises(ValueError):
            ip.gray_level_slicing(img, 120, 40)
        with self.assertRaises(ValueError):
            ip.gray_level_slicing(img, -1, 40)

    def test_bit_planes(self):
        img = np.array([[0b10000001, 0b01111110]], dtype=np.uint8)
        np.testing.assert_array_equal(ip.bit_plane_slice(img, 7), [[255, 0]])
        np.testing.assert_array_equal(ip.bit_plane_slice(img, 0), [[255, 0]])
        np.testing.assert_array_equal(ip.bit_plane_slice(img, 1), [[0, 255]])
        with self.assertRaises(ValueError):
            ip.bit_plane_slice(img, 8)

    def test_bit_plane_reconstruction_with_all_planes(self):
        img = random_gray()
        np.testing.assert_array_equal(ip.bit_plane_reconstruct(img, range(8)), img)

    def test_salt_pepper(self):
        img = np.full((100, 100), 128, dtype=np.uint8)
        noisy = ip.add_salt_and_pepper(img, 0.1, seed=1)
        frac = np.mean(noisy != 128)
        self.assertTrue(0.07 < frac < 0.13)
        self.assertTrue(set(np.unique(noisy)) <= {0, 128, 255})
        np.testing.assert_array_equal(noisy, ip.add_salt_and_pepper(img, 0.1, seed=1))   # repeatable
        with self.assertRaises(ValueError):
            ip.add_salt_and_pepper(img, 1.5)


class TestSpatialFilters(unittest.TestCase):
    def test_mean_matches_opencv(self):
        img = random_gray()
        ref = cv2.blur(img, (3, 3), borderType=cv2.BORDER_REFLECT_101)
        self.assertLessEqual(np.abs(ip.mean_filter(img, 3).astype(int) - ref.astype(int)).max(), 1)

    def test_mean_known_value(self):
        img = np.arange(1, 10, dtype=np.uint8).reshape(3, 3)
        self.assertEqual(ip.mean_filter(img, 3)[1, 1], 5)

    def test_filters_keep_constant_image(self):
        img = np.full((12, 12), 90, dtype=np.uint8)
        for name in ip.SPATIAL_FILTERS:
            np.testing.assert_array_equal(ip.apply_named_filter(img, name, 3, 2), img, err_msg=name)

    def test_median_matches_opencv_interior(self):
        img = random_gray()
        np.testing.assert_array_equal(interior(ip.median_filter(img, 3)), interior(cv2.medianBlur(img, 3)))
        np.testing.assert_array_equal(interior(ip.median_filter(img, 5)), interior(cv2.medianBlur(img, 5)))

    def test_median_removes_single_spike(self):
        img = np.full((9, 9), 100, dtype=np.uint8)
        img[4, 4] = 255
        np.testing.assert_array_equal(ip.median_filter(img, 3), np.full((9, 9), 100))

    def test_max_min_match_opencv_interior(self):
        img = random_gray()
        kern = np.ones((3, 3), np.uint8)
        np.testing.assert_array_equal(interior(ip.max_filter(img, 3)), interior(cv2.dilate(img, kern)))
        np.testing.assert_array_equal(interior(ip.min_filter(img, 3)), interior(cv2.erode(img, kern)))

    def test_midpoint(self):
        img = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 100]], dtype=np.uint8)
        self.assertEqual(ip.midpoint_filter(img, 3)[1, 1], 50)

    def test_alpha_trimmed_limits(self):
        img = random_gray()
        np.testing.assert_array_equal(ip.alpha_trimmed_mean_filter(img, 3, 0), ip.mean_filter(img, 3))
        np.testing.assert_array_equal(ip.alpha_trimmed_mean_filter(img, 3, 8), ip.median_filter(img, 3))
        np.testing.assert_array_equal(ip.alpha_trimmed_mean_filter(img, 5, 24), ip.median_filter(img, 5))

    def test_alpha_trimmed_rejects_outlier(self):
        img = np.full((9, 9), 100, dtype=np.uint8)
        img[4, 4] = 255
        np.testing.assert_array_equal(ip.alpha_trimmed_mean_filter(img, 3, 2), np.full((9, 9), 100))

    def test_alpha_trimmed_validation(self):
        img = random_gray()
        for k, d in ((3, 1), (3, 9), (3, 10), (3, -2), (5, 25), (4, 2), (3, 2.5)):
            with self.assertRaises(ValueError, msg=f"k={k}, d={d}"):
                ip.alpha_trimmed_mean_filter(img, k, d)
        self.assertEqual(ip.max_alpha_trim(5), 24)

    def test_weighted_kernel(self):
        k3 = ip.weighted_mean_kernel(3)
        np.testing.assert_allclose(k3, np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16.0)
        self.assertAlmostEqual(ip.weighted_mean_kernel(5).sum(), 1.0)
        self.assertAlmostEqual(ip.weighted_mean_kernel(5)[2, 2], 36 / 256)

    def test_invalid_kernel_size(self):
        for k in (1, 2, 4, 0, -3):
            with self.assertRaises(ValueError):
                ip.mean_filter(random_gray(), k)

    def test_colour_filter_matches_per_channel(self):
        img = RNG.integers(0, 256, size=(15, 15, 3), dtype=np.uint8)
        out = ip.median_filter(img, 3)
        for c in range(3):
            np.testing.assert_array_equal(out[..., c], ip.median_filter(img[..., c], 3))

    def test_median_beats_mean_on_salt_pepper(self):
        clean = cv2.GaussianBlur(random_gray(60, 60), (9, 9), 3)
        noisy = ip.add_salt_and_pepper(clean, 0.1, seed=3)
        e_mean = utils.mse(clean, ip.mean_filter(noisy, 3))
        e_med = utils.mse(clean, ip.median_filter(noisy, 3))
        self.assertLess(e_med, e_mean)

    def test_tiny_images(self):
        for shape in ((1, 1), (2, 2), (1, 7), (7, 1), (3, 3)):
            img = RNG.integers(0, 256, size=shape, dtype=np.uint8)
            for name in ip.SPATIAL_FILTERS:
                self.assertEqual(ip.apply_named_filter(img, name, 5, 4).shape, shape, msg=f"{name} {shape}")
            self.assertEqual(ip.sobel_magnitude(img).shape, shape)
            self.assertEqual(ip.laplacian_edges(img).shape, shape)


class TestEdges(unittest.TestCase):
    def test_sobel_on_ramp(self):
        x = (np.arange(20) * 10).astype(np.uint8)
        img = np.tile(x, (20, 1))                         # intensity grows left -> right
        gx, gy = ip.sobel_gradients(img)
        self.assertTrue(np.all(gx[2:-2, 2:-2] == 80))     # (1+2+1) * (20) = 80
        self.assertTrue(np.all(gy[2:-2, 2:-2] == 0))

    def test_sobel_y_on_horizontal_ramp(self):
        img = np.tile((np.arange(20) * 10).astype(np.uint8)[:, None], (1, 20))
        gx, gy = ip.sobel_gradients(img)
        self.assertTrue(np.all(gy[2:-2, 2:-2] == 80))
        self.assertTrue(np.all(gx[2:-2, 2:-2] == 0))

    def test_constant_image_has_no_edges(self):
        img = np.full((10, 10), 120, dtype=np.uint8)
        for f in (ip.sobel_x, ip.sobel_y, ip.sobel_magnitude, ip.laplacian_edges):
            self.assertEqual(f(img).max(), 0)

    def test_sobel_step_edge_location(self):
        img = np.zeros((10, 10), dtype=np.uint8)
        img[:, 5:] = 200
        mag = ip.sobel_magnitude(img)
        self.assertIn(int(np.argmax(mag[5])), (4, 5))      # strongest response right at the step
        self.assertEqual(mag[5, 0], 0)

    def test_laplacian_of_linear_ramp_is_zero(self):
        img = np.tile((np.arange(20) * 5).astype(np.uint8), (20, 1))
        self.assertTrue(np.all(ip.laplacian_response(img)[2:-2, 2:-2] == 0))

    def test_laplacian_point(self):
        img = np.zeros((5, 5), dtype=np.uint8)
        img[2, 2] = 10
        resp = ip.laplacian_response(img)
        self.assertEqual(resp[2, 2], -40)
        self.assertEqual(resp[2, 1], 10)

    def test_sharpening_leaves_constant_unchanged(self):
        img = np.full((12, 12), 90, dtype=np.uint8)
        np.testing.assert_array_equal(ip.laplacian_sharpen(img), img)
        np.testing.assert_array_equal(ip.unsharp_mask(img), img)
        np.testing.assert_array_equal(ip.high_boost(img, 1.5, 3.0), img)

    def test_sharpening_increases_edge_contrast(self):
        img = np.zeros((20, 20), dtype=np.uint8)
        img[:, 10:] = 100
        sharp = ip.unsharp_mask(img, 1.5, 1.0)
        self.assertGreater(int(sharp[10, 10]), 100)      # overshoot on the bright side
        self.assertEqual(int(sharp[10, 9]), 0)           # undershoot is clipped at 0

    def test_high_boost_stronger_than_unsharp(self):
        img = np.zeros((20, 20), dtype=np.uint8)
        img[:, 10:] = 100
        self.assertGreaterEqual(int(ip.high_boost(img, 1.5, 3.0)[10, 10]), int(ip.unsharp_mask(img, 1.5, 1.0)[10, 10]))

    def test_high_boost_validation(self):
        with self.assertRaises(ValueError):
            ip.high_boost(random_gray(), 1.5, 1.0)
        with self.assertRaises(ValueError):
            ip.unsharp_mask(random_gray(), 0)


class TestFrequencyDomain(unittest.TestCase):
    def test_fft_roundtrip(self):
        self.assertLess(ff.reconstruction_error(random_gray(33, 47)), 1e-9)

    def test_identity_when_cutoff_covers_everything(self):
        img = random_gray(30, 41)
        for pad in (True, False):
            out = ff.apply_frequency_filter(img, "ideal", "lowpass", 10_000, use_padding=pad)
            np.testing.assert_array_equal(out, img)

    def test_lowpass_preserves_constant_and_highpass_removes_it(self):
        img = np.full((20, 20), 100, dtype=np.uint8)
        for fam in ff.FAMILIES:
            lp = ff.frequency_filter_float(img, fam, "lowpass", 5, use_padding=False)
            hp = ff.frequency_filter_float(img, fam, "highpass", 5, use_padding=False)
            np.testing.assert_allclose(lp, 100.0, atol=1e-6)
            np.testing.assert_allclose(hp, 0.0, atol=1e-6)

    def test_transfer_function_values(self):
        H = ff.transfer_function((64, 64), "butterworth", "lowpass", 10, 2)
        self.assertAlmostEqual(H[32, 32], 1.0)                 # centre = DC
        self.assertAlmostEqual(H[32, 42], 0.5)                 # D = D0 -> 0.5
        # at D = 2*D0 the order matters: 1 / (1 + 2^(2n))
        self.assertAlmostEqual(H[32, 52], 1 / 17.0)
        H3 = ff.transfer_function((64, 64), "butterworth", "lowpass", 10, 3)
        self.assertAlmostEqual(H3[32, 52], 1 / 65.0)
        G = ff.transfer_function((64, 64), "gaussian", "lowpass", 10)
        self.assertAlmostEqual(G[32, 42], np.exp(-0.5))
        I = ff.transfer_function((64, 64), "ideal", "lowpass", 10)
        self.assertEqual(I[32, 42], 1.0)
        self.assertEqual(I[32, 43], 0.0)

    def test_highpass_is_complement(self):
        for fam in ff.FAMILIES:
            lp = ff.transfer_function((50, 60), fam, "lowpass", 7, 3)
            hp = ff.transfer_function((50, 60), fam, "highpass", 7, 3)
            np.testing.assert_allclose(lp + hp, 1.0)

    def test_filtered_output_is_real_valued_and_correct_size(self):
        img = random_gray(31, 45)      # odd sizes on purpose
        out = ff.frequency_filter_float(img, "gaussian", "lowpass", 8)
        self.assertEqual(out.shape, img.shape)
        self.assertTrue(np.isrealobj(out))

    def test_lowpass_smooths_noise_and_highpass_keeps_it(self):
        img = random_gray(64, 64)
        lp = ff.apply_frequency_filter(img, "gaussian", "lowpass", 6)
        self.assertLess(lp.std(), img.std())
        hp = ff.frequency_filter_float(img, "gaussian", "highpass", 6)
        self.assertGreater(hp.std(), 10)          # high-pass keeps the strong fine detail of random noise

    def test_lowpass_equals_spatial_circular_blur_without_padding(self):
        """Convolution theorem: multiplying by H in frequency = circular convolution in space."""
        img = random_gray(32, 32).astype(np.float64)
        H = ff.transfer_function((32, 32), "gaussian", "lowpass", 4)
        via_fft = np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(img)) * H)).real
        h_spatial = np.fft.ifft2(np.fft.ifftshift(H)).real          # impulse response
        direct = np.zeros_like(img)
        for dy in range(32):                                       # circular convolution done by brute force
            for dx in range(32):
                direct += img[(np.arange(32)[:, None] - dy) % 32, (np.arange(32)[None, :] - dx) % 32] * h_spatial[dy, dx]
        np.testing.assert_allclose(via_fft, direct, atol=1e-6)

    def test_validation(self):
        with self.assertRaises(ValueError):
            ff.transfer_function((10, 10), "ideal", "lowpass", 0)
        with self.assertRaises(ValueError):
            ff.transfer_function((10, 10), "butterworth", "lowpass", 5, 0)
        with self.assertRaises(ValueError):
            ff.transfer_function((10, 10), "median", "lowpass", 5)

    def test_colour_and_display_scaling(self):
        img = RNG.integers(0, 256, size=(20, 24, 3), dtype=np.uint8)
        out = ff.apply_frequency_filter(img, "butterworth", "highpass", 5, 2)
        self.assertEqual(out.shape, img.shape)
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(out.min(), 0)
        self.assertEqual(out.max(), 255)

    def test_spectrum_peak_at_centre(self):
        spec = ff.fft_spectrum(random_gray(32, 32))
        self.assertEqual(spec.dtype, np.uint8)
        self.assertEqual(np.unravel_index(np.argmax(spec), spec.shape), (16, 16))


class TestUtils(unittest.TestCase):
    def test_mse_psnr(self):
        a = np.zeros((4, 4), dtype=np.uint8)
        b = np.full((4, 4), 10, dtype=np.uint8)
        self.assertEqual(utils.mse(a, a), 0.0)
        self.assertEqual(utils.psnr(a, a), float("inf"))
        self.assertAlmostEqual(utils.mse(a, b), 100.0)
        self.assertAlmostEqual(utils.psnr(a, b), 10 * np.log10(255 ** 2 / 100.0))

    def test_mse_no_uint8_wraparound(self):
        a = np.zeros((2, 2), dtype=np.uint8)
        b = np.full((2, 2), 255, dtype=np.uint8)
        self.assertAlmostEqual(utils.mse(a, b), 255.0 ** 2)

    def test_mse_shape_mismatch(self):
        with self.assertRaises(ValueError):
            utils.mse(np.zeros((4, 4), np.uint8), np.zeros((4, 5), np.uint8))

    @staticmethod
    def _png_bytes(mode, size=(10, 6), color=None):
        buf = io.BytesIO()
        Image.new(mode, size, color).save(buf, format="PNG")
        return buf.getvalue()

    def test_load_valid_images(self):
        for mode, color in (("RGB", (10, 20, 30)), ("L", 100), ("RGBA", (10, 20, 30, 128)), ("P", 3)):
            arr, info = utils.load_image(self._png_bytes(mode, color=color), "x.png")
            self.assertEqual(arr.shape, (6, 10, 3), msg=mode)
            self.assertEqual(arr.dtype, np.uint8)

    def test_load_rejects_bad_files(self):
        with self.assertRaises(ValueError):
            utils.load_image(b"this is not an image", "fake.jpg")
        with self.assertRaises(ValueError):
            utils.load_image(b"", "empty.png")
        with self.assertRaises(ValueError):
            utils.load_image(self._png_bytes("RGB"), "photo.gif")

    def test_limit_size(self):
        big = np.zeros((1600, 400, 3), dtype=np.uint8)
        small, resized = utils.limit_size(big, 800)
        self.assertTrue(resized)
        self.assertEqual(small.shape, (800, 200, 3))
        same, resized = utils.limit_size(small, 800)
        self.assertFalse(resized)

    def test_png_roundtrip(self):
        img = random_gray(8, 9)
        back = np.array(Image.open(io.BytesIO(utils.to_png_bytes(img))))
        np.testing.assert_array_equal(back, img)

    def test_histogram_figure_and_description(self):
        gray = random_gray()
        colour = RNG.integers(0, 256, size=(10, 10, 3), dtype=np.uint8)
        self.assertIsNotNone(utils.histogram_figure(gray))
        self.assertIsNotNone(utils.histogram_figure(colour))
        self.assertIn("single spike", utils.describe_histogram(np.full((5, 5), 9, dtype=np.uint8)))
        self.assertEqual(utils.make_sample_image().shape, (360, 480, 3))


if __name__ == "__main__":
    unittest.main(verbosity=2)
