import unittest

from PIL import Image, ImageDraw

from manual_tool import colorchecker_correction, trim_dark_scanner_edges


class ColorCheckerCorrectionTest(unittest.TestCase):
    def test_preserves_channel_order_in_cyan_and_violet_hue_sectors(self):
        image = Image.new("RGB", (2, 1))
        image.putdata([(20, 100, 80), (80, 20, 100)])

        corrected = colorchecker_correction(image)
        cyan, violet = list(corrected.getdata())

        self.assertGreater(cyan[1], cyan[2])
        self.assertGreater(cyan[2], cyan[0])
        self.assertGreater(violet[2], violet[0])
        self.assertGreater(violet[0], violet[1])


class TrimDarkScannerEdgesTest(unittest.TestCase):
    def test_trims_thin_black_scanner_margin(self):
        image = Image.new("RGB", (1000, 800), "black")
        ImageDraw.Draw(image).rectangle((6, 5, 993, 794), fill=(230, 225, 210))

        trimmed = trim_dark_scanner_edges(image)

        self.assertEqual(trimmed.size, (988, 790))

    def test_keeps_genuinely_dark_cover(self):
        image = Image.new("RGB", (200, 150), (8, 8, 8))
        ImageDraw.Draw(image).text((70, 65), "TITLE", fill="white")

        trimmed = trim_dark_scanner_edges(image)

        self.assertEqual(trimmed.size, image.size)

    def test_keeps_clean_light_crop(self):
        image = Image.new("RGB", (200, 150), (230, 225, 210))

        trimmed = trim_dark_scanner_edges(image)

        self.assertEqual(trimmed.size, image.size)


if __name__ == "__main__":
    unittest.main()
