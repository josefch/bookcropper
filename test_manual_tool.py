import unittest

from PIL import Image, ImageDraw

from manual_tool import trim_dark_scanner_edges


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
