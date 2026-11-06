import unittest
from nanlmind_hermes.policy import parse_n16txt, validate_duration


class PolicyTests(unittest.TestCase):
    def test_duration_bounds(self):
        self.assertEqual(validate_duration(360), 360)
        for bad in (0, 361, -1, True, 1.5, "30"):
            with self.assertRaises(ValueError):
                validate_duration(bad)

    def test_crlf_and_duplicates_canonicalize(self):
        info = parse_n16txt(b"N16TXT/1\r\nhello\r\nhello\r\nworld")
        self.assertEqual(info.records, ("hello", "world"))
        self.assertEqual(info.canonical_bytes, b"N16TXT/1\nhello\nworld\n")

    def test_rejects_bad_header_and_utf8(self):
        for raw in (b"wrong\ntext", b"N16TXT/1\n\xff"):
            with self.assertRaises(ValueError):
                parse_n16txt(raw)

    def test_rejects_long_record(self):
        with self.assertRaises(ValueError):
            parse_n16txt(b"N16TXT/1\n" + b"a" * 4097)


if __name__ == "__main__":
    unittest.main()
