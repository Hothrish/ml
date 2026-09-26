"""Tests for name, address and country normalisation.

Run from the project root:   python -m unittest discover -s tests -v

The cases are taken from real records in the challenge data rather than
invented, so a regression here corresponds to a real matching failure.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from normalization import (ALL_FIELDS, CANONICAL_COUNTRIES,  # noqa: E402
                           OUT_COLUMNS, basic_clean, byte_ranges,
                           normalize_address, normalize_country,
                           normalize_name, normalize_record)


class TestBasicClean(unittest.TestCase):
    def test_apostrophes_close_up(self):
        self.assertEqual(basic_clean("Orelee's Barbershop"), "orelees barbershop")

    def test_punctuation_becomes_space(self):
        self.assertEqual(basic_clean("PAYNE-ENRTPRMISES"), "payne enrtprmises")

    def test_ampersand_expands(self):
        self.assertEqual(basic_clean("Vasquez & Zepeda"), "vasquez and zepeda")

    def test_ordinal_suffixes_stripped(self):
        # "45th" and the corrupted "45ND" must both reduce to 45
        self.assertEqual(basic_clean("630 45th Terrace"), "630 45 terrace")
        self.assertEqual(basic_clean("630 45ND TERRACE"), "630 45 terrace")

    def test_leading_zeros_stripped(self):
        self.assertEqual(basic_clean("AF-0684"), basic_clean("AF-684"))

    def test_leading_dashes_stripped(self):
        self.assertEqual(basic_clean("-- Holloway Peak Inc"), "holloway peak inc")

    def test_empty(self):
        self.assertEqual(basic_clean(""), "")
        self.assertEqual(basic_clean(None), "")


class TestCountryNormalization(unittest.TestCase):
    def test_canonical_labels_are_stable(self):
        for c in CANONICAL_COUNTRIES:
            self.assertEqual(normalize_country(c), c)

    def test_aliases(self):
        for raw, want in [("usa", "US"), ("U.S.A.", "US"),
                          ("United States", "US"), ("bharat", "India"),
                          ("IN", "India"), ("fr", "France"),
                          ("  France  ", "France")]:
            with self.subTest(raw=raw):
                self.assertEqual(normalize_country(raw), want)

    def test_open_set_unknown_country_passes_through(self):
        # The README warns country is an open set -- the test data adds France,
        # which never appears in training, and more may follow. An unknown
        # label must be preserved, never dropped or forced into a known bucket.
        self.assertEqual(normalize_country("Germany"), "Germany")
        self.assertEqual(normalize_country("  Japan  "), "Japan")

    def test_empty(self):
        self.assertEqual(normalize_country(""), "")
        self.assertEqual(normalize_country(None), "")


class TestNameNormalization(unittest.TestCase):
    def test_legal_form_split_off(self):
        d = normalize_name("Payne Enterprises  LLC")
        self.assertEqual(d["name_core"], "payne enterprises")
        self.assertEqual(d["name_legal"], "llc")

    def test_legal_form_variants_canonicalise(self):
        for raw in ("Ss Food Private Limited", "Ss Food Pvt Ltd",
                    "Ss Food Pvt. Limited"):
            with self.subTest(raw=raw):
                d = normalize_name(raw)
                self.assertEqual(d["name_legal"], "pvtltd")
                self.assertEqual(d["name_core"], "ss food")

    def test_legal_form_not_only_at_the_end(self):
        # real record: "Maure Williams Inc Center"
        d = normalize_name("Maure Williams Inc Center")
        self.assertEqual(d["name_core"], "maure williams center")

    def test_name_of_only_legal_forms_is_not_emptied(self):
        d = normalize_name("Limited Co")
        self.assertTrue(d["name_core"])

    def test_word_order_transposition(self):
        a = normalize_name("Prime Money Corp")["name_sorted"]
        b = normalize_name("Money Prime Corp")["name_sorted"]
        self.assertEqual(a, b)

    def test_domain_form_name(self):
        d = normalize_name("maurewilliamscolombier.com")
        self.assertEqual(d["name_is_domain"], "1")
        spaced = normalize_name("Maure Williams Colombier Inc")
        self.assertEqual(d["name_nospace"], spaced["name_nospace"])

    def test_french_legal_forms(self):
        self.assertEqual(normalize_name("Marina Ecole France Sarl")["name_legal"],
                         "sarl")
        self.assertEqual(normalize_name("QHC Culture [EURL]")["name_legal"],
                         "eurl")
        self.assertEqual(normalize_name("SCI Ptit Àmicale")["name_legal"],
                         "sci")

    def test_transliterated_legal_form_is_recognised(self):
        # Devanagari "private" romanises to "praaivet"; it is matched back to
        # the legal vocabulary through its consonant skeleton
        d = normalize_name("एसएस फूड "
                           "प्राइवेट "
                           "लिमिटेड")
        self.assertEqual(d["name_legal"], "pvtltd")

    def test_all_name_fields_present_and_stringy(self):
        d = normalize_name("Anything Inc")
        for f in ("name_norm", "name_core", "name_cons", "name_script"):
            self.assertIsInstance(d[f], str)

    def test_empty_name(self):
        d = normalize_name("")
        self.assertEqual(d["name_core"], "")
        self.assertEqual(d["name_script"], "other")


class TestAddressNormalization(unittest.TestCase):
    def test_street_type_variants_neutralised(self):
        # Rd/Road, St/Street and the over-expanded "Saint" in this data all
        # vanish from the core, so no disambiguation is needed
        a = normalize_address("3315 FREMONT SAINT, PEORIA, IL", "US")
        b = normalize_address("3315 Fremont St, Peoria, Illinois", "US")
        c = normalize_address("3315 Fremont Street, Peoria, IL", "US")
        self.assertEqual(a["addr_core"], b["addr_core"])
        self.assertEqual(b["addr_core"], c["addr_core"])

    def test_component_reordering(self):
        a = normalize_address("630 45th Terrace, Kansas City, MO", "US")
        b = normalize_address("KANSAS CITY, MO, 630 45ND TERRACE, null", "US")
        self.assertEqual(a["addr_sorted"], b["addr_sorted"])
        self.assertEqual(a["addr_locality"], b["addr_locality"])
        self.assertEqual(a["addr_region"], b["addr_region"])

    def test_region_abbreviation_and_full_name_agree(self):
        a = normalize_address("Af-684, Ghaziabad, Uttar Pradesh", "India")
        b = normalize_address("AF-0684, UP, GHAZIABAD", "India")
        self.assertEqual(a["addr_region"], "up")
        self.assertEqual(a["addr_sorted"], b["addr_sorted"])

    def test_region_found_even_when_not_last(self):
        d = normalize_address("KANSAS CITY, MO, 630 45ND TERRACE", "US")
        self.assertEqual(d["addr_region"], "mo")

    def test_locality_skips_numeric_components(self):
        # "last component" lands on a PIN code as often as on the city
        d = normalize_address("AF-0684, Uttar Pradesh, GHAZIABAD, 9487203",
                              "India")
        self.assertEqual(d["addr_locality"], "ghaziabad")

    def test_landmark_extracted_not_inlined(self):
        d = normalize_address(
            "Af-684, Nandgram Near Mother India Public School, Ghaziabad, "
            "Uttar Pradesh", "India")
        self.assertIn("near", d["addr_landmark"])
        self.assertNotIn("nandgram", d["addr_core"])
        self.assertEqual(d["addr_locality"], "ghaziabad")

    def test_null_token_removed(self):
        d = normalize_address("KANSAS CITY, MO, 630 45ND TERRACE, null", "US")
        self.assertNotIn("null", d["addr_core"])

    def test_house_number(self):
        self.assertEqual(
            normalize_address("630 45th Terrace, Kansas City, MO", "US")["addr_house"],
            "630")

    def test_indian_pin_code(self):
        d = normalize_address("12 Some Road, Ghaziabad, 201001, Uttar Pradesh",
                              "India")
        self.assertEqual(d["addr_postcode"], "201001")

    def test_french_hyphenated_region(self):
        d = normalize_address("63 R. DE DIEPPE, LILLE, Hauts-de-France", "France")
        self.assertEqual(d["addr_region"], "hdf")
        self.assertEqual(d["addr_locality"], "lille")

    def test_empty_address(self):
        d = normalize_address("", "India")
        self.assertEqual(d["addr_core"], "")
        self.assertEqual(d["addr_region"], "")


class TestRecordAndSchema(unittest.TestCase):
    def test_record_has_every_declared_field(self):
        d = normalize_record("Some Co", "1 Main St, Peoria, IL", "US")
        for f in ALL_FIELDS:
            self.assertIn(f, d)
            self.assertIsInstance(d[f], str)

    def test_no_field_contains_a_tab_or_newline(self):
        # output is tab-separated; a stray tab would corrupt every later column
        d = normalize_record("A\tB Co", "1 Main St\nPeoria, IL", "US")
        for f, v in d.items():
            self.assertNotIn("\t", v, f)
            self.assertNotIn("\n", v, f)

    def test_out_columns_match_fields(self):
        self.assertEqual(OUT_COLUMNS, ("entity_id", "country") + ALL_FIELDS)

    def test_country_drives_the_address_tables(self):
        # "IN" as a component is Indiana in the US and nothing in India
        self.assertEqual(
            normalize_address("1 Main St, Gary, IN", "US")["addr_region"], "in")

    def test_cross_script_record_agreement(self):
        a = normalize_record("Modern Consultants Pvt Ltd",
                             "Bengaluru, Karnataka", "India")
        b = normalize_record(
            "ಮಾಡರ್ನ್ "
            "ಕನ್ಸಲ್ಟೆಂಟ"
            "್ಸ್ ಪ್ರೈವೇಟ"
            "್ ಲಿಮಿಟೆಡ್",
            "Bengaluru, KA", "India")
        self.assertEqual(a["name_cons"], b["name_cons"])
        self.assertEqual(a["addr_region"], b["addr_region"])
        self.assertEqual(a["addr_locality"], b["addr_locality"])


class TestByteRanges(unittest.TestCase):
    def test_ranges_are_contiguous_and_cover_the_file(self):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "_tmp_ranges.txt")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("header\n")
            for i in range(500):
                fh.write(f"row{i}\n")
        try:
            size = os.path.getsize(path)
            for n in (1, 3, 12, 64):
                with self.subTest(workers=n):
                    rs = byte_ranges(path, n)
                    self.assertEqual(rs[0][0], 0)
                    self.assertEqual(rs[-1][1], size)
                    for a, b in zip(rs, rs[1:]):
                        self.assertEqual(a[1], b[0])
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestIndicScriptRegions(unittest.TestCase):
    """Administrative divisions written in their own script.

    Regression guard: an earlier table listed a handful of native-script state
    names by hand, which covered Devanagari and Tamil and missed the other
    seven scripts entirely. These resolve through transliteration plus a
    consonant-skeleton lookup instead, so every script is covered without
    enumerating nine spellings of thirty-six divisions.
    """

    CASES = [
        ("\u0ba4\u0bae\u0bbf\u0bb4\u0bcd\u0ba8\u0bbe\u0b9f\u0bc1", "tn"),   # Tamil
        ("\u0909\u0924\u094d\u0924\u0930 \u092a\u094d\u0930\u0926\u0947\u0936", "up"),
        ("\u092e\u0939\u093e\u0930\u093e\u0937\u094d\u091f\u094d\u0930", "mh"),
        ("\u0c95\u0cb0\u0ccd\u0ca8\u0cbe\u0c9f\u0c95", "ka"),               # Kannada
        ("\u0a97\u0ac1\u0a9c\u0ab0\u0abe\u0aa4", "gj"),                     # Gujarati
        ("\u09aa\u09b6\u09cd\u099a\u09bf\u09ae\u09ac\u0999\u09cd\u0997", "wb"),
    ]

    def test_native_script_divisions_resolve(self):
        for native, code in self.CASES:
            with self.subTest(region=code):
                d = normalize_address(f"1 Main Road, Somecity, {native}", "India")
                self.assertEqual(d["addr_region"], code)

    def test_native_and_latin_spellings_agree(self):
        a = normalize_address("6(29), Chennai, \u0ba4\u0bae\u0bbf\u0bb4\u0bcd"
                              "\u0ba8\u0bbe\u0b9f\u0bc1", "India")
        b = normalize_address("6(29), Chennai, Tamil Nadu", "India")
        c = normalize_address("6(29), Chennai, TN", "India")
        self.assertEqual(a["addr_region"], b["addr_region"])
        self.assertEqual(b["addr_region"], c["addr_region"])
        self.assertEqual(a["addr_sorted"], b["addr_sorted"])

    def test_latin_component_cannot_match_a_region_by_skeleton(self):
        # the lossy skeleton lookup is restricted to non-ASCII input, so an
        # ordinary Latin street or city name cannot collide with a state
        d = normalize_address("1 Goan Street, Sometown", "India")
        self.assertEqual(d["addr_region"], "")
