"""Tests for script detection and cross-script normalisation views.

Run from the project root:   python -m unittest discover -s tests -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from script_classifier import (ALL_SCRIPTS, INDIC_SCRIPTS,  # noqa: E402
                               consonant_skeleton, fold_unicode, phonetic,
                               script_distribution, script_of,
                               scripts_present, transliterate)


class TestScriptDetection(unittest.TestCase):
    def test_latin(self):
        self.assertEqual(script_of("Payne Enterprises"), "latin")
        self.assertEqual(script_of("SCI Ptit Amicale"), "latin")

    def test_accented_latin_is_still_latin(self):
        self.assertEqual(script_of("Payne Énterprises"), "latin")

    def test_each_indic_script_is_recognised(self):
        cases = {
            "deva": "एसएस फूड",
            "beng": "গোল্ড",
            "guru": "ਸਕਾਈ",
            "gujr": "શક્તિ",
            "orya": "ଭିଜନ୍",
            "taml": "ராஜ்",
            "telu": "పర్ఫెక్ట్",
            "knda": "ಮಾಡರ್ನ್",
            "mlym": "ശക്തി",
        }
        for tag, text in cases.items():
            with self.subTest(script=tag):
                self.assertEqual(script_of(text), tag)
        self.assertEqual(set(cases), set(INDIC_SCRIPTS))

    def test_mixed_script(self):
        # a Latin brand name beside an Indic legal suffix
        self.assertEqual(
            script_of("Raj Investments எல்எல்பி"),
            "mixed")

    def test_other_is_only_for_genuinely_non_alphabetic(self):
        self.assertEqual(script_of(""), "other")
        self.assertEqual(script_of("12345"), "other")
        self.assertEqual(script_of("--- ###"), "other")

    def test_returns_only_known_tags(self):
        for s in ("abc", "", "123", "పర"):
            self.assertIn(script_of(s), ALL_SCRIPTS)

    def test_scripts_present_diagnoses_mixed(self):
        got = scripts_present("Raj कंपनी")
        self.assertEqual(got, {"latin", "deva"})


class TestScriptDistributionRegression(unittest.TestCase):
    """Guards the two ways this report degenerated to 100% 'other'."""

    def test_rejects_a_derived_column(self):
        # Cause 1: classifying a POST-normalisation column. name_nums is
        # digits-only -> 100% 'other'; name_translit is empty for Latin
        # records -> ~94% 'other'. script_of() must see the raw name.
        with self.assertRaises(ValueError):
            script_distribution(["ignored.tsv"], column="name_nums")
        with self.assertRaises(ValueError):
            script_distribution(["ignored.tsv"], column="addr_core")

    def test_non_latin_scripts_are_not_bucketed_as_other(self):
        # Cause 2: an incomplete script table. Every Brahmic script in this
        # dataset must classify as itself, never as 'other'.
        samples = [
            "పర్ఫెక్ట్",   # Telugu
            "ಮಾಡರ್ನ್",               # Kannada
            "গোল্ড",                           # Bengali
            "શક્તિ",                           # Gujarati
            "ശക്തി",                           # Malayalam
            "ଭିଜନ୍",                           # Odia
            "ਸਕਾਈ",                                 # Gurmukhi
        ]
        for s in samples:
            with self.subTest(text=s):
                self.assertNotEqual(script_of(s), "other")


class TestTransliteration(unittest.TestCase):
    def test_latin_passes_through(self):
        self.assertEqual(transliterate("Payne Enterprises"),
                         "Payne Enterprises")

    def test_indic_acronym_decomposition(self):
        # Devanagari for "SS" is the letter-name for S twice, not a word
        self.assertEqual(transliterate("एसएस"), "ss")
        # Tamil LLP
        self.assertEqual(
            transliterate("எல்எல்பி"),
            "llp")

    def test_two_part_vowel_signs_survive_folding(self):
        # Odia writes "o" as an e-sign plus an aa-sign. NFKD splits them and
        # the scanner consumes one vowel per syllable, so without the NFC
        # recomposition in fold_unicode this reads "e" and the word breaks.
        odia = "ଟେକ୍ନୋଲୋଜିସ୍"
        out = transliterate(fold_unicode(odia))
        self.assertIn("o", out)
        self.assertTrue(out.startswith("tekno"), out)

    def test_malayalam_chillu_and_rra_geminate(self):
        # RRA+virama+RRA is pronounced "tt", not "rr"
        mlym = "ലിമിറ്റഡ്"
        out = transliterate(fold_unicode(mlym))
        self.assertNotIn("rr", out)
        self.assertIn("limitt", out)


class TestCrossScriptViews(unittest.TestCase):
    """The consonant skeleton is the representation that must survive
    transliteration -- it is what blocking across scripts relies on."""

    CASES = [
        ("एसएस फूड प्रा"
         "इवेट लिमिटेड",
         "Ss Food Private Limited"),
        ("গোল্ড প্রডিউ"
         "সার স্টোর্স",
         "Gold Producer Stores"),
        ("ಮಾಡರ್ನ್ ಕನ್"
         "ಸಲ್ಟೆಂಟ್ಸ್",
         "Modern Consultants"),
        ("ராஜ் இன்வெஸ்"
         "ட்மெண்ட்ஸ்",
         "Raj Investments"),
    ]

    def test_skeletons_agree_across_scripts(self):
        for indic, latin in self.CASES:
            with self.subTest(latin=latin):
                a = set(consonant_skeleton(
                    transliterate(fold_unicode(indic))).split())
                b = set(consonant_skeleton(fold_unicode(latin)).split())
                overlap = len(a & b) / max(1, len(a | b))
                self.assertGreaterEqual(
                    overlap, 0.6, f"{sorted(a)} vs {sorted(b)}")

    def test_leading_vowel_is_normalised(self):
        # transliteration is unreliable about vowel quality, so a vowel-initial
        # token must not diverge on its first character alone
        self.assertEqual(consonant_skeleton("east"), consonant_skeleton("iisat"))

    def test_typos_are_left_to_fuzzy_matching(self):
        # random corruption is explicitly NOT the normaliser's job
        self.assertNotEqual(consonant_skeleton("Payne Etrepndiels"),
                            consonant_skeleton("Payne Enterprises"))


class TestPhonetics(unittest.TestCase):
    def test_aspirates_and_vowel_length_collapse(self):
        # phonetic() folds the aspirate (ph -> f) and the doubled vowel
        # (uu -> u), but deliberately keeps vowel *identity*: u is not o.
        self.assertEqual(phonetic("phuud"), "fud")
        self.assertEqual(phonetic("food"), "fod")

    def test_only_the_skeleton_is_vowel_insensitive(self):
        # Devanagari romanises "food" as "phuud"; the two agree only once
        # vowels are dropped, which is why blocking uses the skeleton.
        self.assertNotEqual(phonetic("phuud"), phonetic("food"))
        self.assertEqual(consonant_skeleton("phuud"),
                         consonant_skeleton("food"))

    def test_soft_c(self):
        self.assertEqual(consonant_skeleton("finance"), consonant_skeleton("finanse"))

    def test_empty(self):
        self.assertEqual(consonant_skeleton(""), "")
        self.assertEqual(phonetic(""), "")


class TestFolding(unittest.TestCase):
    def test_ascii_untouched(self):
        self.assertEqual(fold_unicode("Plain ASCII 123"), "Plain ASCII 123")

    def test_latin_diacritics_removed(self):
        self.assertEqual(fold_unicode("Payne Énterprises"),
                         "Payne Enterprises")
        self.assertEqual(fold_unicode("SCI Ptit Àmicale"),
                         "SCI Ptit Amicale")

    def test_indic_marks_preserved(self):
        # the virama is a combining mark but must survive: stripping it
        # destroys the word before it can be transliterated
        deva = "प्राइवेट"
        self.assertIn("्", fold_unicode(deva))


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestNoRawCodepointsLeak(unittest.TestCase):
    """Regression guard for two real records found by running the pipeline on
    a small sample: Gurmukhi's nasalisation mark (TIPPI, U+0A70) sits outside
    the Devanagari-offset position, and Odia's YYA (U+0B5F) has no Devanagari
    letter to derive from at all. Both fell through the syllable scanner
    untranslated and leaked a raw Indic codepoint into the output.
    """

    def test_gurmukhi_tippi_is_translated(self):
        # real record: "ਸਕਾਈ ਅਰਿਹੰਤ ਗਲੋਬਲ ਪ੍ਰਾ. ਲਿ."
        s = ("\u0a38\u0a15\u0a3e\u0a08 \u0a05\u0a30\u0a3f\u0a39\u0a70\u0a24 "
             "\u0a17\u0a32\u0a4b\u0a2c\u0a32")
        out = transliterate(fold_unicode(s))
        self.assertNotIn("\u0a70", out)
        self.assertIn("arihant", out)

    def test_odia_yya_is_translated(self):
        # real record: "ବିଜୟ କନଷ୍ଟ୍ରକସନ୍ସ୍ ପ୍ରାଇଭେଟ୍ ଲିମିଟେଡ୍"
        s = "\u0b2c\u0b3f\u0b1c\u0b5f"
        out = transliterate(fold_unicode(s))
        self.assertNotIn("\u0b5f", out)
        self.assertIn("y", out)

    def test_no_script_character_survives_transliteration(self):
        # general safety net: for every script this classifier claims to
        # support, transliterate() must not leave any character of that
        # script's own block in its output. This is the check that would
        # have caught both leaks above without needing the specific example.
        from script_classifier import SCRIPT_BLOCKS
        samples = {
            "deva": "\u090f\u0938\u090f\u0938 \u092b\u0942\u0921 \u092a\u094d\u0930\u093e"
                    "\u0907\u0935\u0947\u091f \u0932\u093f\u092e\u093f\u091f\u0947\u0921",
            "beng": "\u0997\u09cb\u09b2\u09cd\u09a1 \u09aa\u09cd\u09b0\u09a1\u09bf\u0989"
                    "\u09b8\u09be\u09b0",
            "guru": "\u0a38\u0a15\u0a3e\u0a08 \u0a05\u0a30\u0a3f\u0a39\u0a70\u0a24 "
                    "\u0a17\u0a32\u0a4b\u0a2c\u0a32 \u0a2a\u0a4d\u0a30\u0a3e. \u0a32\u0a3f.",
            "gujr": "\u0ab6\u0a95\u0acd\u0aa4\u0abf \u0a85\u0ab0\u0acd\u0aac\u0aa8",
            "orya": "\u0b2c\u0b3f\u0b1c\u0b5f \u0b15\u0b28\u0b37\u0b4d\u0b1f\u0b4d\u0b30"
                    "\u0b15\u0b38\u0b28\u0b4d\u0b38\u0b4d",
            "taml": "\u0bb0\u0bbe\u0b9c\u0bcd \u0b87\u0ba9\u0bcd\u0bb5\u0bc6\u0bb8\u0bcd",
            "telu": "\u0c15\u0c43\u0c37\u0c4d\u0c23\u0c3e \u0c07\u0c02\u0c2a\u0c46\u0c15\u0c4d\u0c38\u0c4d",
            "knda": "\u0cb8\u0ca8\u0ccd \u0cb5\u0cc6\u0c82\u0c9a\u0cb0\u0ccd\u0cb8\u0ccd",
            "mlym": "\u0d36\u0d15\u0d4d\u0d24\u0d3f \u0d07\u0d02\u0d2a\u0d46\u0d15\u0d4d\u0d38\u0d4d",
        }
        blocks = {tag: (lo, hi) for tag, lo, hi in SCRIPT_BLOCKS}
        for tag, text in samples.items():
            with self.subTest(script=tag):
                out = transliterate(fold_unicode(text))
                lo, hi = blocks[tag]
                leaked = [ch for ch in out if lo <= ord(ch) <= hi]
                self.assertEqual(
                    leaked, [],
                    f"{tag}: {[hex(ord(c)) for c in leaked]} leaked into {out!r}")
