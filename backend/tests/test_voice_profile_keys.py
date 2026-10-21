"""Voice profile keys must stay distinct per person, including non-Latin names."""

import pytest

from backend.services.speaker_identification_service import slugify_name


@pytest.mark.parametrize(
    "name,expected",
    [("Raj", "raj"), ("Raj Kumar", "raj_kumar"), ("  Priya-S. ", "priya_s"), ("", "speaker"), (None, "speaker")],
)
def test_ascii_names_keep_their_historical_slug(name, expected):
    """Existing profiles were stored under these keys and must keep matching."""
    assert slugify_name(name) == expected


def test_different_non_latin_names_no_longer_collide():
    keys = {slugify_name(n) for n in ("राज", "सीता", "రాజు", "ராஜ்", "李雷")}
    assert len(keys) == 5, keys
    assert "speaker" not in keys


def test_non_latin_slug_is_stable_and_case_insensitive_for_the_latin_part():
    assert slugify_name("राज") == slugify_name("  राज ")
    assert slugify_name("Élodie") == slugify_name("élodie"), "accented Latin names are also non-ASCII and must be stable"


def test_mixed_script_name_does_not_collide_with_its_ascii_prefix():
    assert slugify_name("Raj कुमार") != slugify_name("Raj")
    assert slugify_name("Raj कुमार").startswith("raj_")


def test_keys_are_filesystem_safe():
    for name in ("राज", "a/b\\c", "../../x", "Raj कुमार"):
        key = slugify_name(name)
        assert all(ch.isalnum() or ch == "_" for ch in key), key
