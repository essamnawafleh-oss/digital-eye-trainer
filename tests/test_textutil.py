from eye_trainer import textutil as tu


def test_normalize_maps_persian_letters_and_spaces():
    assert tu.normalize_label("  لیلى  کرک ") == "ليلى كرك"
    assert tu.normalize_label("لیلى", persian_to_arabic=False) == "لیلى"
    assert tu.normalize_label("‏رقم‎  ١") == "رقم ١"


def test_direction_rule():
    assert tu.first_strong_rtl("رقم: 12")
    assert not tu.first_strong_rtl("Plot No: رقم")
    assert not tu.first_strong_rtl("123")


def test_visual_order_pure_arabic_round_trips_through_paddlex_bidi():
    label = "اسم المالك محمد"
    visual = tu.to_visual(label)
    assert visual == label[::-1]
    assert tu.digital_eye_view(visual) == label


def test_visual_order_keeps_numbers_left_to_right():
    visual = tu.to_visual("المساحة: ٢٨٩٣٫٦٦ م2")
    assert "٢٨٩٣٫٦٦" in visual  # the number itself is not reversed
    assert tu.to_visual("Plot No: 128") == "Plot No: 128"


def test_known_paddlex_bidi_limit_on_hyphenated_dates():
    # Documented in docs/KNOWN_LIMITATIONS.md: PaddleX's get_display() swaps the parts of a
    # hyphenated date inside an Arabic line, even when every character is recognised.
    visual = tu.to_visual("التاريخ: 2001-08-16")
    assert tu.digital_eye_view(visual) == "التاريخ: 16-08-2001"


def test_categories():
    assert tu.categories("رقم القطعة: 128") >= {"arabic", "ascii_digits", "ascii_digits_in_arabic"}
    assert "mixed_script" in tu.categories("Permit No / رقم الرخصة: BP-2024/0153")
    assert "ids" in tu.categories("Permit No: BP-2024/0153")
    assert {"arabic_indic_digits", "arabic_decimal"} <= tu.categories("المساحة: ٢٨٩٣٫٦٦ م2")
    assert "latin" in tu.categories("Owner Name: Omar")


def test_number_and_id_extraction():
    assert tu.numbers("المساحة ٢٨٩٣٫٦٦ م2 و 1,250.5") == ["٢٨٩٣٫٦٦", "2", "1,250.5"]
    assert tu.ids("رقم الرخصة: BP-2024/0153 و 125/7") == ["BP-2024/0153", "125/7"]
    assert tu.token_hits(["٢٨٩٣٫٦٦"], ["٢٨٩٣٦٦"]) == 0  # a lost decimal separator is an error


def test_levenshtein():
    assert tu.levenshtein("128", "18") == 1
    assert tu.levenshtein("", "abc") == 3
    assert tu.levenshtein("same", "same") == 0
