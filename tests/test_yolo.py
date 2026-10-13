import pytest

from firewatch.data.yolo import (
    LabelParseError,
    YoloBox,
    parse_label_line,
    pixel_to_yolo,
    read_label_file,
    yolo_to_pixel,
)


def test_parse_label_line():
    assert parse_label_line("1 0.5 0.5 0.2 0.4\n") == YoloBox(1, 0.5, 0.5, 0.2, 0.4)


def test_parse_accepts_float_formatted_class_id():
    assert parse_label_line("0.0 0.1 0.1 0.1 0.1").class_id == 0


@pytest.mark.parametrize(
    "line",
    [
        "1 0.5 0.5 0.2",
        "1 0.5 0.5 0.2 0.4 9",
        "a 0.5 0.5 0.2 0.4",
        "1.5 0.5 0.5 0.2 0.4",
        "-1 0 0 1 1",
    ],
)
def test_parse_rejects_malformed(line):
    with pytest.raises(LabelParseError):
        parse_label_line(line)


def test_is_normalised():
    assert YoloBox(0, 0.5, 0.5, 0.2, 0.2).is_normalised()
    assert not YoloBox(0, 1.2, 0.5, 0.2, 0.2).is_normalised()
    assert not YoloBox(0, 0.5, 0.5, 0.0, 0.2).is_normalised()


def test_read_label_file_skips_blank_lines(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("0 0.5 0.5 0.2 0.2\n\n1 0.1 0.1 0.05 0.05\n")
    assert [b.class_id for b in read_label_file(f)] == [0, 1]


def test_read_label_file_empty_is_background(tmp_path):
    f = tmp_path / "bg.txt"
    f.write_text("")
    assert read_label_file(f) == []


def test_read_label_file_reports_location(tmp_path):
    f = tmp_path / "bad.txt"
    f.write_text("0 0.5 0.5 0.2 0.2\n0 oops\n")
    with pytest.raises(LabelParseError, match=r"bad\.txt:2"):
        read_label_file(f)


def test_round_trip_conversion():
    size = (640, 480)
    pixel = [64, 48, 320, 240]
    yolo = pixel_to_yolo(size, pixel)
    assert yolo == pytest.approx([0.3, 0.3, 0.4, 0.4])
    assert yolo_to_pixel(size, yolo) == pixel


def test_yolo_to_pixel_clamps_negative():
    assert yolo_to_pixel((100, 100), (0.05, 0.05, 0.2, 0.2))[:2] == [0, 0]


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_parse_rejects_non_finite_coordinates(value):
    with pytest.raises(LabelParseError, match="non-finite"):
        parse_label_line(f"0 {value} 0.5 0.1 0.1")


def test_read_label_file_tolerates_utf8_bom(tmp_path):
    f = tmp_path / "bom.txt"
    f.write_bytes(b"\xef\xbb\xbf0 0.5 0.5 0.1 0.1\r\n")
    assert read_label_file(f) == [YoloBox(0, 0.5, 0.5, 0.1, 0.1)]


def test_read_label_file_rejects_binary(tmp_path):
    f = tmp_path / "bin.txt"
    f.write_bytes(b"\xff\xfe\x00garbage")
    with pytest.raises(LabelParseError, match="not valid UTF-8"):
        read_label_file(f)
