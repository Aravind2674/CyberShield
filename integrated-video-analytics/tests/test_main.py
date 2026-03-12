import main


def test_parse_size_bytes_supports_human_units():
    assert main.parse_size_bytes("16MB", 1) == 16 * 1024 * 1024
    assert main.parse_size_bytes("2048", 1) == 2048
    assert main.parse_size_bytes("bad", 7) == 7


def test_sanitize_upload_name_drops_path_components():
    assert main.sanitize_upload_name("../unsafe name!!.mp4") == "unsafe_name.mp4"
    assert main.sanitize_upload_name(None) == "video.mp4"
