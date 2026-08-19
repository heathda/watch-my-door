"""encode_image: shrink what the model sees, never what we keep.

A 4 MP alert frame costs ~4.7K vision tokens, which overflows a default context
and OOMs an 8 GB card when two cameras trigger at once. Downscaling is the fix,
but it sits in the alert path -- so the fallback matters as much as the resize.
"""
import base64
import io

import pytest

import cam_watcher

Image = pytest.importorskip("PIL.Image")


def _jpeg(tmp_path, size, name="frame.jpg"):
    p = tmp_path / name
    Image.new("RGB", size, (120, 120, 120)).save(p, format="JPEG")
    return p


def _decoded(b64):
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def test_oversized_frame_is_capped_to_max_edge(tmp_path, monkeypatch):
    monkeypatch.setattr(cam_watcher, "MAX_IMAGE_EDGE", 1600)
    src = _jpeg(tmp_path, (2688, 1520))
    out = _decoded(cam_watcher.encode_image(src))
    assert max(out.size) == 1600


def test_aspect_ratio_is_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr(cam_watcher, "MAX_IMAGE_EDGE", 1600)
    src = _jpeg(tmp_path, (2688, 1520))
    out = _decoded(cam_watcher.encode_image(src))
    assert out.size == (1600, round(1520 * 1600 / 2688))


def test_portrait_frame_caps_the_long_edge(tmp_path, monkeypatch):
    """max(), not width -- one camera is 2560x1920 and aspect ratios vary."""
    monkeypatch.setattr(cam_watcher, "MAX_IMAGE_EDGE", 1600)
    src = _jpeg(tmp_path, (1200, 2400))
    out = _decoded(cam_watcher.encode_image(src))
    assert out.size == (800, 1600)


def test_small_frame_is_not_upscaled(tmp_path, monkeypatch):
    monkeypatch.setattr(cam_watcher, "MAX_IMAGE_EDGE", 1600)
    src = _jpeg(tmp_path, (640, 360))
    out = _decoded(cam_watcher.encode_image(src))
    assert out.size == (640, 360)


def test_zero_disables_downscaling(tmp_path, monkeypatch):
    """MAX_IMAGE_EDGE=0 must send the file through byte-for-byte."""
    monkeypatch.setattr(cam_watcher, "MAX_IMAGE_EDGE", 0)
    src = _jpeg(tmp_path, (2688, 1520))
    assert base64.b64decode(cam_watcher.encode_image(src)) == src.read_bytes()


def test_unreadable_image_falls_back_to_raw_bytes(tmp_path, monkeypatch):
    """Nothing may crash out of main(): a corrupt frame still gets classified."""
    monkeypatch.setattr(cam_watcher, "MAX_IMAGE_EDGE", 1600)
    p = tmp_path / "corrupt.jpg"
    p.write_bytes(b"\xff\xd8\xff\xe0not-actually-a-jpeg")
    assert base64.b64decode(cam_watcher.encode_image(p)) == p.read_bytes()


def test_missing_pillow_falls_back_to_raw_bytes(tmp_path, monkeypatch):
    """Pillow is a new runtime dep; a box that missed it must not go dark."""
    monkeypatch.setattr(cam_watcher, "MAX_IMAGE_EDGE", 1600)
    src = _jpeg(tmp_path, (2688, 1520))
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def boom(name, *a, **kw):
        if name == "PIL" or name.startswith("PIL."):
            raise ImportError("No module named 'PIL'")
        return real_import(name, *a, **kw)

    monkeypatch.setitem(__import__("builtins").__dict__, "__import__", boom)
    try:
        out = cam_watcher.encode_image(src)
    finally:
        monkeypatch.undo()
    assert base64.b64decode(out) == src.read_bytes()


def test_downscaled_payload_is_smaller_than_the_original(tmp_path, monkeypatch):
    """The point of the exercise: fewer bytes and fewer vision tokens."""
    monkeypatch.setattr(cam_watcher, "MAX_IMAGE_EDGE", 1600)
    src = _jpeg(tmp_path, (2688, 1520))
    assert len(base64.b64decode(cam_watcher.encode_image(src))) < src.stat().st_size
