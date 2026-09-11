"""resolve_image: handle both full paths and BlueIris's bare-filename &ALERT_PATH."""
import cam_watcher


def test_absolute_path_that_exists(sample_image, monkeypatch):
    monkeypatch.setattr(cam_watcher, "ALERT_IMAGE_DIR", "")
    assert cam_watcher.resolve_image(str(sample_image)) == sample_image


def test_missing_path_no_alert_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(cam_watcher, "ALERT_IMAGE_DIR", "")
    assert cam_watcher.resolve_image(str(tmp_path / "nope.jpg")) is None


def test_bare_filename_resolved_via_alert_dir(sample_image, monkeypatch):
    monkeypatch.setattr(cam_watcher, "ALERT_IMAGE_DIR", str(sample_image.parent))
    # BlueIris hands us just the filename, not the folder.
    assert cam_watcher.resolve_image(sample_image.name) == sample_image.parent / sample_image.name


def test_bare_filename_not_in_alert_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(cam_watcher, "ALERT_IMAGE_DIR", str(tmp_path))
    assert cam_watcher.resolve_image("ghost.jpg") is None


def test_windows_path_resolved_via_alert_dir(sample_image, monkeypatch):
    """A path stored by the Windows prod box, replayed on a Linux dev box.

    Path("C:\\BlueIris\\Alerts\\alert.jpg").name is the WHOLE string on POSIX,
    so before this the ALERT_IMAGE_DIR fallback silently missed every stored
    frame and no A/B could run off events.db.
    """
    monkeypatch.setattr(cam_watcher, "ALERT_IMAGE_DIR", str(sample_image.parent))
    stored = "C:\\BlueIris\\Alerts\\" + sample_image.name
    assert cam_watcher.resolve_image(stored) == sample_image.parent / sample_image.name
