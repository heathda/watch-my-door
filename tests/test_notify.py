"""resolve_password: Credential Manager (keyring) preferred, .env as fallback."""
import sys
import types

import notify


def _fake_keyring(stored: dict):
    """A stand-in keyring module: stored is {(service, user): password}."""
    mod = types.ModuleType("keyring")
    mod.get_password = lambda service, user: stored.get((service, user))
    return mod


def test_prefers_keyring_over_env(monkeypatch):
    monkeypatch.setenv("GMAIL_PASSWORD", "from-env")
    monkeypatch.setattr(notify, "KEYRING_SERVICE", "watch-my-door")
    monkeypatch.setitem(sys.modules, "keyring",
                        _fake_keyring({("watch-my-door", "me@gmail.com"): "from-vault"}))
    assert notify.resolve_password("me@gmail.com") == "from-vault"


def test_falls_back_to_env_when_vault_empty(monkeypatch):
    monkeypatch.setenv("GMAIL_PASSWORD", "from-env")
    monkeypatch.setitem(sys.modules, "keyring", _fake_keyring({}))  # nothing stored
    assert notify.resolve_password("me@gmail.com") == "from-env"


def test_falls_back_to_env_when_keyring_unavailable(monkeypatch):
    monkeypatch.setenv("GMAIL_PASSWORD", "from-env")
    # Simulate keyring import blowing up (not installed / no backend).
    broken = types.ModuleType("keyring")
    def boom(*a, **k):
        raise RuntimeError("no backend")
    broken.get_password = boom
    monkeypatch.setitem(sys.modules, "keyring", broken)
    assert notify.resolve_password("me@gmail.com") == "from-env"


def test_returns_none_when_nothing_anywhere(monkeypatch):
    monkeypatch.delenv("GMAIL_PASSWORD", raising=False)
    monkeypatch.setitem(sys.modules, "keyring", _fake_keyring({}))
    assert notify.resolve_password("me@gmail.com") is None


# --- send_email recipient handling ------------------------------------------

class _FakeSMTP:
    """Stand-in for smtplib.SMTP_SSL that records the message it's asked to send."""
    sent = []  # list of EmailMessage objects across all instances

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def login(self, user, password):
        pass

    def send_message(self, msg):
        _FakeSMTP.sent.append(msg)


def _configured_env(monkeypatch, notify_email):
    """Minimal valid email config with the password coming from the env fallback."""
    monkeypatch.setenv("GMAIL_USER", "me@gmail.com")
    monkeypatch.setenv("GMAIL_PASSWORD", "app-password")
    monkeypatch.setenv("NOTIFY_EMAIL", notify_email)
    monkeypatch.setitem(sys.modules, "keyring", _fake_keyring({}))  # force env fallback
    monkeypatch.setattr(notify.smtplib, "SMTP_SSL", _FakeSMTP)
    _FakeSMTP.sent = []


def test_single_recipient(monkeypatch):
    _configured_env(monkeypatch, "solo@example.com")
    assert notify.send_email("subj", "body") is True
    assert _FakeSMTP.sent[0]["To"] == "solo@example.com"


def test_multiple_recipients_comma_separated(monkeypatch):
    _configured_env(monkeypatch, "a@example.com, b@example.com")
    assert notify.send_email("subj", "body") is True
    assert _FakeSMTP.sent[0]["To"] == "a@example.com, b@example.com"


def test_recipients_semicolons_and_whitespace_normalized(monkeypatch):
    _configured_env(monkeypatch, " a@example.com ;b@example.com ; ")
    assert notify.send_email("subj", "body") is True
    # Semicolons become commas, whitespace trimmed, empty entries dropped.
    assert _FakeSMTP.sent[0]["To"] == "a@example.com, b@example.com"


def test_no_recipient_fails_without_sending(monkeypatch):
    _configured_env(monkeypatch, "   ")  # only whitespace -> no valid addresses
    assert notify.send_email("subj", "body") is False
    assert _FakeSMTP.sent == []
