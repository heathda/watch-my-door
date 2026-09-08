"""What classify() actually puts on the wire.

Two request fields are load-bearing and neither is visible in the response:

  * options.num_ctx -- Ollama's 4096 default hard-400s a 4 MP frame.
  * think -- a reasoning model puts its answer in `thinking` and leaves
    `content` empty, which parses to an empty label that can never match
    alert_on. That failure is silent: no error, no email, just nothing.
"""
import pytest

import cam_watcher


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def _clear_caps_cache():
    """model_capabilities() memoises in a module global; isolate each test."""
    cam_watcher._MODEL_CAPS = None
    yield
    cam_watcher._MODEL_CAPS = None


def _capture(monkeypatch, caps):
    """Stub /api/show with `caps` and record the /api/chat body."""
    sent = {}

    def fake_post(url, json=None, timeout=None, **kw):
        if url.endswith("/api/show"):
            return FakeResponse({"capabilities": caps})
        sent.update(json)
        return FakeResponse({"message": {"content": "NONE\nnothing here"}})

    monkeypatch.setattr(cam_watcher.requests, "post", fake_post)
    return sent


def test_num_ctx_is_sent_on_every_request(monkeypatch):
    monkeypatch.setattr(cam_watcher, "OLLAMA_NUM_CTX", 8192)
    sent = _capture(monkeypatch, [])
    cam_watcher.classify("prompt", "b64")
    assert sent["options"]["num_ctx"] == 8192


def test_thinking_model_gets_think_disabled(monkeypatch):
    sent = _capture(monkeypatch, ["completion", "vision", "thinking"])
    cam_watcher.classify("prompt", "b64")
    assert sent["think"] is False


def test_non_thinking_model_omits_the_field(monkeypatch):
    """qwen2.5vl has no `thinking`; sending the field risks a 400."""
    sent = _capture(monkeypatch, ["completion", "vision"])
    cam_watcher.classify("prompt", "b64")
    assert "think" not in sent


def test_capabilities_are_probed_once_not_per_call(monkeypatch):
    calls = []

    def fake_post(url, json=None, timeout=None, **kw):
        if url.endswith("/api/show"):
            calls.append(url)
            return FakeResponse({"capabilities": ["thinking"]})
        return FakeResponse({"message": {"content": "NONE\n."}})

    monkeypatch.setattr(cam_watcher.requests, "post", fake_post)
    cam_watcher.classify("prompt", "b64")
    cam_watcher.classify("prompt", "b64")
    assert len(calls) == 1


def test_unreachable_show_endpoint_still_classifies(monkeypatch):
    """Fail open: a probe failure must not cost us the classification."""
    def fake_post(url, json=None, timeout=None, **kw):
        if url.endswith("/api/show"):
            raise cam_watcher.requests.RequestException("boom")
        return FakeResponse({"message": {"content": "PERSON\nsomeone there"}})

    monkeypatch.setattr(cam_watcher.requests, "post", fake_post)
    answer, _ = cam_watcher.classify("prompt", "b64")
    assert answer.startswith("PERSON")
    assert cam_watcher.model_capabilities() == set()


def test_show_endpoint_is_derived_from_the_chat_url(monkeypatch):
    seen = []

    def fake_post(url, json=None, timeout=None, **kw):
        seen.append(url)
        if url.endswith("/api/show"):
            return FakeResponse({"capabilities": []})
        return FakeResponse({"message": {"content": "NONE\n."}})

    monkeypatch.setattr(cam_watcher.requests, "post", fake_post)
    monkeypatch.setattr(cam_watcher, "OLLAMA_URL", "http://box:11434/api/chat")
    cam_watcher.classify("prompt", "b64")
    assert "http://box:11434/api/show" in seen
