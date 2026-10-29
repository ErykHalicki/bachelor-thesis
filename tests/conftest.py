import pytest


@pytest.fixture(autouse=True)
def silence_text_to_speech(monkeypatch):
    try:
        import lerobot.utils.utils as lerobot_utils
    except ImportError:
        return
    monkeypatch.setattr(lerobot_utils, "say", lambda text, blocking=False: None)
