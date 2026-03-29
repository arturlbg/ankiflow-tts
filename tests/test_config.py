from __future__ import annotations

from argparse import Namespace

import pytest

from ankiflow_tts.config import DEFAULT_ANKICONNECT_URL, build_settings
from ankiflow_tts.exceptions import ConfigError


def test_build_settings_requires_gemini_for_live_mode(tmp_path) -> None:
    input_path = tmp_path / "cards.txt"
    input_path.write_text("Hello.;Ola.;;note\n", encoding="utf-8")

    args = Namespace(
        input_path=str(input_path),
        deck="Deck",
        model="Model",
        anki_url=None,
        gemini_api_key=None,
        gemini_model=None,
        gemini_voice=None,
        dry_run=False,
        verbose=False,
    )

    with pytest.raises(ConfigError) as exc_info:
        build_settings(args, env={})

    assert "Live imports require Gemini configuration" in str(exc_info.value)


def test_build_settings_allows_dry_run_without_gemini(tmp_path) -> None:
    input_path = tmp_path / "cards.txt"
    input_path.write_text("Hello.;Ola.;;note\n", encoding="utf-8")

    args = Namespace(
        input_path=str(input_path),
        deck=None,
        model=None,
        anki_url=None,
        gemini_api_key=None,
        gemini_model=None,
        gemini_voice=None,
        dry_run=True,
        verbose=True,
    )

    settings = build_settings(
        args,
        env={"DEFAULT_DECK": "Deck", "DEFAULT_MODEL": "Model"},
    )

    assert settings.deck_name == "Deck"
    assert settings.model_name == "Model"
    assert settings.anki_url == DEFAULT_ANKICONNECT_URL
    assert settings.gemini_api_key is None
    assert settings.verbose is True
