from __future__ import annotations

from argparse import Namespace

import pytest

from ankiflow_tts.config import (
    DEFAULT_ANKICONNECT_URL,
    DEFAULT_DEEPGRAM_MODEL,
    build_import_settings,
    build_settings,
)
from ankiflow_tts.exceptions import ConfigError


def test_build_settings_requires_deepgram_api_key_for_live_mode(tmp_path) -> None:
    input_path = tmp_path / "cards.txt"
    input_path.write_text(
        "Deck\nEnglish Setence PT\nHello.;Ola.;;note\n",
        encoding="utf-8",
    )

    args = Namespace(
        input_path=str(input_path),
        deck="Deck",
        model="Model",
        anki_url=None,
        deepgram_api_key=None,
        deepgram_model=None,
        dry_run=False,
        verbose=False,
    )

    with pytest.raises(ConfigError) as exc_info:
        build_settings(args, env={})

    assert "Live imports require Deepgram configuration" in str(exc_info.value)


def test_build_settings_allows_dry_run_without_deepgram_api_key(tmp_path) -> None:
    input_path = tmp_path / "cards.txt"
    input_path.write_text(
        "Deck\nEnglish Setence PT\nHello.;Ola.;;note\n",
        encoding="utf-8",
    )

    args = Namespace(
        input_path=str(input_path),
        deck=None,
        model=None,
        anki_url=None,
        deepgram_api_key=None,
        deepgram_model=None,
        dry_run=True,
        verbose=True,
    )

    settings = build_settings(
        args,
        env={},
    )

    assert settings.deck_name == ""
    assert settings.model_name == ""
    assert settings.anki_url == DEFAULT_ANKICONNECT_URL
    assert settings.deepgram_api_key is None
    assert settings.deepgram_model == DEFAULT_DEEPGRAM_MODEL
    assert settings.verbose is True
    assert settings.rate_limit_policy.minimum_interval_s == 1.0


def test_build_settings_allows_live_chunk_without_deepgram_api_key(tmp_path) -> None:
    input_path = tmp_path / "chunk.txt"
    input_path.write_text(
        "\n".join(
            [
                "Speaking Chunks",
                "Chunk",
                "What is difficult about speaking English?;What makes it difficult is...;Describe one specific difficulty.;;;",
            ]
        ),
        encoding="utf-8",
    )

    args = Namespace(
        input_path=str(input_path),
        deck=None,
        model=None,
        anki_url=None,
        deepgram_api_key=None,
        deepgram_model=None,
        dry_run=False,
        verbose=False,
    )

    settings = build_settings(args, env={})

    assert settings.deepgram_api_key is None
    assert settings.deepgram_model == DEFAULT_DEEPGRAM_MODEL


def test_build_import_settings_discovers_non_empty_txt_files(
    tmp_path,
    monkeypatch,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "sample_cards.txt").write_text("\n", encoding="utf-8")
    (data_dir / "grammar.txt").write_text(
        "Grammar\nTagNoteType\nHello.;Ola.;;note;;;;\n",
        encoding="utf-8",
    )
    (data_dir / "english.txt").write_text(
        "test\nEnglish Setence PT\nHello.;Ola.;;note\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    settings = build_import_settings(
        _args(input_path=None, dry_run=True),
        env={},
    )

    assert [setting.input_path for setting in settings] == [
        (data_dir / "english.txt").resolve(),
        (data_dir / "grammar.txt").resolve(),
    ]
    assert {setting.deck_name for setting in settings} == {""}
    assert {setting.model_name for setting in settings} == {""}


def test_build_import_settings_returns_no_jobs_when_samples_are_empty(
    tmp_path,
    monkeypatch,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "sample_cards.txt").write_text("", encoding="utf-8")
    (data_dir / "sample_cards_2.txt").write_text("   \n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    settings = build_import_settings(_args(input_path=None, dry_run=False), env={})

    assert settings == ()


def test_build_import_settings_ignores_non_txt_files(
    tmp_path,
    monkeypatch,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "cards.md").write_text("Deck\nModel\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    settings = build_import_settings(_args(input_path=None, dry_run=True), env={})

    assert settings == ()


def _args(*, input_path: str | None, dry_run: bool) -> Namespace:
    return Namespace(
        input_path=input_path,
        deck=None,
        model=None,
        anki_url=None,
        deepgram_api_key=None,
        deepgram_model=None,
        dry_run=dry_run,
        verbose=False,
    )
