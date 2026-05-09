from __future__ import annotations

import pytest

from ankiflow_tts.exceptions import ParseError
from ankiflow_tts.parser import parse_input_file
from ankiflow_tts.reporting import render_parse_error


def test_parse_input_file_returns_rows_and_input_duplicates(tmp_path) -> None:
    input_path = tmp_path / "cards.txt"
    input_path.write_text(
        "\n".join(
            [
                "Deck",
                "English Setence PT",
                "Hello there.;Ola.;;note one",
                "Hello   there.;Ola outra.;;note two",
                "Goodbye.;Tchau.;;",
            ]
        ),
        encoding="utf-8",
    )

    parsed = parse_input_file(
        input_path,
        tts_model="model-a",
    )

    assert parsed.deck_name == "Deck"
    assert parsed.model_name == "English Setence PT"
    assert parsed.total_lines == 5
    assert parsed.valid_rows == 3
    assert [row.line_number for row in parsed.unique_rows] == [3, 5]
    assert len(parsed.duplicate_outcomes) == 1
    assert parsed.duplicate_outcomes[0].line_number == 4
    assert parsed.unique_rows[0].audio_filename.endswith(".wav")
    assert parsed.unique_rows[0].fields["SentenceEN"] == "Hello there."


def test_parse_input_file_reports_all_malformed_lines(tmp_path) -> None:
    input_path = tmp_path / "cards.txt"
    input_path.write_text(
        "\n".join(
            [
                "Deck",
                "English Setence PT",
                "Valid.;Valido.;;note",
                "MissingTranslation;;;",
                "Too;Many",
                "BlankThird;Campo;not-empty;note",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ParseError) as exc_info:
        parse_input_file(input_path, tts_model="model-a")

    issues = exc_info.value.issues
    assert [issue.line_number for issue in issues] == [4, 5, 6]
    assert exc_info.value.input_path == input_path
    assert f"File: {input_path}" in render_parse_error(exc_info.value)


def test_parse_input_file_handles_utf8_bom(tmp_path) -> None:
    input_path = tmp_path / "cards.txt"
    input_path.write_text(
        "\ufeffDeck\nEnglish Setence PT\nHello.;Ola.;;note\n",
        encoding="utf-8",
    )

    parsed = parse_input_file(
        input_path,
        tts_model="model-a",
    )

    assert parsed.unique_rows[0].sentence_en == "Hello."


def test_parse_input_file_supports_tag_note_type_with_semicolon_tags(tmp_path) -> None:
    input_path = tmp_path / "tag_cards.txt"
    input_path.write_text(
        "\n".join(
            [
                "Grammar",
                "TagNoteType",
                (
                    "I’m trying to improve my speaking.;"
                    "Estou tentando melhorar minha fala.;;"
                    "use I’m trying to + verb to talk about an effort you are making now.;"
                    "I’m trying to be more consistent. / I’m trying to speak more naturally.;"
                    "say it about English / say it about work / replace speaking with pronunciation.;"
                    "daily routine;conversation;pattern"
                ),
            ]
        ),
        encoding="utf-8",
    )

    parsed = parse_input_file(input_path, tts_model="model-a")

    assert parsed.deck_name == "Grammar"
    assert parsed.model_name == "TagNoteType"
    row = parsed.unique_rows[0]
    assert row.tts_text == "I’m trying to improve my speaking."
    assert row.audio_field_name == "Audio"
    assert row.fields["Audio"] == ""
    assert row.fields["Tags / Pattern"] == "daily routine;conversation;pattern"


def test_parse_input_file_supports_chunk_text_only_model(tmp_path) -> None:
    input_path = tmp_path / "chunk_cards.txt"
    input_path.write_text(
        "\n".join(
            [
                "Speaking Chunks",
                "Chunk",
                (
                    "What is something you are trying to improve in your English?;"
                    "I'm trying to get better at...;"
                    "Speak for 45 to 90 seconds about your current English goals.;"
                    "I'm trying to get better at expressing my ideas more naturally.;"
                    "express my ideas, wording, get stuck;"
                    "Say get better at expressing, not get better in expressing."
                ),
            ]
        ),
        encoding="utf-8",
    )

    parsed = parse_input_file(input_path, tts_model="model-a")

    assert parsed.deck_name == "Speaking Chunks"
    assert parsed.model_name == "Chunk"
    row = parsed.unique_rows[0]
    assert row.tts_text == "I'm trying to get better at..."
    assert row.audio_field_name is None
    assert row.audio_filename is None
    assert row.fields["Question"] == "What is something you are trying to improve in your English?"
    assert row.fields["Target Chunk"] == "I'm trying to get better at..."
