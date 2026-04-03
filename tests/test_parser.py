from __future__ import annotations

import pytest

from ankiflow_tts.exceptions import ParseError
from ankiflow_tts.parser import parse_input_file


def test_parse_input_file_returns_rows_and_input_duplicates(tmp_path) -> None:
    input_path = tmp_path / "cards.txt"
    input_path.write_text(
        "\n".join(
            [
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

    assert parsed.total_lines == 3
    assert parsed.valid_rows == 3
    assert [row.line_number for row in parsed.unique_rows] == [1, 3]
    assert len(parsed.duplicate_outcomes) == 1
    assert parsed.duplicate_outcomes[0].line_number == 2
    assert parsed.unique_rows[0].audio_filename.endswith(".wav")


def test_parse_input_file_reports_all_malformed_lines(tmp_path) -> None:
    input_path = tmp_path / "cards.txt"
    input_path.write_text(
        "\n".join(
            [
                "Valid.;Valido.;;note",
                "MissingTranslation;;;",
                "Too;Many;Fields;Here;Oops",
                "BlankThird;Campo;not-empty;note",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ParseError) as exc_info:
        parse_input_file(input_path, tts_model="model-a")

    issues = exc_info.value.issues
    assert [issue.line_number for issue in issues] == [2, 3, 4]


def test_parse_input_file_handles_utf8_bom(tmp_path) -> None:
    input_path = tmp_path / "cards.txt"
    input_path.write_text("\ufeffHello.;Ola.;;note\n", encoding="utf-8")

    parsed = parse_input_file(
        input_path,
        tts_model="model-a",
    )

    assert parsed.unique_rows[0].sentence_en == "Hello."
