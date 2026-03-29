from __future__ import annotations

from ankiflow_tts.cli import build_parser, main
from ankiflow_tts.config import Settings
from ankiflow_tts.exceptions import ConfigError
from ankiflow_tts.types import CardOutcome, CardOutcomeStatus, RetryPolicy, RunSummary


def test_build_parser_accepts_import_command() -> None:
    args = build_parser().parse_args(["import", "--input", "cards.txt", "--dry-run"])

    assert args.command == "import"
    assert args.input_path == "cards.txt"
    assert args.dry_run is True


def test_main_returns_error_code_for_application_error(monkeypatch, capsys, tmp_path) -> None:
    input_path = tmp_path / "cards.txt"
    input_path.write_text("Hello.;Ola.;;note\n", encoding="utf-8")

    def raise_config_error(args):
        _ = args
        raise ConfigError("boom")

    monkeypatch.setattr("ankiflow_tts.cli.build_settings", raise_config_error)

    exit_code = main(["import", "--input", str(input_path), "--dry-run"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Error: boom" in captured.err


def test_main_returns_zero_for_success(monkeypatch, capsys, tmp_path) -> None:
    input_path = tmp_path / "cards.txt"
    input_path.write_text("Hello.;Ola.;;note\n", encoding="utf-8")
    settings = Settings(
        input_path=input_path,
        deck_name="Deck",
        model_name="Model",
        anki_url="http://127.0.0.1:8765",
        gemini_api_key=None,
        gemini_model=None,
        gemini_voice=None,
        dry_run=True,
        verbose=False,
        retry_policy=RetryPolicy(),
    )
    summary = RunSummary(mode="dry-run", input_path=input_path, total_lines=1, valid_rows=1)
    summary.add_outcome(
        CardOutcome(
            line_number=1,
            status=CardOutcomeStatus.WOULD_IMPORT,
            message="placeholder",
        )
    )

    monkeypatch.setattr("ankiflow_tts.cli.build_settings", lambda args: settings)
    monkeypatch.setattr("ankiflow_tts.cli.configure_logging", lambda settings: None)
    monkeypatch.setattr("ankiflow_tts.cli.AnkiConnectClient", lambda url: object())
    monkeypatch.setattr("ankiflow_tts.cli._build_tts_client", lambda settings: None)

    class FakeImporter:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def run(self, settings_arg):
            assert settings_arg is settings
            return summary

    monkeypatch.setattr("ankiflow_tts.cli.Importer", FakeImporter)
    monkeypatch.setattr("ankiflow_tts.cli.render_summary", lambda result: "summary output")

    exit_code = main(["import", "--input", str(input_path), "--dry-run"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "summary output" in captured.out
