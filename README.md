# ankiflow-tts

`ankiflow-tts` is a Python CLI that reads cards from a text file, generates Deepgram TTS audio for each sentence, uploads the audio to Anki through AnkiConnect, and creates notes in a chosen deck and model.

## Current Status

- The v1 CLI, parser, retry logic, Deepgram adapter, AnkiConnect client, importer, and test suite are implemented under `src/ankiflow_tts`.
- The automated baseline currently passes with `pytest -q`.
- Live imports now wait at least 1 second between Deepgram TTS requests.
- Notes are only created after a valid WAV is generated and the uploaded media can be retrieved back from Anki and verified.

## Requirements

- Python 3.11 or newer
- Anki Desktop with AnkiConnect listening at `http://127.0.0.1:8765`

## Input Format

Each non-empty line must use the exact format:

`SentenceEN;TranslationPT;;Notes`

Example:

`Where is the nearest station?;¿Dónde está la estación más cercana?;;spanish`

Malformed lines fail before import starts and include line numbers in the error output.

## Environment Variables

The CLI reads process environment variables directly. If you keep values in a `.env` file, load that file into your shell before running the command.

- `ANKICONNECT_URL`
  - Optional. Defaults to `http://127.0.0.1:8765`
- `DEFAULT_DECK`
  - Required unless `--deck` is passed
- `DEFAULT_MODEL`
  - Required unless `--model` is passed
- `DEEPGRAM_API_KEY`
  - Required for live imports
- `DEEPGRAM_MODEL`
  - Optional. Defaults to `aura-2-thalia-en`

## Installation

Create a virtual environment and install the project in editable mode:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -e .[dev]
```

The console entry point is:

```powershell
ankiflow-tts --help
```

You can also run the module directly after editable install:

```powershell
python -m ankiflow_tts.cli --help
```

## Loading `.env` In PowerShell

```powershell
Get-Content .env | ForEach-Object {
  if ($_ -match '^\s*#' -or $_ -match '^\s*$') { return }
  $parts = $_ -split '=', 2
  if ($parts.Length -eq 2) {
    [Environment]::SetEnvironmentVariable($parts[0].Trim(), $parts[1], 'Process')
  }
}
```

## Dry-Run Usage

Dry-run validates the file, checks duplicates against Anki, and prints a summary without generating audio or creating notes.

```powershell
ankiflow-tts import --input data/sample_cards.txt --dry-run
```

## Live Import Usage

Set the required environment variables first:

```powershell
$env:DEFAULT_DECK="test"
$env:DEFAULT_MODEL="English Setence PT"
$env:DEEPGRAM_API_KEY="your-deepgram-key"
$env:DEEPGRAM_MODEL="aura-2-thalia-en"
```

Then run:

```powershell
ankiflow-tts import --input data/sample_cards.txt
```

Live imports deliberately pace Deepgram calls by at least 1 second per text and will skip note creation if the generated or uploaded audio cannot be verified.

## Testing

Run the automated baseline with:

```powershell
pytest -q
```

## Troubleshooting

- `Could not reach AnkiConnect`
  - Start Anki Desktop, confirm the AnkiConnect add-on is installed, and verify it is listening at `http://127.0.0.1:8765`
- `Anki deck does not exist`
  - Confirm the value passed with `--deck` or stored in `DEFAULT_DECK` matches an existing Anki deck name exactly
- `Anki model does not exist`
  - Confirm the value passed with `--model` or stored in `DEFAULT_MODEL` matches an existing Anki note type name exactly
- `Live imports require Deepgram configuration`
  - Set `DEEPGRAM_API_KEY`, or pass `--deepgram-api-key`
- `Generated WAV payload contains no audio frames`
  - The provider returned an invalid audio payload; rerun after a short pause and inspect the logs if it continues
- Duplicate notes are skipped
  - This is the expected v1 behavior; rerunning the same file should not create uncontrolled duplicates
