# ankiflow-tts

`ankiflow-tts` is a Python CLI that reads cards from a text file, generates Gemini TTS audio for each sentence, uploads the audio to Anki through AnkiConnect, and creates notes in a chosen deck and model.

## Current Status

- The v1 CLI, parser, retry logic, Gemini adapter, AnkiConnect client, importer, and test suite are implemented under `src/ankiflow_tts`.
- The automated baseline currently passes with `pytest -q`.
- Real dry-run and live smoke validation still require a reachable local AnkiConnect server plus deck/model and Gemini environment variables.

## Requirements

- Python 3.11 or newer
- Anki Desktop with AnkiConnect listening at `http://127.0.0.1:8765`

## Input Format

Each non-empty line must use the exact format:

`Sentence;Translation;;Notes`

Example:

`Sara realized this was serious.;Sara percebeu que isso era serio.;;realized = percebeu`

Malformed lines fail before import starts and include line numbers in the error output.

## Environment Variables

The CLI reads process environment variables directly. If you keep values in a `.env` file, load that file into your shell before running the command.

- `ANKICONNECT_URL`
  - Optional. Defaults to `http://127.0.0.1:8765`
- `DEFAULT_DECK`
  - Required unless `--deck` is passed
- `DEFAULT_MODEL`
  - Required unless `--model` is passed
- `GEMINI_API_KEY`
  - Required for live imports
- `GEMINI_MODEL`
  - Required for live imports
- `GEMINI_VOICE`
  - Required for live imports

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

## Dry-Run Usage

Dry-run validates the file, checks duplicates against Anki, and prints a summary without generating audio or creating notes.

```powershell
ankiflow-tts import --input data/sample_cards.txt --deck "English::TTS" --model "Sentence Card" --dry-run
```

## Live Import Usage

Set the required environment variables first:

```powershell
$env:DEFAULT_DECK="English::TTS"
$env:DEFAULT_MODEL="Sentence Card"
$env:GEMINI_API_KEY="your-api-key"
$env:GEMINI_MODEL="your-gemini-tts-model"
$env:GEMINI_VOICE="your-gemini-voice"
```

Then run:

```powershell
ankiflow-tts import --input data/sample_cards.txt
```

## Testing

Run the automated baseline with:

```powershell
pytest -q
```
