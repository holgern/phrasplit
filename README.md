[![PyPI - Version](https://img.shields.io/pypi/v/phrasplit)](https://pypi.org/project/phrasplit/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/phrasplit)
![PyPI - Downloads](https://img.shields.io/pypi/dm/phrasplit)
[![codecov](https://codecov.io/gh/holgern/phrasplit/graph/badge.svg?token=iCHXwbjAXG)](https://codecov.io/gh/holgern/phrasplit)

# phrasplit

A Python library for splitting text into sentences, clauses, or paragraphs. Choose
between spaCy NLP for best accuracy or fast regex-based splitting for simple use cases.

## Features

- **Two modes**: spaCy (accurate) or simple regex (fast, no ML dependencies)
- **Sentence splitting**: Intelligent sentence boundary detection
- **Clause splitting**: Split sentences at commas for natural pause points
- **Paragraph splitting**: Split text at double newlines (no spaCy needed)
- **Hierarchical splitting**: Split text with paragraph/sentence position tracking
- **Long line splitting**: Break long lines at sentence/clause boundaries
- **Abbreviation handling**: Correctly handles Mr., Dr., U.S.A., etc.
- **Ellipsis support**: Preserves ellipses without incorrect splitting
- **25+ languages**: Multi-language abbreviation support

## Installation

### Basic Installation (Simple Mode)

For fast, regex-based splitting without ML dependencies:

```bash
pip install phrasplit
```

### Full Installation (spaCy Mode)

For best accuracy with complex text, install with spaCy:

```bash
pip install phrasplit[nlp]
python -m spacy download en_core_web_sm
```

## Performance Comparison

| Mode       | Speed       | Accuracy | Dependencies            | When to Use                      |
| ---------- | ----------- | -------- | ----------------------- | -------------------------------- |
| **Simple** | ~60x faster | ~85-90%  | None (regex only)       | Simple text, speed-critical apps |
| **spaCy**  | Baseline    | ~95%+    | spaCy + models (~500MB) | Complex text, best accuracy      |

Benchmark results (1000 sentences):

- spaCy: 1091ms
- Simple: 17ms (63x faster)

Both modes produce nearly identical results for well-formatted text.

## Quick Start

### Auto-Detection (Recommended)

Phrasplit automatically uses spaCy if installed, otherwise falls back to simple mode:

```python
from phrasplit import split_sentences, split_clauses, split_paragraphs

# Uses spaCy if installed, otherwise simple mode
text = "Dr. Smith is here. She has a Ph.D. in Chemistry."
sentences = split_sentences(text)
# ['Dr. Smith is here.', 'She has a Ph.D. in Chemistry.']

# Force simple mode (even if spaCy is installed)
sentences = split_sentences(text, use_spacy=False)

# Force spaCy mode (error if not installed)
sentences = split_sentences(text, use_spacy=True)
```

### Python API

```python
from phrasplit import split_sentences, split_clauses, split_paragraphs, split_long_lines

# Split text into sentences
text = "Dr. Smith is here. She has a Ph.D. in Chemistry."
sentences = split_sentences(text)
# ['Dr. Smith is here.', 'She has a Ph.D. in Chemistry.']

# Split sentences into comma-separated parts (for audiobook pauses)
text = "I like coffee, and I like tea."
clauses = split_clauses(text)
# ['I like coffee,', 'and I like tea.']

# Split text into paragraphs (no spaCy needed)
text = "First paragraph.\n\nSecond paragraph."
paragraphs = split_paragraphs(text)
# ['First paragraph.', 'Second paragraph.']

# Split long lines at natural boundaries
text = "This is a very long sentence that needs to be split."
lines = split_long_lines(text, max_length=30)
```

### Hierarchical Splitting with Position Tracking

For audiobook generation where you need different pause lengths between paragraphs,
sentences, and clauses, use `split_text()`:

```python
from phrasplit import split_text, Segment

# Split into sentences with paragraph tracking
text = "First sentence. Second sentence.\n\nNew paragraph here."
segments = split_text(text, mode="sentence")

for seg in segments:
    print(f"P{seg.paragraph} S{seg.sentence}: {seg.text}")
# P0 S0: First sentence.
# P0 S1: Second sentence.
# P1 S0: New paragraph here.

# Detect paragraph changes for longer pauses
for i, seg in enumerate(segments):
    if i > 0 and seg.paragraph != segments[i-1].paragraph:
        print("--- paragraph break (add longer pause) ---")
    print(seg.text)
```

Available modes:

- `"paragraph"`: Returns paragraphs (sentence=None)
- `"sentence"`: Returns sentences with paragraph index
- `"clause"`: Returns clauses with paragraph and sentence indices

### Command Line Interface

```bash
# Split into sentences
phrasplit sentences input.txt -o output.txt

# Split into clauses
phrasplit clauses input.txt -o output.txt

# Split into paragraphs
phrasplit paragraphs input.txt -o output.txt

# Split long lines (default max 80 characters)
phrasplit longlines input.txt -o output.txt --max-length 60

# Use a different spaCy model
phrasplit sentences input.txt --model en_core_web_lg

# Read from stdin (pipe or redirect)
echo "Hello world. This is a test." | phrasplit sentences
cat input.txt | phrasplit clauses -o output.txt

# Explicit stdin with dash
phrasplit sentences - < input.txt
```

## API Reference

### `split_sentences(text, language_model="en_core_web_sm", apply_corrections=True, use_spacy=None)`

Split text into sentences.

**Parameters:**

- `text`: Input text string
- `language_model`: Language model name (e.g., "en_core_web_sm", "de_core_news_sm")
  - For spaCy mode: Name of the spaCy model to use
  - For simple mode: Used to determine language for abbreviation handling
- `apply_corrections`: Apply post-processing corrections for URLs and abbreviations
  (default: True, only applies to spaCy mode)
- `use_spacy`: Choose implementation:
  - `None` (default): Auto-detect (use spaCy if available)
  - `True`: Force spaCy mode (raises ImportError if not installed)
  - `False`: Force simple regex mode

**Returns:** List of sentences

**Raises:** `ImportError` if `use_spacy=True` but spaCy is not installed

### `split_clauses(text, language_model="en_core_web_sm", use_spacy=None)`

Split text into comma-separated parts. Useful for creating natural pause points in
audiobook/TTS applications.

**Parameters:**

- `text`: Input text string
- `language_model`: Language model name (default: "en_core_web_sm")
- `use_spacy`: Choose implementation (default: None for auto-detect)

**Returns:** List of clauses (comma stays at end of each part)

### `split_paragraphs(text)`

Split text into paragraphs at double newlines. Works without spaCy.

**Parameters:**

- `text`: Input text string

**Returns:** List of paragraphs

### `split_text(text, mode="sentence", language_model="en_core_web_sm", apply_corrections=True, use_spacy=None)`

Split text into segments with hierarchical position information.

**Parameters:**

- `text`: Input text string
- `mode`: Splitting mode - "paragraph", "sentence", or "clause"
- `language_model`: Language model name (default: "en_core_web_sm")
- `apply_corrections`: Apply post-processing corrections (default: True)
- `use_spacy`: Choose implementation (default: None for auto-detect)

**Returns:** List of `Segment` namedtuples with fields:

- `text`: The segment text
- `paragraph`: Paragraph index (0-based)
- `sentence`: Sentence index within paragraph (0-based), None for paragraph mode

### `split_long_lines(text, max_length, language_model="en_core_web_sm", use_spacy=None)`

Split lines exceeding max_length at sentence/clause boundaries.

**Parameters:**

- `text`: Input text string
- `max_length`: Maximum line length in characters (must be >= 1)
- `language_model`: Language model name (default: "en_core_web_sm")
- `use_spacy`: Choose implementation (default: None for auto-detect)

**Returns:** List of lines, each within max_length (except single words exceeding limit)

**Raises:** `ValueError` if max_length is less than 1

## Use Cases

### Audiobook Creation

Split text with paragraph awareness for different pause lengths:

```python
from phrasplit import split_text

text = """When the sun rose, the birds began to sing.

A new day had started. The adventure continues."""

segments = split_text(text, mode="clause")

for i, seg in enumerate(segments):
    # Add longer pause between paragraphs
    if i > 0 and seg.paragraph != segments[i-1].paragraph:
        add_pause(duration=1.0)  # Long pause for paragraph
    # Add medium pause between sentences
    elif i > 0 and seg.sentence != segments[i-1].sentence:
        add_pause(duration=0.5)  # Medium pause for sentence
    else:
        add_pause(duration=0.2)  # Short pause for clause

    synthesize_speech(seg.text)
```

### Subtitle Generation

Split long lines to fit subtitle constraints:

```python
from phrasplit import split_long_lines

text = "This is a very long sentence that would not fit on a single subtitle line."
lines = split_long_lines(text, max_length=42)
```

### Text Processing Pipelines

```python
from phrasplit import split_paragraphs, split_sentences

text = open("book.txt").read()

for paragraph in split_paragraphs(text):
    for sentence in split_sentences(paragraph):
        process(sentence)
```

## Requirements

- Python 3.10+
- click 8.0+
- rich 13.0+
- spaCy 3.5+ (optional, for best accuracy)

## Choosing Between Modes

### Use Simple Mode When:

- Processing simple, well-formatted text
- Speed is critical (60-100x faster)
- Deploying in constrained environments (no ML dependencies)
- Installing spaCy models is not feasible (~500MB per language)

### Use spaCy Mode When:

- Processing complex, informal, or poorly formatted text
- Accuracy is paramount (5-10% better)
- Already using spaCy in your pipeline
- Working with academic or literary texts

## Migration Guide

### Upgrading from Previous Versions

Version 1.x made spaCy optional. Your existing code continues to work:

```python
# Old code (still works, auto-uses spaCy if installed)
from phrasplit import split_sentences
sentences = split_sentences(text)

# New: Explicit control
sentences = split_sentences(text, use_spacy=False)  # Force simple
sentences = split_sentences(text, use_spacy=True)   # Force spaCy
```

The `split_on_colon` parameter is deprecated and will be removed in a future version.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
