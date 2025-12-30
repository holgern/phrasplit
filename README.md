[![PyPI - Version](https://img.shields.io/pypi/v/phrasplit)](https://pypi.org/project/phrasplit/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/phrasplit)
![PyPI - Downloads](https://img.shields.io/pypi/dm/phrasplit)
[![codecov](https://codecov.io/gh/holgern/phrasplit/graph/badge.svg?token=iCHXwbjAXG)](https://codecov.io/gh/holgern/phrasplit)

# phrasplit

A Python library for splitting text into sentences, clauses, or paragraphs using spaCy
NLP. Designed for audiobook creation and text-to-speech processing.

## Features

- **Sentence splitting**: Intelligent sentence boundary detection using spaCy
- **Clause splitting**: Split sentences at commas for natural pause points
- **Paragraph splitting**: Split text at double newlines
- **Hierarchical splitting**: Split text with paragraph/sentence position tracking
- **Long line splitting**: Break long lines at sentence/clause boundaries
- **Abbreviation handling**: Correctly handles Mr., Dr., U.S.A., etc.
- **Ellipsis support**: Preserves ellipses without incorrect splitting

## Installation

```bash
pip install phrasplit
```

You'll also need to download a spaCy language model:

```bash
python -m spacy download en_core_web_sm
```

## Quick Start

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

# Split text into paragraphs
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

### `split_sentences(text, language_model="en_core_web_sm", apply_corrections=True, split_on_colon=True)`

Split text into sentences using spaCy's sentence boundary detection.

**Parameters:**

- `text`: Input text string
- `language_model`: spaCy model to use (default: "en_core_web_sm")
- `apply_corrections`: Apply post-processing corrections for URLs and abbreviations
  (default: True)
- `split_on_colon`: Treat colons as sentence terminators (default: True)

**Returns:** List of sentences

### `split_clauses(text, language_model="en_core_web_sm")`

Split text into comma-separated parts. Useful for creating natural pause points in
audiobook/TTS applications.

**Parameters:**

- `text`: Input text string
- `language_model`: spaCy model to use (default: "en_core_web_sm")

**Returns:** List of clauses (comma stays at end of each part)

### `split_paragraphs(text)`

Split text into paragraphs at double newlines.

**Parameters:**

- `text`: Input text string

**Returns:** List of paragraphs

### `split_text(text, mode="sentence", language_model="en_core_web_sm", apply_corrections=True, split_on_colon=True)`

Split text into segments with hierarchical position information.

**Parameters:**

- `text`: Input text string
- `mode`: Splitting mode - "paragraph", "sentence", or "clause"
- `language_model`: spaCy model to use (default: "en_core_web_sm")
- `apply_corrections`: Apply post-processing corrections (default: True)
- `split_on_colon`: Treat colons as sentence terminators (default: True)

**Returns:** List of `Segment` namedtuples with fields:

- `text`: The segment text
- `paragraph`: Paragraph index (0-based)
- `sentence`: Sentence index within paragraph (0-based), None for paragraph mode

### `split_long_lines(text, max_length, language_model="en_core_web_sm")`

Split lines exceeding max_length at sentence/clause boundaries.

**Parameters:**

- `text`: Input text string
- `max_length`: Maximum line length in characters (must be >= 1)
- `language_model`: spaCy model to use (default: "en_core_web_sm")

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

- Python 3.9+
- spaCy 3.5+
- click 8.0+
- rich 13.0+

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
