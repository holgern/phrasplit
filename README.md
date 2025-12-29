# phrasplit

A Python library for splitting text into sentences, clauses, or paragraphs using spaCy
NLP. Designed for audiobook creation and text-to-speech processing.

## Features

- **Sentence splitting**: Intelligent sentence boundary detection using spaCy
- **Clause splitting**: Split sentences at commas for natural pause points
- **Paragraph splitting**: Split text at double newlines
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

### `split_sentences(text, language_model="en_core_web_sm")`

Split text into sentences using spaCy's sentence boundary detection.

**Parameters:**

- `text`: Input text string
- `language_model`: spaCy model to use (default: "en_core_web_sm")

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

Split text at commas to create natural pause points for text-to-speech:

```python
from phrasplit import split_clauses

text = "When the sun rose, the birds began to sing, and the day started."
parts = split_clauses(text)
# ['When the sun rose,', 'the birds began to sing,', 'and the day started.']
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
