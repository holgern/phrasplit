"""Example scripts demonstrating phrasplit functionality.

This package contains example scripts showing various use cases for phrasplit:

- basic_usage.py: Core functionality (sentences, clauses, paragraphs, long lines)
- audiobook_preparation.py: Text-to-speech preparation with SSML and timing scripts
- subtitle_generation.py: SRT/WebVTT subtitle generation from transcripts
- text_analysis.py: Text statistics, readability analysis, and structure analysis
- batch_processing.py: Processing multiple files with progress tracking

To run an example:
    python -m examples.basic_usage
    python -m examples.audiobook_preparation
    python -m examples.subtitle_generation
    python -m examples.text_analysis
    python -m examples.batch_processing
"""

__all__ = [
    "audiobook_preparation",
    "basic_usage",
    "batch_processing",
    "subtitle_generation",
    "text_analysis",
]
