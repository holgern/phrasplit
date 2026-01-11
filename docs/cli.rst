Command Line Interface
======================

phrasplit provides a command-line interface for processing text files.

By default, the CLI automatically detects if spaCy is installed and uses it for
best accuracy. You can use the ``--simple`` flag to force fast regex-based
splitting that doesn't require spaCy.

Basic Usage
-----------

The CLI has four main commands:

- ``sentences`` - Split text into sentences
- ``clauses`` - Split text into clauses (at commas)
- ``paragraphs`` - Split text into paragraphs
- ``longlines`` - Split long lines at natural boundaries

Getting Help
------------

.. code-block:: bash

   # Show available commands
   phrasplit --help

   # Show help for a specific command
   phrasplit sentences --help

Splitting Sentences
-------------------

Split a text file into sentences:

.. code-block:: bash

   # Auto-detect mode (uses spaCy if available)
   phrasplit sentences input.txt

   # Output to file
   phrasplit sentences input.txt -o output.txt

   # Force simple mode (60x faster, no spaCy required)
   phrasplit sentences input.txt --simple

   # Use a different spaCy model (only when using spaCy mode)
   phrasplit sentences input.txt --model en_core_web_lg

Splitting Clauses
-----------------

Split text into comma-separated parts:

.. code-block:: bash

   phrasplit clauses input.txt
   phrasplit clauses input.txt -o output.txt

   # Use simple mode for faster processing
   phrasplit clauses input.txt --simple

Splitting Paragraphs
--------------------

Split text into paragraphs (no spaCy needed):

.. code-block:: bash

   phrasplit paragraphs input.txt
   phrasplit paragraphs input.txt -o output.txt

Splitting Long Lines
--------------------

Split long lines to fit within a maximum length:

.. code-block:: bash

   # Default max length is 80 characters
   phrasplit longlines input.txt

   # Custom max length
   phrasplit longlines input.txt --max-length 60

   # Use simple mode for faster processing
   phrasplit longlines input.txt --simple --max-length 60

   # Output to file with custom length
   phrasplit longlines input.txt -o output.txt -l 100

Reading from stdin
------------------

All commands support reading from stdin by omitting the input file or using ``-``:

.. code-block:: bash

   # Pipe input
   echo "Hello world. This is a test." | phrasplit sentences

   # Redirect input
   phrasplit sentences < input.txt

   # Explicit stdin with dash
   phrasplit sentences - < input.txt

   # Combine with output file
   cat input.txt | phrasplit clauses -o output.txt

Command Options
---------------

sentences
^^^^^^^^^

.. code-block:: text

   Usage: phrasplit sentences [OPTIONS] [INPUT_FILE]

   Split text into sentences.

   By default, uses spaCy if available for best accuracy. Use --simple for
   faster regex-based splitting that doesn't require spaCy.

   Options:
     -o, --output PATH   Output file (default: stdout)
     -m, --model TEXT    spaCy language model (default: en_core_web_sm)
     --simple            Use simple regex-based splitting (faster, no spaCy required)
     --help              Show this message and exit.

clauses
^^^^^^^

.. code-block:: text

   Usage: phrasplit clauses [OPTIONS] [INPUT_FILE]

   Split text into clauses (at commas).

   By default, uses spaCy if available for best accuracy. Use --simple for
   faster regex-based splitting that doesn't require spaCy.

   Options:
     -o, --output PATH   Output file (default: stdout)
     -m, --model TEXT    spaCy language model (default: en_core_web_sm)
     --simple            Use simple regex-based splitting (faster, no spaCy required)
     --help              Show this message and exit.

paragraphs
^^^^^^^^^^

.. code-block:: text

   Usage: phrasplit paragraphs [OPTIONS] [INPUT_FILE]

   Split text into paragraphs.

   Options:
     -o, --output PATH   Output file (default: stdout)
     --help              Show this message and exit.

longlines
^^^^^^^^^

.. code-block:: text

   Usage: phrasplit longlines [OPTIONS] [INPUT_FILE]

   Split long lines at sentence/clause boundaries.

   By default, uses spaCy if available for best accuracy. Use --simple for
   faster regex-based splitting that doesn't require spaCy.

   Options:
     -o, --output PATH        Output file (default: stdout)
     -l, --max-length INTEGER Maximum line length (default: 80, must be >= 1)
     -m, --model TEXT         spaCy language model (default: en_core_web_sm)
     --simple                 Use simple regex-based splitting (faster, no spaCy required)
     --help                   Show this message and exit.

Examples
--------

Process a book for audiobook creation:

.. code-block:: bash

   # Split into sentences first (using simple mode for speed)
   phrasplit sentences book.txt --simple -o book_sentences.txt

   # Then split long sentences into clauses
   phrasplit clauses book_sentences.txt --simple -o book_clauses.txt

Create subtitles with line length limits:

.. code-block:: bash

   phrasplit longlines transcript.txt -o subtitles.txt --max-length 42

Speed comparison example:

.. code-block:: bash

   # Using spaCy mode (higher accuracy, slower)
   time phrasplit sentences large_book.txt -o output_spacy.txt

   # Using simple mode (60x faster, good for well-formatted text)
   time phrasplit sentences large_book.txt --simple -o output_simple.txt

Pipeline example with multiple tools:

.. code-block:: bash

   cat book.txt | phrasplit sentences --simple | phrasplit clauses --simple > output.txt
