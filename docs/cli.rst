Command Line Interface
======================

phrasplit provides a command-line interface for processing text files.

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

   # Output to stdout
   phrasplit sentences input.txt

   # Output to file
   phrasplit sentences input.txt -o output.txt

   # Use a different spaCy model
   phrasplit sentences input.txt --model en_core_web_lg

Splitting Clauses
-----------------

Split text into comma-separated parts:

.. code-block:: bash

   phrasplit clauses input.txt
   phrasplit clauses input.txt -o output.txt

Splitting Paragraphs
--------------------

Split text into paragraphs:

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

   Options:
     -o, --output PATH   Output file (default: stdout)
     -m, --model TEXT    spaCy language model (default: en_core_web_sm)
     --help              Show this message and exit.

clauses
^^^^^^^

.. code-block:: text

   Usage: phrasplit clauses [OPTIONS] [INPUT_FILE]

   Split text into clauses (at commas).

   Options:
     -o, --output PATH   Output file (default: stdout)
     -m, --model TEXT    spaCy language model (default: en_core_web_sm)
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

   Options:
     -o, --output PATH        Output file (default: stdout)
     -l, --max-length INTEGER Maximum line length (default: 80, must be >= 1)
     -m, --model TEXT         spaCy language model (default: en_core_web_sm)
     --help                   Show this message and exit.

Examples
--------

Process a book for audiobook creation:

.. code-block:: bash

   # Split into sentences first
   phrasplit sentences book.txt -o book_sentences.txt

   # Then split long sentences into clauses
   phrasplit clauses book_sentences.txt -o book_clauses.txt

Create subtitles with line length limits:

.. code-block:: bash

   phrasplit longlines transcript.txt -o subtitles.txt --max-length 42

Pipeline example with multiple tools:

.. code-block:: bash

   cat book.txt | phrasplit sentences | phrasplit clauses > output.txt
