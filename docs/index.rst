phrasplit Documentation
=======================

A Python library for splitting text into sentences, clauses, or paragraphs using
spaCy NLP. Designed for audiobook creation and text-to-speech processing.

Features
--------

- **Sentence splitting**: Intelligent sentence boundary detection using spaCy
- **Clause splitting**: Split sentences at commas for natural pause points
- **Paragraph splitting**: Split text at double newlines
- **Long line splitting**: Break long lines at sentence/clause boundaries
- **Abbreviation handling**: Correctly handles Mr., Dr., U.S.A., etc.
- **Ellipsis support**: Preserves ellipses without incorrect splitting

Installation
------------

Install phrasplit using pip:

.. code-block:: bash

   pip install phrasplit

You'll also need to download a spaCy language model:

.. code-block:: bash

   python -m spacy download en_core_web_sm

Quick Start
-----------

.. code-block:: python

   from phrasplit import split_sentences, split_clauses, split_paragraphs

   # Split text into sentences
   text = "Dr. Smith is here. She has a Ph.D. in Chemistry."
   sentences = split_sentences(text)
   # ['Dr. Smith is here.', 'She has a Ph.D. in Chemistry.']

   # Split sentences into comma-separated parts
   text = "I like coffee, and I like tea."
   clauses = split_clauses(text)
   # ['I like coffee,', 'and I like tea.']

   # Split text into paragraphs
   text = "First paragraph.\n\nSecond paragraph."
   paragraphs = split_paragraphs(text)
   # ['First paragraph.', 'Second paragraph.']

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   cli
   api
   examples

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
