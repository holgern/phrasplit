Usage Guide
===========

This guide covers how to use phrasplit's Python API for text splitting.

Splitting Sentences
-------------------

The :func:`~phrasplit.split_sentences` function uses spaCy's NLP pipeline to
intelligently detect sentence boundaries:

.. code-block:: python

   from phrasplit import split_sentences

   text = "Dr. Smith is here. She has a Ph.D. in Chemistry."
   sentences = split_sentences(text)
   print(sentences)
   # ['Dr. Smith is here.', 'She has a Ph.D. in Chemistry.']

The function correctly handles:

- **Abbreviations**: Mr., Mrs., Dr., Prof., etc.
- **Acronyms**: U.S.A., U.K., etc.
- **Titles**: Ph.D., M.D., etc.
- **URLs**: www.example.com
- **Ellipses**: Text with... ellipses

Example with abbreviations:

.. code-block:: python

   text = "Mr. Brown met Prof. Green. They discussed the U.S.A. case."
   sentences = split_sentences(text)
   # ['Mr. Brown met Prof. Green.', 'They discussed the U.S.A. case.']

Splitting Clauses
-----------------

The :func:`~phrasplit.split_clauses` function splits text at commas, creating
natural pause points ideal for audiobook and text-to-speech applications:

.. code-block:: python

   from phrasplit import split_clauses

   text = "I like coffee, and I like tea."
   clauses = split_clauses(text)
   print(clauses)
   # ['I like coffee,', 'and I like tea.']

The comma is kept at the end of each clause, preserving the original punctuation.

More complex example:

.. code-block:: python

   text = "When the sun rose, the birds began to sing, and the day started."
   clauses = split_clauses(text)
   # ['When the sun rose,', 'the birds began to sing,', 'and the day started.']

Splitting Paragraphs
--------------------

The :func:`~phrasplit.split_paragraphs` function splits text at double newlines:

.. code-block:: python

   from phrasplit import split_paragraphs

   text = """First paragraph with some text.

   Second paragraph with more text.

   Third paragraph."""

   paragraphs = split_paragraphs(text)
   # ['First paragraph with some text.',
   #  'Second paragraph with more text.',
   #  'Third paragraph.']

The function handles multiple blank lines and whitespace-only lines:

.. code-block:: python

   text = "First.\n\n\n\nSecond."  # Multiple blank lines
   paragraphs = split_paragraphs(text)
   # ['First.', 'Second.']

Splitting Long Lines
--------------------

The :func:`~phrasplit.split_long_lines` function breaks long lines at natural
boundaries (sentences and clauses) to fit within a maximum length:

.. code-block:: python

   from phrasplit import split_long_lines

   text = "This is a very long sentence. This is another sentence that makes it even longer."
   lines = split_long_lines(text, max_length=40)
   # Each line will be <= 40 characters when possible

The splitting strategy:

1. First, try to split at sentence boundaries
2. If still too long, split at clause boundaries (commas)
3. If still too long, split at word boundaries

Using Different Language Models
-------------------------------

All functions that use spaCy accept a ``language_model`` parameter:

.. code-block:: python

   from phrasplit import split_sentences

   # Use a larger, more accurate model
   sentences = split_sentences(text, language_model="en_core_web_lg")

   # Use a model for another language
   sentences = split_sentences(german_text, language_model="de_core_news_sm")

Make sure to download the model first:

.. code-block:: bash

   python -m spacy download de_core_news_sm

Processing Pipeline Example
---------------------------

Here's a complete example of processing a document:

.. code-block:: python

   from phrasplit import split_paragraphs, split_sentences, split_clauses

   def process_document(text):
       """Process a document into structured parts."""
       result = []

       for para_idx, paragraph in enumerate(split_paragraphs(text)):
           para_data = {"paragraph": para_idx + 1, "sentences": []}

           for sent_idx, sentence in enumerate(split_sentences(paragraph)):
               sent_data = {
                   "sentence": sent_idx + 1,
                   "text": sentence,
                   "clauses": split_clauses(sentence)
               }
               para_data["sentences"].append(sent_data)

           result.append(para_data)

       return result

   # Example usage
   text = """Hello world, this is a test. Another sentence here.

   Second paragraph with more content, and some clauses."""

   structure = process_document(text)
