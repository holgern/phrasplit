API Reference
=============

This page contains the complete API reference for phrasplit.

Main Functions
--------------

.. module:: phrasplit

split_sentences
^^^^^^^^^^^^^^^

.. autofunction:: phrasplit.split_sentences

**Example:**

.. code-block:: python

   from phrasplit import split_sentences

   text = "Dr. Smith is here. She has a Ph.D. in Chemistry."
   sentences = split_sentences(text)
   # ['Dr. Smith is here.', 'She has a Ph.D. in Chemistry.']

split_clauses
^^^^^^^^^^^^^

.. autofunction:: phrasplit.split_clauses

**Example:**

.. code-block:: python

   from phrasplit import split_clauses

   text = "I like coffee, and I like tea."
   clauses = split_clauses(text)
   # ['I like coffee,', 'and I like tea.']

split_paragraphs
^^^^^^^^^^^^^^^^

.. autofunction:: phrasplit.split_paragraphs

**Example:**

.. code-block:: python

   from phrasplit import split_paragraphs

   text = "First paragraph.\n\nSecond paragraph."
   paragraphs = split_paragraphs(text)
   # ['First paragraph.', 'Second paragraph.']

split_long_lines
^^^^^^^^^^^^^^^^

.. autofunction:: phrasplit.split_long_lines

**Example:**

.. code-block:: python

   from phrasplit import split_long_lines

   text = "This is a very long sentence that needs to be split into smaller parts."
   lines = split_long_lines(text, max_length=40)

Module Contents
---------------

splitter module
^^^^^^^^^^^^^^^

.. automodule:: phrasplit.splitter
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: _get_nlp, _protect_ellipsis, _restore_ellipsis, _split_sentence_into_clauses, _split_at_clauses, _hard_split, _split_at_boundaries

Type Information
----------------

phrasplit is fully typed and includes a ``py.typed`` marker file for PEP 561
compliance. You can use it with mypy and other type checkers.

Function signatures:

.. code-block:: python

   def split_sentences(
       text: str,
       language_model: str = "en_core_web_sm",
   ) -> list[str]: ...

   def split_clauses(
       text: str,
       language_model: str = "en_core_web_sm",
   ) -> list[str]: ...

   def split_paragraphs(text: str) -> list[str]: ...

   def split_long_lines(
       text: str,
       max_length: int,
       language_model: str = "en_core_web_sm",
   ) -> list[str]: ...
