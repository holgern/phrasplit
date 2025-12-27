Installation
============

Requirements
------------

- Python 3.9 or higher
- spaCy 3.5 or higher
- click 8.0 or higher
- rich 13.0 or higher

Installing phrasplit
--------------------

The easiest way to install phrasplit is using pip:

.. code-block:: bash

   pip install phrasplit

Installing from Source
----------------------

To install from source, clone the repository and install:

.. code-block:: bash

   git clone https://github.com/holgern/phrasplit.git
   cd phrasplit
   pip install -e .

For development, install with dev dependencies:

.. code-block:: bash

   pip install -e ".[dev]"

Installing spaCy Language Models
--------------------------------

phrasplit requires a spaCy language model for sentence detection. The default
model is ``en_core_web_sm`` (English). Install it with:

.. code-block:: bash

   python -m spacy download en_core_web_sm

For better accuracy, you can use larger models:

.. code-block:: bash

   # Medium model (more accurate)
   python -m spacy download en_core_web_md

   # Large model (most accurate)
   python -m spacy download en_core_web_lg

For other languages, see the `spaCy models documentation
<https://spacy.io/models>`_.

Verifying Installation
----------------------

You can verify your installation by running:

.. code-block:: python

   import phrasplit
   print(phrasplit.__version__)

   from phrasplit import split_sentences
   print(split_sentences("Hello world. How are you?"))
   # ['Hello world.', 'How are you?']
