Examples
========

This page provides practical examples of using phrasplit for various use cases.

Audiobook Creation
------------------

Split text at natural pause points for text-to-speech processing:

.. code-block:: python

   from phrasplit import split_sentences, split_clauses

   def prepare_for_tts(text):
       """Prepare text for text-to-speech with natural pauses."""
       parts = []

       for sentence in split_sentences(text):
           # Split long sentences at commas for natural pauses
           clauses = split_clauses(sentence)
           parts.extend(clauses)

       return parts

   text = """
   When the sun rose over the mountains, the valley was filled with golden light.
   Birds began to sing their morning songs, and the world slowly awakened.
   """

   parts = prepare_for_tts(text)
   for part in parts:
       print(part)
       # Each part can be sent to TTS with appropriate pauses between them

Subtitle Generation
-------------------

Create subtitles that fit within character limits:

.. code-block:: python

   from phrasplit import split_long_lines

   def create_subtitles(transcript, max_chars=42):
       """Create subtitles from transcript with length limits."""
       lines = split_long_lines(transcript, max_length=max_chars)

       subtitles = []
       for i, line in enumerate(lines, 1):
           subtitle = {
               "index": i,
               "text": line,
               "chars": len(line)
           }
           subtitles.append(subtitle)

       return subtitles

   transcript = """
   This is a very long sentence that would not fit on a single subtitle line
   and needs to be broken up into smaller, more readable chunks for the viewer.
   """

   subtitles = create_subtitles(transcript)
   for sub in subtitles:
       print(f"{sub['index']}: {sub['text']} ({sub['chars']} chars)")

E-book Processing
-----------------

Process an e-book into structured data:

.. code-block:: python

   from phrasplit import split_paragraphs, split_sentences
   import json

   def process_ebook(text):
       """Convert e-book text to structured JSON."""
       chapters = []
       current_chapter = {"paragraphs": []}

       for para in split_paragraphs(text):
           # Detect chapter headers (simple example)
           if para.startswith("Chapter"):
               if current_chapter["paragraphs"]:
                   chapters.append(current_chapter)
               current_chapter = {
                   "title": para,
                   "paragraphs": []
               }
           else:
               sentences = split_sentences(para)
               current_chapter["paragraphs"].append({
                   "text": para,
                   "sentences": sentences,
                   "sentence_count": len(sentences)
               })

       if current_chapter["paragraphs"]:
           chapters.append(current_chapter)

       return chapters

   # Example usage
   book_text = """
   Chapter 1

   It was the best of times. It was the worst of times.

   The city was alive with activity. People rushed through the streets.

   Chapter 2

   A new day dawned. The adventure continued.
   """

   structure = process_ebook(book_text)
   print(json.dumps(structure, indent=2))

Text Analysis
-------------

Analyze text statistics:

.. code-block:: python

   from phrasplit import split_paragraphs, split_sentences, split_clauses

   def analyze_text(text):
       """Generate text statistics."""
       paragraphs = split_paragraphs(text)

       total_sentences = 0
       total_clauses = 0
       sentence_lengths = []

       for para in paragraphs:
           sentences = split_sentences(para)
           total_sentences += len(sentences)

           for sent in sentences:
               sentence_lengths.append(len(sent))
               clauses = split_clauses(sent)
               total_clauses += len(clauses)

       stats = {
           "paragraphs": len(paragraphs),
           "sentences": total_sentences,
           "clauses": total_clauses,
           "avg_sentence_length": sum(sentence_lengths) / len(sentence_lengths),
           "avg_sentences_per_paragraph": total_sentences / len(paragraphs),
           "avg_clauses_per_sentence": total_clauses / total_sentences,
       }

       return stats

   text = """
   The quick brown fox jumps over the lazy dog. This sentence is shorter.

   Another paragraph here, with some clauses, and more content.
   Final sentence of the document.
   """

   stats = analyze_text(text)
   for key, value in stats.items():
       print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")

Batch Processing
----------------

Process multiple files:

.. code-block:: python

   from pathlib import Path
   from phrasplit import split_sentences

   def process_directory(input_dir, output_dir):
       """Process all text files in a directory."""
       input_path = Path(input_dir)
       output_path = Path(output_dir)
       output_path.mkdir(exist_ok=True)

       for txt_file in input_path.glob("*.txt"):
           print(f"Processing {txt_file.name}...")

           text = txt_file.read_text(encoding="utf-8")
           sentences = split_sentences(text)

           output_file = output_path / txt_file.name
           output_file.write_text("\n".join(sentences), encoding="utf-8")

           print(f"  -> {len(sentences)} sentences written to {output_file}")

   # Example usage
   # process_directory("./books", "./processed")

Working with Different Languages
--------------------------------

Use language-specific models:

.. code-block:: python

   from phrasplit import split_sentences

   # German text
   german_text = "Guten Tag. Wie geht es Ihnen? Das Wetter ist schön."
   # First: python -m spacy download de_core_news_sm
   german_sentences = split_sentences(german_text, language_model="de_core_news_sm")

   # French text
   french_text = "Bonjour. Comment allez-vous? Il fait beau aujourd'hui."
   # First: python -m spacy download fr_core_news_sm
   french_sentences = split_sentences(french_text, language_model="fr_core_news_sm")

   # Spanish text
   spanish_text = "Hola. ¿Cómo estás? El tiempo es bueno."
   # First: python -m spacy download es_core_news_sm
   spanish_sentences = split_sentences(spanish_text, language_model="es_core_news_sm")

Integration with pandas
-----------------------

Process text data in DataFrames:

.. code-block:: python

   import pandas as pd
   from phrasplit import split_sentences, split_clauses

   # Sample data
   data = {
       "id": [1, 2, 3],
       "text": [
           "Hello world. How are you?",
           "The cat sat on the mat, and the dog barked.",
           "Dr. Smith arrived. He was late, unfortunately."
       ]
   }
   df = pd.DataFrame(data)

   # Add sentence count
   df["sentence_count"] = df["text"].apply(lambda x: len(split_sentences(x)))

   # Add clause count
   df["clause_count"] = df["text"].apply(lambda x: len(split_clauses(x)))

   # Explode into one row per sentence
   df_sentences = df.assign(
       sentence=df["text"].apply(split_sentences)
   ).explode("sentence")

   print(df_sentences)
