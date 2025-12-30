# Phrasplit Benchmarks

This directory contains benchmarking tools to evaluate sentence segmentation performance
against gold standard datasets from
[Universal Dependencies](https://universaldependencies.org/).

The benchmark compares multiple segmenters:

- **spaCy** - Raw spaCy sentence segmentation (parser-based, baseline)
- **phrasplit** - spaCy + phrasplit's preprocessing and post-processing corrections
- **Sentencizer** - spaCy's rule-based sentence segmenter (no model required)

## Quick Start

```bash
# 1. Download English dataset and build test variants
python build-testset.py --download en

# 2. Run evaluation (compares both spaCy and phrasplit)
python runbatcheval.py en

# 3. Clean up when done
python cleanup.py
```

## Scripts

### `build-testset.py`

Downloads CoNLL-U files from Universal Dependencies and creates gold standard test sets.

```bash
# List available datasets (32 languages)
python build-testset.py --list

# Download and build a specific language
python build-testset.py --download en
python build-testset.py --download de

# Download all languages
python build-testset.py --download

# Build test variants from existing .gold files
python build-testset.py
```

Creates three test variants for each gold standard:

- `.gold` - Gold standard (one sentence per line)
- `.all` - Same as gold (trivial test case)
- `.none` - All sentences on one line (hardest test, tests chunking for large texts)
- `.mixed` - Paragraph-like (3-8 sentences per line, realistic test case)

### `runbatcheval.py`

Evaluates and compares spaCy vs phrasplit on test sets. For languages with official
spaCy models, tests all available model sizes (sm, md, lg).

```bash
# Evaluate a single language (tests both segmenters)
python runbatcheval.py en

# Evaluate only raw spaCy
python runbatcheval.py en --spacy

# Evaluate only phrasplit
python runbatcheval.py en --phrasplit

# Evaluate only rule-based Sentencizer
python runbatcheval.py en --sentencizer

# Evaluate all languages
python runbatcheval.py --all

# Quiet mode (summary only)
python runbatcheval.py -q en

# List supported languages and their models
python runbatcheval.py --list

# Skip saving error files
python runbatcheval.py --no-errors en
```

The output includes a comparison table showing the F-measure difference between
phrasplit and raw spaCy, highlighting the improvement from phrasplit's corrections.

### `runeverything.py`

Runs evaluation on all languages and generates a summary report.

```bash
# Run all evaluations
python runeverything.py

# Save results to file
python runeverything.py -o results.txt

# Quiet mode
python runeverything.py -q
```

### `segmenteval.py`

Low-level evaluation script that compares segmenter output against gold standard. By
default, saves error context files for analysis.

```bash
# Basic evaluation (saves error files by default)
python segmenteval.py testsets/UD_English.dataset.gold outfiles/output.out

# Disable error file generation
python segmenteval.py testsets/UD_English.dataset.gold outfiles/output.out --no-errors

# Custom prefix for error files
python segmenteval.py testsets/UD_English.dataset.gold outfiles/output.out \
    --errors-prefix outfiles/my_errors
```

Output:

```
True positives: 10478
False positives: 724
False negatives: 2066
Precision: 0.935
Recall: 0.835
F-measure: 0.883
Saved false positives to: outfiles/output_errors_false_positives.txt
Saved false negatives to: outfiles/output_errors_false_negatives.txt
```

Error files created (by default, derived from test filename):

- `*_false_positives.txt` - Places where the segmenter incorrectly added a sentence
  break
- `*_false_negatives.txt` - Places where the segmenter missed a sentence break

Each error shows context around the break point:

```
[1] ...after the Chernobyl Accident - Lessons learned. |BREAK| which contradict the earlier reports...
```

### `debug_sentence.py`

Debug tool for investigating segmentation issues with specific sentences from a dataset.
Extracts a sentence by number from a gold standard file and shows detailed spaCy
tokenization and sentence boundary information.

```bash
# Debug sentence #65 from English dataset
python debug_sentence.py testsets/UD_English.dataset.gold 65

# Include surrounding sentences for context
python debug_sentence.py testsets/UD_English.dataset.gold 65 --context 2

# Use a different spaCy model
python debug_sentence.py testsets/UD_English.dataset.gold 65 --model en_core_web_lg
```

Output includes:

- **Gold sentences**: The expected sentence boundaries
- **Input text**: The joined text being analyzed
- **spaCy sentences**: How spaCy segments the text
- **Token analysis**: Detailed token information around sentence boundaries

Example output:

```
Gold file: testsets/UD_English.dataset.gold
Sentence number: 65 (of 12543)

================================================================================
GOLD SENTENCES (should be separate):
================================================================================
[65] This is sentence one.
[66] This is sentence two.

================================================================================
INPUT TEXT:
================================================================================
This is sentence one. This is sentence two.

================================================================================
SPACY SENTENCES:
================================================================================
[1] chars 0-22:
    'This is sentence one.'
[2] chars 23-45:
    'This is sentence two.'

================================================================================
TOKEN ANALYSIS (around sentence boundaries):
================================================================================
Token 4: '.'
  is_sent_start: False
  Context: ['sentence', 'one', '.', 'This', 'is']
Token 5: 'This'
  is_sent_start: True
  Context: ['one', '.', 'This', 'is', 'sentence']
```

### `phrasplit_segmenter.py`

Sentence segmenter using the full phrasplit library (`phrasplit.split_sentences()`).
This tests the actual phrasplit functionality including:

- Whitespace normalization
- Hyphenated line break fixing
- Ellipsis protection
- Post-processing corrections (abbreviation merge, URL split)

```bash
# Use default model for the language
python phrasplit_segmenter.py English testsets/UD_English.dataset.none outfiles/output.out

# Use a specific spaCy model
python phrasplit_segmenter.py English testsets/UD_English.dataset.none outfiles/output.out \
    --model en_core_web_lg
```

### `spacy_segmenter.py`

Raw spaCy sentence segmenter (no phrasplit corrections). This provides a baseline to
compare against phrasplit.

```bash
# Use default model for the language
python spacy_segmenter.py English testsets/UD_English.dataset.none outfiles/output.out

# Use a specific spaCy model
python spacy_segmenter.py English testsets/UD_English.dataset.none outfiles/output.out \
    --model en_core_web_lg
```

### `sentencizer_segmenter.py`

Rule-based sentence segmenter using spaCy's `Sentencizer` component. Unlike the parser-
based segmenters, this uses simple punctuation rules and doesn't require a trained
model. Useful for comparing rule-based vs parser-based approaches.

```bash
# Basic usage (uses blank spaCy model with Sentencizer)
python sentencizer_segmenter.py English testsets/UD_English.dataset.none outfiles/output.out

# Customize punctuation characters
python sentencizer_segmenter.py English input.txt output.txt \
    --punct-chars '.!?'
```

The Sentencizer is faster but generally less accurate than parser-based segmentation,
especially for complex sentences with abbreviations or unusual punctuation.

### `cleanup.py`

Removes all downloaded datasets and generated output files.

```bash
# Preview what will be deleted
python cleanup.py --dry-run

# Delete all generated files
python cleanup.py
```

### `compare_segmenters.py`

Compares two segmenters against a gold standard to find where they differ. Useful for
understanding what one segmenter fixes or breaks compared to another.

The script automatically finds files based on language code, segmenter names, model
size, and test variant.

```bash
# List available options (languages, segmenters, model sizes, variants)
python compare_segmenters.py --list

# Compare spaCy vs phrasplit on English with lg model, all variant
python compare_segmenters.py en spacy phrasplit -m lg -v all

# Compare on the hardest test (none variant = all sentences on one line)
python compare_segmenters.py en spacy phrasplit -m lg -v none

# Save output to file
python compare_segmenters.py en spacy phrasplit -m lg -v all -o comparison.txt

# Compare sentencizer vs phrasplit
python compare_segmenters.py en sentencizer phrasplit -m sm -v all

# Override auto-detected file paths (for custom files)
python compare_segmenters.py en spacy phrasplit -m lg -v all \
    --file-a custom_a.out --file-b custom_b.out --gold custom.gold
```

**Required arguments:**

- `lang` - Language code (e.g., `en`, `de`, `fr`)
- `segmenter_a` - First segmenter (`spacy`, `phrasplit`, `sentencizer`)
- `segmenter_b` - Second segmenter
- `-m/--model-size` - Model size (`sm`, `md`, `lg`)
- `-v/--variant` - Test variant (`all`, `none`, `mixed`)

**File path resolution:**

The script automatically constructs file paths:

- Gold: `testsets/UD_{Language}.dataset.gold` (e.g., `UD_English.dataset.gold`)
- Output: `outfiles/UD_{lang}_{segmenter}_{size}.{variant}.out`
- Sentencizer (no model): `outfiles/UD_{lang}_sentencizer.{variant}.out`

Output includes:

- **Metrics comparison**: True positives, false positives/negatives, precision, recall,
  F1 for both segmenters with differences
- **B Regressions**: Boundaries that A got right but B missed
- **B Improvements**: Boundaries that B got right but A missed
- **B Introduced FP**: False positives B added that A didn't have
- **B Fixed FP**: False positives A had that B correctly avoided

Each case includes full context around the boundary:

```
[1] Gold sentence 635
    ...Karzai, Musharraf new regional equations / KABUL: |BREAK| For the past 25 years landlocked Afghanistan...
```

Example summary output:

```
=== SUMMARY ===
Total gold boundaries: 12544
Both correct: 83
Neither correct: 12015
spacy only (B regression): 217
phrasplit only (B improvement): 229
spacy false positives: 451
phrasplit false positives: 409
```

## Directory Structure

After downloading datasets:

```
benchmarks/
  build-testset.py
  cleanup.py
  compare_segmenters.py
  debug_sentence.py
  phrasplit_segmenter.py
  sentencizer_segmenter.py
  spacy_segmenter.py
  runbatcheval.py
  runeverything.py
  segmenteval.py
  testsets/           # Downloaded datasets (not in git)
    UD_English.dataset.gold
    UD_English.dataset.all
    UD_English.dataset.none
    UD_English.dataset.mixed
    ...
  outfiles/           # Evaluation outputs (not in git)
    UD_en_spacy_sm.none.out
    UD_en_phrasplit_sm.none.out
    UD_en_sentencizer.none.out
    spacy_vs_phrasplit_lg_comparison.txt
    ...
```

## Supported Languages

Languages marked with ✓ have official spaCy models and will be tested with sm, md, lg
variants. Others use the default spaCy pipeline.

| Code | Language          | Dataset               | spaCy Models |
| ---- | ----------------- | --------------------- | ------------ |
| bg   | Bulgarian         | UD_Bulgarian-BTB      |              |
| ca   | Catalan           | UD_Catalan-AnCora     | ✓            |
| cnr  | Montenegrin       | UD_Montenegrin-MESUBS |              |
| cs   | Czech             | UD_Czech-PDT          |              |
| da   | Danish            | UD_Danish-DDT         | ✓            |
| de   | German            | UD_German-GSD         | ✓            |
| el   | Greek             | UD_Greek-GDT          | ✓            |
| en   | English           | UD_English-EWT        | ✓            |
| es   | Spanish           | UD_Spanish-GSD        | ✓            |
| et   | Estonian          | UD_Estonian-EDT       |              |
| fi   | Finnish           | UD_Finnish-TDT        | ✓            |
| fr   | French            | UD_French-GSD         | ✓            |
| hr   | Croatian          | UD_Croatian-SET       | ✓            |
| hu   | Hungarian         | UD_Hungarian-Szeged   |              |
| is   | Icelandic         | UD_Icelandic-IcePaHC  |              |
| it   | Italian           | UD_Italian-ISDT       | ✓            |
| ja   | Japanese          | UD_Japanese-GSD       | ✓            |
| ko   | Korean            | UD_Korean-Kaist       | ✓            |
| lt   | Lithuanian        | UD_Lithuanian-ALKSNIS | ✓            |
| lv   | Latvian           | UD_Latvian-LVTB       |              |
| mk   | Macedonian        | UD_Macedonian-MTB     | ✓            |
| mt   | Maltese           | UD_Maltese-MUDT       |              |
| nb   | Norwegian Bokmål  | UD_Norwegian-Bokmaal  | ✓            |
| nl   | Dutch             | UD_Dutch-Alpino       | ✓            |
| nn   | Norwegian Nynorsk | UD_Norwegian-Nynorsk  |              |
| pl   | Polish            | UD_Polish-PDB         | ✓            |
| pt   | Portuguese        | UD_Portuguese-Bosque  | ✓            |
| ro   | Romanian          | UD_Romanian-RRT       | ✓            |
| sk   | Slovak            | UD_Slovak-SNK         |              |
| sl   | Slovenian         | UD_Slovenian-SSJ      | ✓            |
| sq   | Albanian          | UD_Albanian-TSA       |              |
| sr   | Serbian           | UD_Serbian-SET        |              |
| sv   | Swedish           | UD_Swedish-Talbanken  | ✓            |
| tr   | Turkish           | UD_Turkish-IMST       |              |
| uk   | Ukrainian         | UD_Ukrainian-IU       | ✓            |
| zh   | Chinese           | UD_Chinese-GSD        | ✓            |

## Metrics

- **Precision**: Proportion of predicted sentence boundaries that are correct
- **Recall**: Proportion of actual sentence boundaries that were found
- **F-measure**: Harmonic mean of precision and recall

## Example Results (English)

Comparison of raw spaCy vs phrasplit on the `.none` test set (all sentences on one
line):

| Segmenter | Model | Precision | Recall | F-measure |
| --------- | ----- | --------- | ------ | --------- |
| spaCy     | sm    | 0.932     | 0.830  | 0.878     |
| phrasplit | sm    | 0.935     | 0.835  | 0.883     |
| spaCy     | lg    | 0.942     | 0.845  | 0.891     |
| phrasplit | lg    | 0.945     | 0.850  | 0.895     |

Phrasplit's post-processing corrections (abbreviation handling, URL splitting) typically
improve F-measure by 0.3-0.5% over raw spaCy.

Note: `.all` test sets always achieve 1.000 F-measure as they're already properly
segmented.

## Data Source

Datasets are downloaded from
[Universal Dependencies](https://universaldependencies.org/) GitHub repositories. The
gold standard sentences are extracted from the `# text = ` lines in CoNLL-U format
files.
