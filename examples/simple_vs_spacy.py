"""Example demonstrating simple mode vs spaCy mode.

This script shows how to use phrasplit with and without spaCy.
"""

from phrasplit import split_sentences

# Sample text with common challenges
text = """Dr. Smith has a Ph.D. in Chemistry. She said, "It's working!"
The results were impressive. Visit www.example.com for more info.
U.S.A. is a big country. The research was published in Nature, Vol. 42."""

print("=" * 70)
print("Phrasplit: Simple Mode vs spaCy Mode Demo")
print("=" * 70)

# Auto-detection (uses spaCy if available)
print("\n1. AUTO-DETECTION MODE (recommended)")
print("-" * 70)
sentences = split_sentences(text)
print(f"Found {len(sentences)} sentences:")
for i, sent in enumerate(sentences, 1):
    print(f"  {i}. {sent}")

# Force simple mode
print("\n2. SIMPLE MODE (fast, no ML dependencies)")
print("-" * 70)
sentences_simple = split_sentences(text, use_spacy=False)
print(f"Found {len(sentences_simple)} sentences:")
for i, sent in enumerate(sentences_simple, 1):
    print(f"  {i}. {sent}")

# Compare
print("\n" + "=" * 70)
print("COMPARISON")
print("=" * 70)
same_count = len(sentences) == len(sentences_simple)
print(f"Both modes found the same number of sentences: {same_count}")
print("\nKey differences:")
print("  • Simple mode: ~60x faster, no ML dependencies")
print("  • spaCy mode: Best accuracy for complex/informal text")
print("  • For most well-formatted text: results are nearly identical")
print("\nRecommendation: Use auto-detection and let phrasplit choose!")
print("=" * 70)
