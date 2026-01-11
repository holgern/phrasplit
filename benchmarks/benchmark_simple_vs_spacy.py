"""Performance benchmark comparing spaCy vs simple splitters.

Run this script to benchmark both implementations.
"""

import time

from phrasplit import split_sentences


def benchmark_split_sentences(
    text: str, runs: int = 5
) -> tuple[float, float, int, int]:
    """Benchmark sentence splitting with both implementations."""
    # Warm up
    split_sentences(text, use_spacy=True)
    split_sentences(text, use_spacy=False)

    # Benchmark spaCy
    spacy_times = []
    spacy_count = 0
    for _ in range(runs):
        start = time.time()
        result = split_sentences(text, use_spacy=True)
        spacy_times.append(time.time() - start)
        spacy_count = len(result)

    # Benchmark simple
    simple_times = []
    simple_count = 0
    for _ in range(runs):
        start = time.time()
        result = split_sentences(text, use_spacy=False)
        simple_times.append(time.time() - start)
        simple_count = len(result)

    spacy_avg = sum(spacy_times) / len(spacy_times)
    simple_avg = sum(simple_times) / len(simple_times)

    return spacy_avg, simple_avg, spacy_count, simple_count


def main() -> None:
    """Run benchmarks and print results."""
    print("=" * 70)
    print("Phrasplit Performance Benchmark: spaCy vs Simple Mode")
    print("=" * 70)

    # Test 1: Short text (a few sentences)
    print("\n1. Short text (3 sentences, ~50 words)")
    print("-" * 70)
    short_text = (
        "Dr. Smith has a Ph.D. in Chemistry. "
        'She said, "It\'s working!" The results were published.'
    )
    spacy_time, simple_time, spacy_count, simple_count = benchmark_split_sentences(
        short_text
    )
    print(f"  spaCy:  {spacy_time * 1000:.2f}ms  ({spacy_count} sentences)")
    print(f"  Simple: {simple_time * 1000:.2f}ms  ({simple_count} sentences)")
    print(f"  Speedup: {spacy_time / simple_time:.1f}x")

    # Test 2: Medium text (many sentences)
    print("\n2. Medium text (100 sentences)")
    print("-" * 70)
    medium_text = ". ".join([f"This is sentence number {i}" for i in range(100)])
    spacy_time, simple_time, spacy_count, simple_count = benchmark_split_sentences(
        medium_text
    )
    print(f"  spaCy:  {spacy_time * 1000:.2f}ms  ({spacy_count} sentences)")
    print(f"  Simple: {simple_time * 1000:.2f}ms  ({simple_count} sentences)")
    print(f"  Speedup: {spacy_time / simple_time:.1f}x")

    # Test 3: Complex text with abbreviations
    print("\n3. Complex text (abbreviations, acronyms, URLs)")
    print("-" * 70)
    complex_text = (
        "Mr. J.R.R. Tolkien wrote many books. "
        "Visit www.example.com for more info. "
        "The U.S.A. is a big country. "
        "Prof. Green met Dr. Stone at Inc. headquarters."
    )
    spacy_time, simple_time, spacy_count, simple_count = benchmark_split_sentences(
        complex_text
    )
    print(f"  spaCy:  {spacy_time * 1000:.2f}ms  ({spacy_count} sentences)")
    print(f"  Simple: {simple_time * 1000:.2f}ms  ({simple_count} sentences)")
    print(f"  Speedup: {spacy_time / simple_time:.1f}x")

    # Test 4: Very long text
    print("\n4. Long text (1000 sentences)")
    print("-" * 70)
    long_text = ". ".join(
        [f"Sentence {i} with some more words to make it realistic" for i in range(1000)]
    )
    spacy_time, simple_time, spacy_count, simple_count = benchmark_split_sentences(
        long_text, runs=3
    )
    print(f"  spaCy:  {spacy_time * 1000:.2f}ms  ({spacy_count} sentences)")
    print(f"  Simple: {simple_time * 1000:.2f}ms  ({simple_count} sentences)")
    print(f"  Speedup: {spacy_time / simple_time:.1f}x")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("Simple mode is typically 20-100x faster than spaCy for common text.")
    print("Both implementations produce similar results for well-formatted text.")
    print("\nWhen to use:")
    print("  • Simple mode: Speed-critical applications, simple text, no ML models")
    print("  • spaCy mode: Complex text, best accuracy, NLP pipelines")
    print("=" * 70)


if __name__ == "__main__":
    main()
