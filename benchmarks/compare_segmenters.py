#!/usr/bin/env python
"""Compare sentence segmentation between two segmenters.

Finds sentences that:
1. Segmenter A gets right but Segmenter B gets wrong (B regressions)
2. Segmenter B gets right but Segmenter A gets wrong (B improvements)

This helps identify where corrections or changes between segmenters
help or hurt sentence boundary detection.

Usage:
    python compare_segmenters.py gold.txt segmenter_a.txt segmenter_b.txt
    python compare_segmenters.py gold.txt spacy.out phrasplit.out --names spacy phrasplit
    python compare_segmenters.py gold.txt a.out b.out -o comparison_results.txt

Example with benchmark files:
    python compare_segmenters.py testsets/UD_English.dataset.gold \\
        outfiles/UD_en_spacy_lg.all.out \\
        outfiles/UD_en_phrasplit_lg.all.out \\
        --names spacy phrasplit
"""

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BoundaryResult:
    """Result of checking a boundary for a specific segmenter."""

    position: int  # Position in stripped text (no spaces)
    original_position: int  # Position in original text
    found: bool  # Whether the segmenter found this boundary
    context: str  # Context around the boundary


@dataclass
class BoundaryComparison:
    """Comparison of a single boundary between two segmenters."""

    gold_sentence_num: int  # Sentence number in gold (1-based)
    position: int  # Position in original gold text
    context: str  # Context around the boundary
    segmenter_a_found: bool
    segmenter_b_found: bool

    @property
    def a_only(self) -> bool:
        """True if only segmenter A found this boundary correctly."""
        return self.segmenter_a_found and not self.segmenter_b_found

    @property
    def b_only(self) -> bool:
        """True if only segmenter B found this boundary correctly."""
        return self.segmenter_b_found and not self.segmenter_a_found

    @property
    def both(self) -> bool:
        """True if both segmenters found this boundary correctly."""
        return self.segmenter_a_found and self.segmenter_b_found

    @property
    def neither(self) -> bool:
        """True if neither segmenter found this boundary."""
        return not self.segmenter_a_found and not self.segmenter_b_found


@dataclass
class ComparisonResult:
    """Full comparison results between two segmenters."""

    segmenter_a_name: str
    segmenter_b_name: str
    total_boundaries: int  # Total gold boundaries

    # Counts
    both_correct: int = 0
    neither_correct: int = 0
    a_only_correct: int = 0  # A got it, B missed it (B regression)
    b_only_correct: int = 0  # B got it, A missed it (B improvement)

    # Detailed comparisons for differences
    a_only_cases: list[BoundaryComparison] = field(default_factory=list)
    b_only_cases: list[BoundaryComparison] = field(default_factory=list)

    # False positive analysis (extra boundaries not in gold)
    a_false_positives: list[tuple[int, str]] = field(
        default_factory=list
    )  # (position, context)
    b_false_positives: list[tuple[int, str]] = field(default_factory=list)
    a_only_false_positives: list[tuple[int, str]] = field(default_factory=list)
    b_only_false_positives: list[tuple[int, str]] = field(default_factory=list)


def get_context(text: str, pos: int, context_size: int = 50) -> str:
    """Extract context around a position in text.

    Args:
        text: The text to extract context from
        pos: Position in text (points to newline character)
        context_size: Number of characters before and after

    Returns:
        Context string with break point marked as |BREAK|
    """
    start = max(0, pos - context_size)
    # Skip the newline at pos to show what comes after the break
    after_start = pos + 1 if pos < len(text) and text[pos] == "\n" else pos
    end = min(len(text), after_start + context_size)

    before = text[start:pos].replace("\n", " / ")
    after = text[after_start:end].replace("\n", " / ")

    return f"...{before} |BREAK| {after}..."


def strip_and_map(text: str) -> tuple[str, list[int]]:
    """Strip spaces from text and create position mapping.

    Args:
        text: Original text

    Returns:
        Tuple of (stripped text, mapping from stripped pos to original pos)
    """
    stripped = text.replace(" ", "")
    pos_map: list[int] = []
    for i, c in enumerate(text):
        if c != " ":
            pos_map.append(i)
    return stripped, pos_map


def get_boundary_positions(stripped_text: str) -> set[int]:
    """Get positions of all newlines (sentence boundaries) in stripped text.

    Args:
        stripped_text: Text with spaces removed

    Returns:
        Set of positions where newlines occur
    """
    return {i for i, c in enumerate(stripped_text) if c == "\n"}


def verify_content_match(gold_content: str, test_content: str, test_name: str) -> None:
    """Verify that content (ignoring newlines) matches between gold and test.

    Args:
        gold_content: Gold text with newlines removed
        test_content: Test text with newlines removed
        test_name: Name of test file for error messages

    Raises:
        ValueError: If content doesn't match
    """
    if gold_content != test_content:
        # Find first difference
        for i, (gc, tc) in enumerate(zip(gold_content, test_content, strict=False)):
            if gc != tc:
                raise ValueError(
                    f"Content mismatch in {test_name} at position {i}:\n"
                    f"  Gold: {gold_content[max(0, i - 20) : i + 20]!r}\n"
                    f"  Test: {test_content[max(0, i - 20) : i + 20]!r}"
                )
        if len(gold_content) != len(test_content):
            raise ValueError(
                f"Content length mismatch in {test_name}: "
                f"gold={len(gold_content)}, test={len(test_content)}"
            )


def compare_segmenters(
    gold_text: str,
    segmenter_a_text: str,
    segmenter_b_text: str,
    segmenter_a_name: str = "Segmenter A",
    segmenter_b_name: str = "Segmenter B",
) -> ComparisonResult:
    """Compare two segmenters against a gold standard.

    Args:
        gold_text: Gold standard text (one sentence per line)
        segmenter_a_text: Output from segmenter A
        segmenter_b_text: Output from segmenter B
        segmenter_a_name: Display name for segmenter A
        segmenter_b_name: Display name for segmenter B

    Returns:
        ComparisonResult with detailed analysis
    """
    # Strip spaces and create position mappings
    gold_stripped, gold_pos_map = strip_and_map(gold_text)
    a_stripped, a_pos_map = strip_and_map(segmenter_a_text)
    b_stripped, b_pos_map = strip_and_map(segmenter_b_text)

    # Verify content matches (ignoring boundaries)
    gold_content = gold_stripped.replace("\n", "")
    a_content = a_stripped.replace("\n", "")
    b_content = b_stripped.replace("\n", "")

    verify_content_match(gold_content, a_content, segmenter_a_name)
    verify_content_match(gold_content, b_content, segmenter_b_name)

    # Get boundary positions in stripped text
    gold_boundaries = get_boundary_positions(gold_stripped)
    a_boundaries = get_boundary_positions(a_stripped)
    b_boundaries = get_boundary_positions(b_stripped)

    result = ComparisonResult(
        segmenter_a_name=segmenter_a_name,
        segmenter_b_name=segmenter_b_name,
        total_boundaries=len(gold_boundaries),
    )

    # Compare each gold boundary
    gold_sent_num = 1
    for pos in sorted(gold_boundaries):
        a_found = pos in a_boundaries
        b_found = pos in b_boundaries

        # Get context from original gold text
        orig_pos = gold_pos_map[pos] if pos < len(gold_pos_map) else len(gold_text)
        context = get_context(gold_text, orig_pos)

        comparison = BoundaryComparison(
            gold_sentence_num=gold_sent_num,
            position=orig_pos,
            context=context,
            segmenter_a_found=a_found,
            segmenter_b_found=b_found,
        )

        if comparison.both:
            result.both_correct += 1
        elif comparison.neither:
            result.neither_correct += 1
        elif comparison.a_only:
            result.a_only_correct += 1
            result.a_only_cases.append(comparison)
        elif comparison.b_only:
            result.b_only_correct += 1
            result.b_only_cases.append(comparison)

        gold_sent_num += 1

    # Analyze false positives (boundaries in test but not in gold)
    # We need to map test positions to gold positions for comparison
    # Since content matches, we can compare positions after removing newlines

    # Build mapping: for each position in stripped text (no spaces),
    # what position is it in content (no spaces, no newlines)?
    def build_content_pos_map(stripped: str) -> list[int]:
        """Map stripped positions to content positions (excluding newlines)."""
        content_pos = 0
        pos_map = []
        for c in stripped:
            if c == "\n":
                pos_map.append(content_pos)  # newline maps to position after
            else:
                pos_map.append(content_pos)
                content_pos += 1
        return pos_map

    gold_to_content = build_content_pos_map(gold_stripped)
    a_to_content = build_content_pos_map(a_stripped)
    b_to_content = build_content_pos_map(b_stripped)

    # Convert gold boundaries to content positions
    gold_boundary_content_pos = {gold_to_content[p] for p in gold_boundaries}

    # Find false positives for each segmenter
    a_fp_content_pos = set()
    for pos in a_boundaries:
        content_pos = a_to_content[pos]
        if content_pos not in gold_boundary_content_pos:
            a_fp_content_pos.add(content_pos)
            orig_pos = a_pos_map[pos] if pos < len(a_pos_map) else len(segmenter_a_text)
            result.a_false_positives.append(
                (orig_pos, get_context(segmenter_a_text, orig_pos))
            )

    b_fp_content_pos = set()
    for pos in b_boundaries:
        content_pos = b_to_content[pos]
        if content_pos not in gold_boundary_content_pos:
            b_fp_content_pos.add(content_pos)
            orig_pos = b_pos_map[pos] if pos < len(b_pos_map) else len(segmenter_b_text)
            result.b_false_positives.append(
                (orig_pos, get_context(segmenter_b_text, orig_pos))
            )

    # Find false positives unique to each segmenter
    a_only_fp = a_fp_content_pos - b_fp_content_pos
    b_only_fp = b_fp_content_pos - a_fp_content_pos

    # Get contexts for unique false positives
    for pos in a_boundaries:
        content_pos = a_to_content[pos]
        if content_pos in a_only_fp:
            orig_pos = a_pos_map[pos] if pos < len(a_pos_map) else len(segmenter_a_text)
            result.a_only_false_positives.append(
                (orig_pos, get_context(segmenter_a_text, orig_pos))
            )

    for pos in b_boundaries:
        content_pos = b_to_content[pos]
        if content_pos in b_only_fp:
            orig_pos = b_pos_map[pos] if pos < len(b_pos_map) else len(segmenter_b_text)
            result.b_only_false_positives.append(
                (orig_pos, get_context(segmenter_b_text, orig_pos))
            )

    return result


def format_report(result: ComparisonResult) -> str:
    """Format a comparison result as a human-readable report.

    Args:
        result: ComparisonResult to format

    Returns:
        Formatted report string
    """
    lines = []
    sep = "=" * 80

    lines.append(sep)
    lines.append("SEGMENTER COMPARISON REPORT")
    lines.append(sep)
    lines.append(f"Segmenter A: {result.segmenter_a_name}")
    lines.append(f"Segmenter B: {result.segmenter_b_name}")
    lines.append("")

    # Summary statistics
    lines.append(sep)
    lines.append("SUMMARY: Sentence Boundary Detection (True Positives)")
    lines.append(sep)
    lines.append(f"Total gold boundaries:     {result.total_boundaries}")
    lines.append(f"Both correct:              {result.both_correct}")
    lines.append(f"Neither correct:           {result.neither_correct}")
    lines.append(
        f"{result.segmenter_a_name} only (B regression): {result.a_only_correct}"
    )
    lines.append(
        f"{result.segmenter_b_name} only (B improvement): {result.b_only_correct}"
    )
    lines.append("")

    # Calculate metrics for each
    a_tp = result.both_correct + result.a_only_correct
    a_fn = result.neither_correct + result.b_only_correct
    b_tp = result.both_correct + result.b_only_correct
    b_fn = result.neither_correct + result.a_only_correct

    a_fp = len(result.a_false_positives)
    b_fp = len(result.b_false_positives)

    a_precision = a_tp / (a_tp + a_fp) if (a_tp + a_fp) > 0 else 0.0
    a_recall = a_tp / (a_tp + a_fn) if (a_tp + a_fn) > 0 else 0.0
    a_f1 = (
        2 * a_precision * a_recall / (a_precision + a_recall)
        if (a_precision + a_recall) > 0
        else 0.0
    )

    b_precision = b_tp / (b_tp + b_fp) if (b_tp + b_fp) > 0 else 0.0
    b_recall = b_tp / (b_tp + b_fn) if (b_tp + b_fn) > 0 else 0.0
    b_f1 = (
        2 * b_precision * b_recall / (b_precision + b_recall)
        if (b_precision + b_recall) > 0
        else 0.0
    )

    lines.append(sep)
    lines.append("METRICS COMPARISON")
    lines.append(sep)
    lines.append(
        f"{'Metric':<20} {result.segmenter_a_name:<15} {result.segmenter_b_name:<15} {'Difference':<15}"
    )
    lines.append("-" * 65)
    lines.append(f"{'True Positives':<20} {a_tp:<15} {b_tp:<15} {b_tp - a_tp:+d}")
    lines.append(f"{'False Negatives':<20} {a_fn:<15} {b_fn:<15} {b_fn - a_fn:+d}")
    lines.append(f"{'False Positives':<20} {a_fp:<15} {b_fp:<15} {b_fp - a_fp:+d}")
    lines.append(
        f"{'Precision':<20} {a_precision:<15.4f} {b_precision:<15.4f} {b_precision - a_precision:+.4f}"
    )
    lines.append(
        f"{'Recall':<20} {a_recall:<15.4f} {b_recall:<15.4f} {b_recall - a_recall:+.4f}"
    )
    lines.append(f"{'F1-Score':<20} {a_f1:<15.4f} {b_f1:<15.4f} {b_f1 - a_f1:+.4f}")
    lines.append("")

    # False positive summary
    lines.append(sep)
    lines.append("FALSE POSITIVES SUMMARY")
    lines.append(sep)
    lines.append(f"{result.segmenter_a_name} total FP: {len(result.a_false_positives)}")
    lines.append(f"{result.segmenter_b_name} total FP: {len(result.b_false_positives)}")
    lines.append(
        f"{result.segmenter_a_name} only FP (B fixed): {len(result.a_only_false_positives)}"
    )
    lines.append(
        f"{result.segmenter_b_name} only FP (B introduced): {len(result.b_only_false_positives)}"
    )
    lines.append("")

    # Detailed cases: B regressions (A got it right, B missed it)
    if result.a_only_cases:
        lines.append(sep)
        lines.append(
            f"B REGRESSIONS: {result.segmenter_a_name} correct, "
            f"{result.segmenter_b_name} missed ({len(result.a_only_cases)} cases)"
        )
        lines.append(sep)
        lines.append(
            "These are boundaries that B failed to detect but A detected correctly."
        )
        lines.append("")
        for i, case in enumerate(result.a_only_cases, 1):
            lines.append(f"[{i}] Gold sentence {case.gold_sentence_num}")
            lines.append(f"    {case.context}")
            lines.append("")

    # Detailed cases: B improvements (B got it right, A missed it)
    if result.b_only_cases:
        lines.append(sep)
        lines.append(
            f"B IMPROVEMENTS: {result.segmenter_b_name} correct, "
            f"{result.segmenter_a_name} missed ({len(result.b_only_cases)} cases)"
        )
        lines.append(sep)
        lines.append(
            "These are boundaries that B detected correctly but A failed to detect."
        )
        lines.append("")
        for i, case in enumerate(result.b_only_cases, 1):
            lines.append(f"[{i}] Gold sentence {case.gold_sentence_num}")
            lines.append(f"    {case.context}")
            lines.append("")

    # Detailed cases: B-only false positives (B introduced errors)
    if result.b_only_false_positives:
        lines.append(sep)
        lines.append(
            f"B INTRODUCED FALSE POSITIVES: "
            f"{result.segmenter_b_name} only ({len(result.b_only_false_positives)} cases)"
        )
        lines.append(sep)
        lines.append(
            f"These are incorrect boundaries that {result.segmenter_b_name} added "
            f"but {result.segmenter_a_name} did not."
        )
        lines.append("")
        for i, (pos, context) in enumerate(result.b_only_false_positives, 1):
            lines.append(f"[{i}] Position {pos}")
            lines.append(f"    {context}")
            lines.append("")

    # Detailed cases: A-only false positives (B fixed errors)
    if result.a_only_false_positives:
        lines.append(sep)
        lines.append(
            f"B FIXED FALSE POSITIVES: "
            f"{result.segmenter_a_name} only ({len(result.a_only_false_positives)} cases)"
        )
        lines.append(sep)
        lines.append(
            f"These are incorrect boundaries that {result.segmenter_a_name} added "
            f"but {result.segmenter_b_name} correctly avoided."
        )
        lines.append("")
        for i, (pos, context) in enumerate(result.a_only_false_positives, 1):
            lines.append(f"[{i}] Position {pos}")
            lines.append(f"    {context}")
            lines.append("")

    return "\n".join(lines)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog=os.path.basename(sys.argv[0]),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )

    parser.add_argument(
        "gold",
        type=str,
        help="Gold standard file (one sentence per line)",
    )
    parser.add_argument(
        "segmenter_a",
        type=str,
        help="Output file from segmenter A",
    )
    parser.add_argument(
        "segmenter_b",
        type=str,
        help="Output file from segmenter B",
    )
    parser.add_argument(
        "--names",
        "-n",
        nargs=2,
        metavar=("A_NAME", "B_NAME"),
        default=["Segmenter A", "Segmenter B"],
        help="Names for the two segmenters (default: 'Segmenter A' 'Segmenter B')",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        metavar="FILE",
        help="Output file for the comparison report (default: print to stdout)",
    )

    args = parser.parse_args()

    # Read files
    try:
        with open(args.gold, encoding="utf-8") as f:
            gold_text = f.read()
        with open(args.segmenter_a, encoding="utf-8") as f:
            segmenter_a_text = f.read()
        with open(args.segmenter_b, encoding="utf-8") as f:
            segmenter_b_text = f.read()
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Compare
    try:
        result = compare_segmenters(
            gold_text,
            segmenter_a_text,
            segmenter_b_text,
            segmenter_a_name=args.names[0],
            segmenter_b_name=args.names[1],
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Format report
    report = format_report(result)

    # Output
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Comparison report saved to: {output_path}")

        # Also print summary to stdout
        print()
        print(f"=== SUMMARY ===")
        print(f"Total gold boundaries: {result.total_boundaries}")
        print(f"Both correct: {result.both_correct}")
        print(f"Neither correct: {result.neither_correct}")
        print(f"{args.names[0]} only (B regression): {result.a_only_correct}")
        print(f"{args.names[1]} only (B improvement): {result.b_only_correct}")
        print(f"{args.names[0]} false positives: {len(result.a_false_positives)}")
        print(f"{args.names[1]} false positives: {len(result.b_false_positives)}")
    else:
        print(report)


if __name__ == "__main__":
    main()
