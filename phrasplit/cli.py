"""Command-line interface for phrasplit."""

import sys
from pathlib import Path

import click
from rich.console import Console

from .splitter import split_clauses, split_long_lines, split_paragraphs, split_sentences

console = Console()
error_console = Console(stderr=True)


def read_input(input_file: str | None) -> str:
    """Read input from file or stdin.

    Args:
        input_file: Path to input file, '-' for stdin, or None for stdin

    Returns:
        Text content
    """
    if input_file is None or input_file == "-":
        return sys.stdin.read()
    return Path(input_file).read_text(encoding="utf-8")


def write_output(text: str, output: Path | None, use_rich: bool = True) -> None:
    """Write output to file or stdout.

    Args:
        text: Text to write
        output: Output file path or None for stdout
        use_rich: Whether to use rich console for stdout
    """
    if output:
        output.write_text(text, encoding="utf-8")
        error_console.print(f"[green]Output written to {output}[/green]")
    elif use_rich:
        console.print(text)
    else:
        print(text)


@click.group()
@click.version_option()
def main() -> None:
    """Phrasplit - Split text into sentences, clauses, or paragraphs."""
    pass


@main.command()
@click.argument("input_file", required=False, default=None)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    help="Output file (default: stdout)",
)
@click.option(
    "-m",
    "--model",
    default="en_core_web_sm",
    help="spaCy language model (default: en_core_web_sm)",
)
def sentences(
    input_file: str | None,
    output: Path | None,
    model: str,
) -> None:
    """Split text into sentences.

    INPUT_FILE: Path to input file, or '-' for stdin. Reads from stdin if omitted.
    """
    try:
        text = read_input(input_file)
    except FileNotFoundError:
        error_console.print(f"[red]Error:[/red] File not found: {input_file}")
        sys.exit(1)

    try:
        result = split_sentences(text, language_model=model)
    except (ImportError, OSError) as e:
        error_console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    output_text = "\n".join(result)
    write_output(output_text, output)


@main.command()
@click.argument("input_file", required=False, default=None)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    help="Output file (default: stdout)",
)
@click.option(
    "-m",
    "--model",
    default="en_core_web_sm",
    help="spaCy language model (default: en_core_web_sm)",
)
def clauses(
    input_file: str | None,
    output: Path | None,
    model: str,
) -> None:
    """Split text into clauses (at commas).

    INPUT_FILE: Path to input file, or '-' for stdin. Reads from stdin if omitted.
    """
    try:
        text = read_input(input_file)
    except FileNotFoundError:
        error_console.print(f"[red]Error:[/red] File not found: {input_file}")
        sys.exit(1)

    try:
        result = split_clauses(text, language_model=model)
    except (ImportError, OSError) as e:
        error_console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    output_text = "\n".join(result)
    write_output(output_text, output)


@main.command()
@click.argument("input_file", required=False, default=None)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    help="Output file (default: stdout)",
)
def paragraphs(
    input_file: str | None,
    output: Path | None,
) -> None:
    """Split text into paragraphs.

    INPUT_FILE: Path to input file, or '-' for stdin. Reads from stdin if omitted.
    """
    try:
        text = read_input(input_file)
    except FileNotFoundError:
        error_console.print(f"[red]Error:[/red] File not found: {input_file}")
        sys.exit(1)

    result = split_paragraphs(text)
    output_text = "\n\n".join(result)
    write_output(output_text, output)


@main.command()
@click.argument("input_file", required=False, default=None)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    help="Output file (default: stdout)",
)
@click.option(
    "-l",
    "--max-length",
    default=80,
    type=click.IntRange(min=1),
    help="Maximum line length (default: 80, must be >= 1)",
)
@click.option(
    "-m",
    "--model",
    default="en_core_web_sm",
    help="spaCy language model (default: en_core_web_sm)",
)
def longlines(
    input_file: str | None,
    output: Path | None,
    max_length: int,
    model: str,
) -> None:
    """Split long lines at sentence/clause boundaries.

    INPUT_FILE: Path to input file, or '-' for stdin. Reads from stdin if omitted.
    """
    try:
        text = read_input(input_file)
    except FileNotFoundError:
        error_console.print(f"[red]Error:[/red] File not found: {input_file}")
        sys.exit(1)

    try:
        result = split_long_lines(text, max_length=max_length, language_model=model)
    except (ImportError, OSError, ValueError) as e:
        error_console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    output_text = "\n".join(result)
    write_output(output_text, output)


if __name__ == "__main__":
    main()
