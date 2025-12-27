"""Command-line interface for phrasplit."""

import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console

from .splitter import split_clauses, split_long_lines, split_paragraphs, split_sentences

console = Console()
error_console = Console(stderr=True)


@click.group()
@click.version_option()
def main() -> None:
    """Phrasplit - Split text into sentences, clauses, or paragraphs."""
    pass


@main.command()
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
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
    input_file: Path,
    output: Optional[Path],
    model: str,
) -> None:
    """Split text into sentences."""
    text = input_file.read_text(encoding="utf-8")

    try:
        result = split_sentences(text, language_model=model)
    except (ImportError, OSError) as e:
        error_console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    output_text = "\n".join(result)

    if output:
        output.write_text(output_text, encoding="utf-8")
        console.print(f"[green]Output written to {output}[/green]")
    else:
        console.print(output_text)


@main.command()
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
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
    input_file: Path,
    output: Optional[Path],
    model: str,
) -> None:
    """Split text into clauses (at commas, semicolons, colons, dashes)."""
    text = input_file.read_text(encoding="utf-8")

    try:
        result = split_clauses(text, language_model=model)
    except (ImportError, OSError) as e:
        error_console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    output_text = "\n".join(result)

    if output:
        output.write_text(output_text, encoding="utf-8")
        console.print(f"[green]Output written to {output}[/green]")
    else:
        console.print(output_text)


@main.command()
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    help="Output file (default: stdout)",
)
def paragraphs(
    input_file: Path,
    output: Optional[Path],
) -> None:
    """Split text into paragraphs."""
    text = input_file.read_text(encoding="utf-8")

    result = split_paragraphs(text)
    output_text = "\n\n".join(result)

    if output:
        output.write_text(output_text, encoding="utf-8")
        console.print(f"[green]Output written to {output}[/green]")
    else:
        console.print(output_text)


@main.command()
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
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
    type=int,
    help="Maximum line length (default: 80)",
)
@click.option(
    "-m",
    "--model",
    default="en_core_web_sm",
    help="spaCy language model (default: en_core_web_sm)",
)
def longlines(
    input_file: Path,
    output: Optional[Path],
    max_length: int,
    model: str,
) -> None:
    """Split long lines at sentence/clause boundaries."""
    text = input_file.read_text(encoding="utf-8")

    try:
        result = split_long_lines(text, max_length=max_length, language_model=model)
    except (ImportError, OSError) as e:
        error_console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    output_text = "\n".join(result)

    if output:
        output.write_text(output_text, encoding="utf-8")
        console.print(f"[green]Output written to {output}[/green]")
    else:
        console.print(output_text)


if __name__ == "__main__":
    main()
