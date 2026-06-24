"""CLI command to verify a Lit2Vec pipeline output directory."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from lit2vec.cli.console import console, print_header, print_success, print_warning
from lit2vec.config import DEFAULT_EMBEDDING_MODEL, DEFAULT_VEC_DIM
from lit2vec.proctor import ProctorError, proctor_pipeline_output, print_report


def verify_output(
    output_dir: Path = typer.Option(
        ..., "--output-dir", "-o", help="Pipeline output directory to verify."
    ),
    expect_zip: bool = typer.Option(
        False, "--expect-zip", "-z", help="Require a data.zip archive in the output directory."
    ),
    model_name: str = typer.Option(
        DEFAULT_EMBEDDING_MODEL, "--model-name", "-m", help="Model name used to derive the Annoy filename."
    ),
    vec_dim: int = typer.Option(
        DEFAULT_VEC_DIM, "--vec-dim", "-D", help="Expected embedding dimensionality."
    ),
):
    """Verify a Lit2Vec pipeline output directory with the proctor agent."""
    print_header("Lit2Vec Proctor", f"Verifying: {output_dir}")

    report = proctor_pipeline_output(
        output_dir,
        expect_zip=expect_zip,
        model_name=model_name,
        vec_dim=vec_dim,
    )

    print_report(report)

    if report.passed:
        console.print()
        print_success("All proctor checks passed")
        raise typer.Exit(0)
    else:
        console.print()
        print_warning(f"{len(report.failed)} proctor check(s) failed")
        raise typer.Exit(1)
