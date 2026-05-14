
import contextlib
import subprocess
import sys
from typing import Annotated

import typer

app = typer.Typer(help="Run pypff test suite.")

@app.command()
def all(
    lint: Annotated[bool, typer.Option("--lint", help="Run linters (Ruff/MyPy)")] = False,
    cov: Annotated[bool, typer.Option("--cov", help="Enable coverage reporting")] = False,
) -> None:
    """Run all pypff tests (Unit + Logic + Legacy)."""
    if lint:
        run_lint()

    cov_args = ["--cov=src/pypff", "--cov-report=term-missing"] if cov else []

    print("Running Tier 1 (Unit) tests...")
    subprocess.run([sys.executable, "-m", "pytest", *cov_args, "src/ci/tier1_unit"], check=True)
    
    if cov:
        cov_args.append("--cov-append")

    print("Running Tier 2 (Logic) tests...")
    subprocess.run([sys.executable, "-m", "pytest", *cov_args, "src/ci/tier2_logic"], check=True)
    
    print("Running Legacy Integration tests...")
    subprocess.run([sys.executable, "-m", "pytest", *cov_args, "src/ci/legacy_tests"], check=True)

@app.command()
def unit() -> None:
    """Run Tier 1 (Unit) tests."""
    print("Running Tier 1 (Unit) tests...")
    subprocess.run([sys.executable, "-m", "pytest", "src/ci/tier1_unit"], check=True)

@app.command()
def logic() -> None:
    """Run Tier 2 (Logic) tests."""
    print("Running Tier 2 (Logic) tests...")
    subprocess.run([sys.executable, "-m", "pytest", "src/ci/tier2_logic"], check=True)

@app.command()
def legacy() -> None:
    """Run Legacy Integration tests."""
    print("Running Legacy Integration tests...")
    subprocess.run([sys.executable, "-m", "pytest", "src/ci/legacy_tests"], check=True)

@app.command()
def lint() -> None:
    """Run linters (Ruff/MyPy)."""
    run_lint()

def run_lint() -> None:
    print("Running linters...")
    with contextlib.suppress(subprocess.CalledProcessError):
        subprocess.run([sys.executable, "-m", "ruff", "check", "--fix", "."], check=True)
        subprocess.run([sys.executable, "-m", "mypy", "src"], check=True)
