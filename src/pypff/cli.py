import typer
from typing import Optional
import subprocess
import os

app = typer.Typer(help="Pypff Management CLI")

@app.command()
def test(
    tier: str = typer.Argument("unit", help="Test tier to run: unit, logic, or all"),
    lint: bool = typer.Option(False, "--lint", help="Run linters (Ruff/MyPy)"),
):
    """Run pypff test suite."""
    if lint:
        print("Running linters...")
        subprocess.run(["ruff", "check", "."], check=True)
        subprocess.run(["mypy", "src"], check=True)

    if tier == "unit" or tier == "all":
        print("Running Tier 1 (Unit) tests...")
        subprocess.run(["pytest", "src/ci/tier1_unit"], check=True)

    if tier == "logic" or tier == "all":
        print("Running Tier 2 (Logic) tests...")
        subprocess.run(["pytest", "src/ci/tier2_logic"], check=True)

if __name__ == "__main__":
    app()
