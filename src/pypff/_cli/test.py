import typer
import subprocess
from typing import Annotated

app = typer.Typer(help="Run pypff test suite.")

@app.command()
def all(
    lint: Annotated[bool, typer.Option("--lint", help="Run linters (Ruff/MyPy)")] = False,
):
    """Run all pypff tests (Unit + Logic + Legacy)."""
    if lint:
        run_lint()

    print("Running Tier 1 (Unit) tests...")
    subprocess.run(["pytest", "src/ci/tier1_unit"], check=True)
    
    print("Running Tier 2 (Logic) tests...")
    subprocess.run(["pytest", "src/ci/tier2_logic"], check=True)
    
    print("Running Legacy Integration tests...")
    subprocess.run(["pytest", "src/ci/legacy_tests"], check=True)

@app.command()
def unit():
    """Run Tier 1 (Unit) tests."""
    print("Running Tier 1 (Unit) tests...")
    subprocess.run(["pytest", "src/ci/tier1_unit"], check=True)

@app.command()
def logic():
    """Run Tier 2 (Logic) tests."""
    print("Running Tier 2 (Logic) tests...")
    subprocess.run(["pytest", "src/ci/tier2_logic"], check=True)

@app.command()
def legacy():
    """Run Legacy Integration tests."""
    print("Running Legacy Integration tests...")
    subprocess.run(["pytest", "src/ci/legacy_tests"], check=True)

@app.command()
def lint():
    """Run linters (Ruff/MyPy)."""
    run_lint()

def run_lint():
    print("Running linters...")
    subprocess.run(["ruff", "check", "."], check=True)
    subprocess.run(["mypy", "src"], check=True)
