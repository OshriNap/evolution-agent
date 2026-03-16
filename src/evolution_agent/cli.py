"""Click CLI: run, dashboard, resume."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import click

from evolution_agent.core.config import load_config
from evolution_agent.core.engine import EvolutionEngine


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group()
def cli() -> None:
    """Evolution Agent — evolutionary code optimization framework."""


@cli.command()
@click.option("--config", "-c", type=click.Path(exists=True), help="YAML config file")
@click.option("--seed", "-s", multiple=True, help="Seed code files")
@click.option("--evaluator", "-e", type=click.Path(exists=True), help="Evaluator script")
@click.option("--function-spec", "-f", type=str, help="Function specification")
@click.option("--direction", type=click.Choice(["maximize", "minimize"]), default="maximize")
@click.option("--max-generations", "-g", type=int, help="Max generations")
@click.option("--population-size", "-p", type=int, help="Population size")
@click.option("--run-dir", type=click.Path(), help="Output directory for this run")
@click.option("--optimizer", type=click.Choice(["heuristic", "optuna", "none"]), default="heuristic", help="Meta-optimizer type")
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging")
def run(
    config: str | None,
    seed: tuple[str, ...],
    evaluator: str | None,
    function_spec: str | None,
    direction: str,
    max_generations: int | None,
    population_size: int | None,
    run_dir: str | None,
    optimizer: str,
    verbose: bool,
) -> None:
    """Run an evolution optimization."""
    _setup_logging(verbose)

    # Load config
    overrides = {"meta_optimizer_type": optimizer}
    if max_generations:
        overrides["max_generations"] = max_generations
    if population_size:
        overrides["population_size"] = population_size
    if direction:
        overrides["direction"] = direction

    evo_config = load_config(config, overrides)

    # Load seeds
    seeds: list[str] = []
    for s in seed:
        path = Path(s)
        if path.exists():
            seeds.append(path.read_text(encoding="utf-8"))
        else:
            seeds.append(s)

    if not seeds:
        click.echo("Error: At least one seed is required (--seed)", err=True)
        sys.exit(1)

    # Set up evaluator
    if evaluator:
        from evolution_agent.evaluation.subprocess_eval import SubprocessEvaluator
        from evolution_agent.core.types import OptimizationDirection

        eval_instance = SubprocessEvaluator(
            script_path=evaluator,
            function_spec=function_spec or "Evolve a Python function",
            direction=OptimizationDirection(direction),
            timeout_s=evo_config.eval_timeout_s,
        )
    else:
        click.echo("Error: An evaluator is required (--evaluator)", err=True)
        sys.exit(1)

    # Run
    engine = EvolutionEngine(
        config=evo_config,
        evaluator=eval_instance,
        seeds=seeds,
        run_dir=run_dir,
    )

    summary = asyncio.run(engine.run())

    click.echo(f"\nEvolution complete!")
    click.echo(f"  Generations: {summary['total_generations']}")
    click.echo(f"  Best fitness: {summary['best_fitness']:.6f}")
    click.echo(f"  Elapsed: {summary['elapsed_s']:.1f}s")
    click.echo(f"  Run dir: {run_dir or 'runs/'}")


@cli.command()
@click.option("--run-dir", "-d", type=click.Path(exists=True), required=True)
@click.option("--port", "-p", type=int, default=8050)
def dashboard(run_dir: str, port: int) -> None:
    """Launch the web dashboard for a run."""
    from evolution_agent.logging.dashboard import serve_dashboard
    serve_dashboard(run_dir, port)


if __name__ == "__main__":
    cli()
