"""CLI for the lab."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
import yaml

from .graph import build_graph
from .metrics import MetricsReport, metric_from_state, summarize_metrics, write_metrics
from .persistence import build_checkpointer
from .report import write_report
from .scenarios import load_scenarios
from .state import initial_state

app = typer.Typer(no_args_is_help=True)


@app.command("run-scenarios")
def run_scenarios(
    config: Annotated[Path, typer.Option("--config")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Run all grading scenarios and write metrics JSON."""
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    scenarios = load_scenarios(cfg["scenarios_path"])
    checkpointer_kind = cfg.get("checkpointer", "memory")
    checkpointer = build_checkpointer(checkpointer_kind, cfg.get("database_url"))
    graph = build_graph(checkpointer=checkpointer)
    metrics = []
    for scenario in scenarios:
        state = initial_state(scenario)
        run_config = {"configurable": {"thread_id": state["thread_id"]}}
        final_state = graph.invoke(state, config=run_config)
        metrics.append(metric_from_state(final_state, scenario.expected_route.value, scenario.requires_approval))

    # resume_success=True when using a durable checkpointer (sqlite/postgres)
    resume_success = checkpointer_kind in ("sqlite", "postgres")
    report = summarize_metrics(metrics, resume_success=resume_success)
    write_metrics(report, output)
    if cfg.get("report_path"):
        write_report(report, cfg["report_path"])
    typer.echo(f"Wrote metrics to {output}")
    typer.echo(f"Success rate: {report.success_rate:.2%} ({sum(1 for m in metrics if m.success)}/{len(metrics)})")


@app.command("validate-metrics")
def validate_metrics(metrics: Annotated[Path, typer.Option("--metrics")]) -> None:
    """Validate metrics JSON schema for grading."""
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    report = MetricsReport.model_validate(payload)
    if report.total_scenarios < 6:
        raise typer.BadParameter("Expected at least 6 scenarios")
    typer.echo(f"Metrics valid. success_rate={report.success_rate:.2%}")


@app.command("make-diagram")
def make_diagram(
    output: Annotated[Path, typer.Option("--output")] = Path("outputs/graph.md"),
) -> None:
    """Export the graph as a Mermaid diagram to outputs/graph.md."""
    graph = build_graph()
    mermaid = graph.get_graph().draw_mermaid()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(f"```mermaid\n{mermaid}\n```\n", encoding="utf-8")
    typer.echo(f"Diagram written to {output}")
    typer.echo(mermaid)


@app.command("time-travel")
def time_travel(
    thread_id: Annotated[str, typer.Option("--thread-id")],
    config: Annotated[Path, typer.Option("--config")] = Path("configs/lab.yaml"),
) -> None:
    """Replay state history for a thread using get_state_history() (time travel demo)."""
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    checkpointer_kind = cfg.get("checkpointer", "memory")
    if checkpointer_kind == "memory":
        typer.echo("Time travel requires a durable checkpointer. Set checkpointer: sqlite in lab.yaml", err=True)
        raise typer.Exit(1)

    checkpointer = build_checkpointer(checkpointer_kind, cfg.get("database_url"))
    graph = build_graph(checkpointer=checkpointer)
    run_config = {"configurable": {"thread_id": thread_id}}

    history = list(graph.get_state_history(run_config))
    if not history:
        typer.echo(f"No checkpoints found for thread_id={thread_id!r}")
        raise typer.Exit(1)

    typer.echo(f"Found {len(history)} checkpoint(s) for thread_id={thread_id!r}\n")
    for i, snapshot in enumerate(reversed(history)):
        step = snapshot.metadata.get("step", i)
        next_nodes = snapshot.next
        route = snapshot.values.get("route", "")
        attempt = snapshot.values.get("attempt", 0)
        typer.echo(f"  step={step:3d}  next={next_nodes}  route={route!r}  attempt={attempt}")


if __name__ == "__main__":
    app()
