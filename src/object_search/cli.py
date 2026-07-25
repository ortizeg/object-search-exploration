"""The ``object_search`` command-line entry point (Typer).

Every ``pixi run`` task that is not a quality gate routes through here. Logging is Loguru
only: user-facing lines use ``typer.echo`` (which is not ``print`` and is not banned), while
diagnostic logging uses ``logger``.

Placeholders fail loudly on purpose. ``render-samples`` and ``benchmark`` exit non-zero with
a "later phase implements this" message rather than exiting 0, because a no-op success is
indistinguishable from a silently broken renderer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from loguru import logger

from object_search.inference import models as model_registry
from object_search.log import setup_logging
from object_search.provenance import repo_root

# Default output dirs as module-level singletons -- computed here rather than in an argument
# default so no function call happens in a signature (flake8-bugbear B008).
_SYNTH_DIR = repo_root() / "assets" / "demo" / "synthetic"
_CHIPSET_DIR = repo_root() / "assets" / "demo" / "chipset"

app = typer.Typer(
    name="object-search",
    help="Exemplar-based object search: contracts, models, and synthetic data.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def _main(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="DEBUG logging.")] = False,
) -> None:
    """Configure logging once for the whole process, at the single CLI entry point."""
    setup_logging("DEBUG" if verbose else "INFO")


@app.command("fetch-models")
def fetch_models(
    force: Annotated[bool, typer.Option("--force", help="Re-download even if present.")] = False,
    only: Annotated[str | None, typer.Option("--only", help="Fetch one model by key.")] = None,
    list_only: Annotated[bool, typer.Option("--list", help="Print the registry and exit.")] = False,
) -> None:
    """Download or export every registered ONNX model into the gitignored ``models/`` dir."""
    registry = model_registry.MODEL_REGISTRY

    if list_only:
        typer.echo(f"{len(registry)} registered model(s):")
        for key in sorted(registry):
            spec = registry[key]
            typer.echo(
                f"  {key:14s} source={spec.source:15s} phase={spec.added_in_phase}  "
                f"licence={spec.license}"
            )
            typer.echo(f"{'':16s}{spec.source_note}")
        return

    if only is not None:
        if only not in registry:
            known = ", ".join(sorted(registry))
            typer.echo(f"unknown model {only!r}; known: {known}", err=True)
            raise typer.Exit(code=1)
        model_registry.fetch(registry[only], force=force)
        return

    model_registry.fetch_all(force=force)


@app.command("synth")
def synth(
    out: Annotated[Path, typer.Option("--out", help="Output directory.")] = _SYNTH_DIR,
    spec: Annotated[str | None, typer.Option("--spec", help="One DEMO_SPECS name, or all.")] = None,
) -> None:
    """Write the synthetic demo set (all ``DEMO_SPECS`` or one named spec) with ground truth."""
    from object_search.synthetic import DEMO_SPECS, save, synthesize

    if spec is not None and spec not in DEMO_SPECS:
        known = ", ".join(sorted(DEMO_SPECS))
        typer.echo(f"unknown spec {spec!r}; known: {known}", err=True)
        raise typer.Exit(code=1)

    names = [spec] if spec is not None else sorted(DEMO_SPECS)
    for name in names:
        image = synthesize(DEMO_SPECS[name])
        save(image, out / f"{name}.png")
    typer.echo(f"wrote {len(names)} synthetic image(s) to {out}")


@app.command("chipset")
def chipset(
    out: Annotated[Path, typer.Option("--out", help="Output directory.")] = _CHIPSET_DIR,
    force: Annotated[bool, typer.Option("--force", help="Overwrite existing files.")] = False,
) -> None:
    """Generate the ten-image chip-insertion benchmark set (EVAL-19) with exact ground truth."""
    from object_search.synthetic.chipset import write_chipset

    written = write_chipset(out, force=force)
    typer.echo(f"wrote {len(written)} chip image(s) to {out}")


_SAMPLES_DIR = repo_root() / "docs" / "samples"


@app.command("render-samples")
def render_samples(
    method: Annotated[
        str | None, typer.Option("--method", help="Render one registered method, or all.")
    ] = None,
    out: Annotated[Path, typer.Option("--out", help="Gallery output root.")] = _SAMPLES_DIR,
) -> None:
    """Render the committed sample gallery under ``docs/samples/<method>/`` (DOC-02).

    Iterates the method registry, so every registered method is rendered by default and a
    method added in a later phase is picked up here with no code change.
    """
    from object_search.samples import render_samples as _render
    from object_search.search import has_method

    if method is not None and not has_method(method):
        typer.echo(f"unknown method {method!r}; nothing rendered", err=True)
        raise typer.Exit(code=1)

    names = [method] if method is not None else None
    paths = _render(names, out_root=out)
    typer.echo(f"rendered {len(paths)} sample artifact(s) under {out}")


@app.command("benchmark")
def benchmark() -> None:
    """Placeholder: the benchmark runner arrives in Phase 8 (EVAL-04)."""
    logger.error("benchmark is not implemented until Phase 8 (EVAL-04)")
    typer.echo("benchmark: implemented in Phase 8 (EVAL-04). Nothing run.", err=True)
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
