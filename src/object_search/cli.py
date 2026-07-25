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
_MARKERS_DIR = repo_root() / "assets" / "demo" / "markers"

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


@app.command("markers")
def markers(
    out: Annotated[Path, typer.Option("--out", help="Output directory.")] = _MARKERS_DIR,
    spec: Annotated[
        str | None, typer.Option("--spec", help="One MARKER_DEMO_SPECS name, or all.")
    ] = None,
) -> None:
    """Write the synthetic marker demo set (Milestone 2) with exact per-marker ground truth.

    Each ``<name>.png`` gets a ``<name>.markers.json`` sidecar carrying every marker's exact
    tip, direction, centroid, box, and (when present) the pointed-at target box -- the oracles the
    orientation estimator is tested against. Deterministic from each spec's seed, so the images are
    committed and fully regenerable.
    """
    from object_search.synthetic.generator import (
        MARKER_DEMO_SPECS,
        save_marker_image,
        synthesize_markers,
    )

    if spec is not None and spec not in MARKER_DEMO_SPECS:
        known = ", ".join(sorted(MARKER_DEMO_SPECS))
        typer.echo(f"unknown marker spec {spec!r}; known: {known}", err=True)
        raise typer.Exit(code=1)

    names = [spec] if spec is not None else sorted(MARKER_DEMO_SPECS)
    for name in names:
        image = synthesize_markers(MARKER_DEMO_SPECS[name])
        save_marker_image(image, out / f"{name}.png")
    typer.echo(f"wrote {len(names)} marker image(s) to {out}")


@app.command("chipset")
def chipset(
    out: Annotated[Path, typer.Option("--out", help="Output directory.")] = _CHIPSET_DIR,
    force: Annotated[bool, typer.Option("--force", help="Overwrite existing files.")] = False,
) -> None:
    """Generate the ten-image chip-insertion benchmark set (EVAL-19) with exact ground truth."""
    from object_search.synthetic.chipset import write_chipset

    written = write_chipset(out, force=force)
    typer.echo(f"wrote {len(written)} chip image(s) to {out}")


_TEXTURED_DIR = repo_root() / "assets" / "demo" / "textured"


@app.command("textured")
def textured(
    out: Annotated[Path, typer.Option("--out", help="Output directory.")] = _TEXTURED_DIR,
    force: Annotated[bool, typer.Option("--force", help="Overwrite existing files.")] = False,
) -> None:
    """Generate the textured benchmark regimes (EVAL-20) with exact ground truth.

    Three regimes -- plain (keypoint-favourable), varied (scale/rotation/brightness), and
    cluttered (noise + distractors) -- complementing the NCC-favourable chipset.
    """
    from object_search.synthetic.textured import write_textured

    written = write_textured(out, force=force)
    typer.echo(f"wrote {len(written)} textured image(s) to {out}")


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
    typer.echo(f"rendered {len(paths)} method sample artifact(s) under {out}")

    # The exploration analogue of the method loop: render the marker-conditioned gallery too.
    # The proposal stage needs the FastSAM weight; pin the CPU provider so the committed panels
    # are reproducible across machines, and skip gracefully (like other model-gated paths) when
    # the weight is absent, so `pixi run samples` still works with no models fetched.
    if method is None:
        from object_search.samples import render_marker_samples
        from object_search.search.proposals import default_backend

        try:
            backend = default_backend(providers=["CPUExecutionProvider"])
        except FileNotFoundError:
            logger.info("FastSAM weight absent; skipping the marker exploration gallery")
        else:
            marker_paths = render_marker_samples(backend=backend, out_root=out)
            typer.echo(f"rendered {len(marker_paths)} marker sample artifact(s) under {out}")


@app.command("benchmark")
def benchmark() -> None:
    """Redirect: the benchmark is a Hydra entry point, not a Typer subcommand (EVAL-04).

    ``@hydra.main`` takes over ``sys.argv`` and cannot compose with Typer's parser, so the
    benchmark lives at ``object_search.eval.benchmark`` and is invoked via ``pixi run bench``
    (full sweep) or ``pixi run bench-ci`` (model-free chipset subset). This shim exits
    non-zero so a stale ``object-search benchmark`` invocation fails loudly with the right
    command rather than silently doing nothing.
    """
    logger.error("benchmark moved to object_search.eval.benchmark (Hydra owns sys.argv)")
    typer.echo(
        "benchmark is a Hydra entry point, not a Typer subcommand. "
        "Run `pixi run bench` (full) or `pixi run bench-ci` (model-free chipset subset).",
        err=True,
    )
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
