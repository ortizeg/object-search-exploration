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

from object_search.eval import datasets as dataset_registry
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


@app.command("fetch-datasets")
def fetch_datasets(
    force: Annotated[bool, typer.Option("--force", help="Reconvert even if present.")] = False,
    only: Annotated[str | None, typer.Option("--only", help="Fetch one dataset by key.")] = None,
    list_only: Annotated[
        bool, typer.Option("--list", help="Print the dataset registry and exit.")
    ] = False,
) -> None:
    """Convert every registered research dataset into the gitignored ``datasets/`` tree (EVAL-21).

    Licence-gated datasets (CARPK) are ``requires_manual``: a human accepts the licence and drops
    the raw archive at the printed ``datasets/_incoming/<key>/`` path first. Raw bytes never enter
    git; ``datasets/provenance.json`` records each file's SHA-256 + source URL + licence (D-08).
    """
    registry = dataset_registry.DATASET_REGISTRY

    if list_only:
        typer.echo(f"{len(registry)} registered dataset(s):")
        for key in sorted(registry):
            spec = registry[key]
            drop = repo_root() / "datasets" / "_incoming" / spec.incoming_subdir
            typer.echo(
                f"  {key:10s} source={spec.source:16s} phase={spec.added_in_phase}  "
                f"licence={spec.license}"
            )
            typer.echo(f"{'':12s}manual={spec.requires_manual}  drop archive at: {drop}")
        return

    if only is not None:
        if only not in registry:
            known = ", ".join(sorted(registry))
            typer.echo(f"unknown dataset {only!r}; known: {known}", err=True)
            raise typer.Exit(code=1)
        dataset_registry.fetch(registry[only], force=force)
        return

    dataset_registry.fetch_all(force=force)


def _parse_methods(raw: str | None) -> tuple[str, ...]:
    """Parse ``--methods a,b`` into a validated key tuple; ``None`` -> the full default set.

    Every key is validated through :func:`object_search.search.get_method` **at the CLI boundary**,
    so a typo exits non-zero here rather than producing a tuning report with an empty ``methods``
    list that looks like a successful run of nothing.
    """
    from object_search.eval.tuning import DEFAULT_TUNING_METHODS
    from object_search.search import UnknownMethodError, get_method

    if raw is None:
        return DEFAULT_TUNING_METHODS
    keys = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not keys:
        typer.echo("--methods was empty; give at least one registry key", err=True)
        raise typer.Exit(code=1)
    for key in keys:
        try:
            get_method(key)
        except UnknownMethodError as exc:
            typer.echo(str(exc.args[0]), err=True)
            raise typer.Exit(code=1) from exc
    return keys


@app.command("tune-floorplans")
def tune_floorplans(
    dataset: Annotated[
        str, typer.Option("--dataset", help="floorplans-door | floorplans-window | both.")
    ] = "both",
    research_root: Annotated[
        Path, typer.Option("--research-root", help="Base dir of converted datasets.")
    ] = Path("datasets"),
    exemplars: Annotated[int, typer.Option("--exemplars", help="Exemplars per query.")] = 1,
    methods: Annotated[
        str | None,
        typer.Option(
            "--methods",
            help="Comma-separated registry keys to tune (default: all six).",
        ),
    ] = None,
) -> None:
    """Tune each method's acceptance threshold on the floor-plan val split, freeze, report on test.

    Selects the argmax-F1 @ IoU 0.5 config per method on ``val``, freezes it, and scores both the
    frozen and the default config on ``test`` -- the tuned-vs-default table. Requires
    ``fetch-datasets`` to have converted the floor-plan tree first.

    ``--methods`` narrows the run set to a subset (e.g. ``--methods sparse-geo``). Tuning all six
    on CPU is prohibitive -- OWLv2/DINOv2/FastSAM dominate the wall-clock -- so a single-method
    sweep is what makes an iterate/measure loop on one method practical. Omitting the option keeps
    the full six-method default, so the committed report shape is unchanged.
    """
    from object_search.eval.tuning import run_domain_tuning

    keys = ("floorplans-door", "floorplans-window") if dataset == "both" else (dataset,)
    selected = _parse_methods(methods)
    for key in keys:
        out = f"docs/benchmark/{key}-tuning-results.json"
        report = run_domain_tuning(
            key, research_root, methods=selected, exemplar_count=exemplars, out=out
        )
        typer.echo(f"{key}: tuned {len(report['methods'])} method(s) -> {out}")
        for entry in report["methods"]:
            typer.echo(
                f"  {entry['method']:16s} overrides={entry['tuned_overrides']}  "
                f"testF1 tuned={entry['tuned_test'].get('f1')} "
                f"default={entry['default_test'].get('f1')}"
            )


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


_REAL_OBJECTS_DIR = repo_root() / "assets" / "demo" / "real-objects"
_REAL_RAW_DIR = repo_root() / "assets" / "demo" / "real-objects" / "_raw"
_REAL_CUTOUTS_DIR = repo_root() / "assets" / "demo" / "real-objects" / "cutouts"


@app.command("fetch-real-photos")
def fetch_real_photos_cmd(
    raw_dir: Annotated[Path, typer.Option("--raw-dir", help="Raw download directory.")] = (
        _REAL_RAW_DIR
    ),
    force: Annotated[bool, typer.Option("--force", help="Re-download even if present.")] = False,
) -> None:
    """Download the real-object-insertion set's source photos from Wikimedia Commons.

    Gitignored raw material for ``real-objects``; every file's title/author/licence/source URL
    and SHA-256 is recorded in ``<raw-dir>/provenance.json``. A missing/renamed Commons file logs
    a warning and is skipped rather than aborting the whole fetch.
    """
    from object_search.synthetic.real_insertion import fetch_real_photos

    written = fetch_real_photos(raw_dir, force=force)
    typer.echo(f"wrote {len(written)} raw photo(s) to {raw_dir}")


@app.command("real-objects")
def real_objects(
    out: Annotated[Path, typer.Option("--out", help="Output directory.")] = _REAL_OBJECTS_DIR,
    raw_dir: Annotated[Path, typer.Option("--raw-dir", help="Raw photo directory.")] = (
        _REAL_RAW_DIR
    ),
    cutouts_dir: Annotated[Path, typer.Option("--cutouts-dir", help="Cutout cache dir.")] = (
        _REAL_CUTOUTS_DIR
    ),
    force: Annotated[bool, typer.Option("--force", help="Overwrite existing files.")] = False,
) -> None:
    """Generate the real-object-insertion benchmark set with exact ground truth.

    Segments each raw object photo with FastSAM (``pixi run fetch-real-photos`` must have
    populated ``raw-dir`` first, and ``pixi run -e export fetch-models --only fastsam-s`` must
    have exported the weight), caches the cutout, then pastes it onto its background(s) at
    known, non-overlapping positions. Images/cutouts with a missing input are logged and skipped.
    """
    from object_search.synthetic.real_insertion import write_real_insertion

    written = write_real_insertion(out, raw_dir, cutouts_dir, force=force)
    typer.echo(f"wrote {len(written)} real-object image(s) to {out}")


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
