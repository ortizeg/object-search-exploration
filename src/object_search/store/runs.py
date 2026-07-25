"""Persist and reconstruct a :class:`RunRecord`, including its sub-threshold candidates.

A run is stored across three tables in one transaction: the wide ``runs`` row, the
``matches`` it claimed, and the ``candidates`` it observed below threshold (EVAL-08) so
an offline threshold sweep can rebuild a full PR curve later without re-running anything.

Two things this module must never do:

* **Coerce a ``None`` slice field to ``0``.** ``slice_metadata`` fields are nullable and
  ``None`` means "unknown", a different claim from ``0`` (EVAL-10). They round-trip as
  ``None``.
* **Lose an error.** ``outcome='error'`` and ``outcome='empty'`` are both persisted with
  zero matches -- an error is evidence too (EVAL-12), not a row to drop.

The one bulky field is ``diagnostics_json``: a similarity-heatmap PNG dwarfs everything
else. It is size-capped (:data:`_DIAGNOSTICS_MAX_BYTES`) -- above the cap the heatmap is
dropped and the rest kept, so a fat diagnostic can never make a run unstorable.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime

from loguru import logger

from object_search.schemas.geometry import BBox, ExemplarBox
from object_search.schemas.records import Provenance, RunRecord, SliceMetadata
from object_search.schemas.search import (
    Candidate,
    Diagnostics,
    LatencyBreakdown,
    Match,
    MethodError,
    SearchOutcome,
    SearchResult,
)

_DIAGNOSTICS_MAX_BYTES = 256 * 1024


def _serialize_diagnostics(diagnostics: Diagnostics) -> str:
    """JSON for the diagnostics payload, dropping the heatmap if it blows the size cap."""
    payload = diagnostics.model_dump(mode="json")
    text = json.dumps(payload, sort_keys=True)
    if len(text.encode("utf-8")) <= _DIAGNOSTICS_MAX_BYTES:
        return text
    if payload.get("similarity_heatmap") is not None:
        payload["similarity_heatmap"] = None
        logger.warning(
            "diagnostics exceeded the {} KB cap; dropped similarity_heatmap to keep the "
            "run storable (EVAL-08 candidates and all other fields are retained)",
            _DIAGNOSTICS_MAX_BYTES // 1024,
        )
    return json.dumps(payload, sort_keys=True)


def insert_run(conn: sqlite3.Connection, run: RunRecord) -> int:
    """Write a run, its matches and its candidates in one transaction; return the new id.

    ``config_json`` is stored verbatim from the record -- it is already the canonical,
    sorted-key JSON that ``config_hash`` was computed over, so a hash mismatch stays
    diffable. Latency is three columns, provenance and slice metadata are their own
    columns (``None`` preserved), and diagnostics are size-capped JSON.
    """
    result = run.result
    prov = run.provenance
    box = run.exemplar.box
    retrieved = len(result.matches)
    error = result.error

    with conn:
        # exploration is now passed through explicitly (Milestone 2): RunRecord carries it and
        # defaults it to 'same-image-search', so the default path stores exactly the value the
        # column DEFAULT would have, and a marker-conditioned run stores 'marker-conditioned'.
        cursor = conn.execute(
            """
            INSERT INTO runs (
                exploration, image_id, method, method_version,
                exemplar_x, exemplar_y, exemplar_w, exemplar_h, exemplar_label,
                config_json, config_hash, outcome, error_kind, error_message,
                retrieved, threshold_applied,
                preprocess_ms, inference_ms, postprocess_ms,
                git_sha, python_version, numpy_version, cv2_version,
                onnxruntime_version, ort_providers, model_hashes_json,
                pixi_lock_sha256, provenance_created_at,
                slice_true_instance_count, slice_scale_min, slice_scale_max,
                slice_rotation_min, slice_rotation_max, slice_clutter,
                slice_exemplar_keypoint_count, diagnostics_json, created_at
            ) VALUES (
                ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?,
                ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?,
                ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?
            )
            """,
            (
                run.exploration,
                run.image_id,
                run.method,
                result.method_version,
                box.x,
                box.y,
                box.w,
                box.h,
                run.exemplar.label,
                run.config_json,
                run.config_hash,
                result.outcome.value,
                error.kind if error is not None else None,
                error.message if error is not None else None,
                retrieved,
                result.threshold_applied,
                result.latency.preprocess_ms,
                result.latency.inference_ms,
                result.latency.postprocess_ms,
                prov.git_sha,
                prov.python_version,
                prov.numpy_version,
                prov.cv2_version,
                prov.onnxruntime_version,
                prov.ort_providers,
                json.dumps(dict(prov.model_hashes), sort_keys=True),
                prov.pixi_lock_sha256,
                prov.created_at.isoformat(),
                run.slice_metadata.true_instance_count,
                run.slice_metadata.instance_scale_min,
                run.slice_metadata.instance_scale_max,
                run.slice_metadata.rotation_min_deg,
                run.slice_metadata.rotation_max_deg,
                run.slice_metadata.clutter_level,
                run.slice_metadata.exemplar_keypoint_count,
                _serialize_diagnostics(result.diagnostics),
                prov.created_at.isoformat(),
            ),
        )
        run_id = int(cursor.lastrowid or 0)

        conn.executemany(
            """
            INSERT INTO matches (run_id, idx, x, y, w, h, score, is_exemplar, transform_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    idx,
                    match.box.x,
                    match.box.y,
                    match.box.w,
                    match.box.h,
                    match.score,
                    int(match.is_exemplar),
                    None if match.transform is None else json.dumps(list(match.transform)),
                )
                for idx, match in enumerate(result.matches)
            ],
        )

        conn.executemany(
            """
            INSERT INTO candidates (run_id, rank, x, y, w, h, score)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    rank,
                    candidate.box.x,
                    candidate.box.y,
                    candidate.box.w,
                    candidate.box.h,
                    candidate.score,
                )
                for rank, candidate in enumerate(result.candidates)
            ],
        )

    logger.debug(
        "stored run {} ({}): {} match(es), {} candidate(s)",
        run_id,
        result.outcome.value,
        retrieved,
        len(result.candidates),
    )
    return run_id


def _load_matches(conn: sqlite3.Connection, run_id: int) -> tuple[Match, ...]:
    rows = conn.execute(
        "SELECT x, y, w, h, score, is_exemplar, transform_json "
        "FROM matches WHERE run_id = ? ORDER BY idx",
        (run_id,),
    ).fetchall()
    return tuple(
        Match(
            box=BBox(x=row["x"], y=row["y"], w=row["w"], h=row["h"]),
            score=row["score"],
            is_exemplar=bool(row["is_exemplar"]),
            transform=(
                None if row["transform_json"] is None else tuple(json.loads(row["transform_json"]))
            ),
        )
        for row in rows
    )


def _load_candidates(conn: sqlite3.Connection, run_id: int) -> tuple[Candidate, ...]:
    rows = conn.execute(
        "SELECT x, y, w, h, score FROM candidates WHERE run_id = ? ORDER BY rank",
        (run_id,),
    ).fetchall()
    return tuple(
        Candidate(
            box=BBox(x=row["x"], y=row["y"], w=row["w"], h=row["h"]),
            score=row["score"],
        )
        for row in rows
    )


def get_run(conn: sqlite3.Connection, run_id: int) -> RunRecord:
    """Reconstruct the full :class:`RunRecord`, matches and candidates included.

    Raises:
        KeyError: If no run with ``run_id`` exists.
    """
    row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    if row is None:
        raise KeyError(f"no run with id {run_id}")

    outcome = SearchOutcome(row["outcome"])
    error = (
        MethodError(kind=row["error_kind"], message=row["error_message"] or "")
        if outcome is SearchOutcome.ERROR
        else None
    )
    diagnostics = Diagnostics.model_validate(json.loads(row["diagnostics_json"]))

    result = SearchResult(
        method=row["method"],
        method_version=row["method_version"],
        outcome=outcome,
        matches=_load_matches(conn, run_id),
        latency=LatencyBreakdown(
            preprocess_ms=row["preprocess_ms"],
            inference_ms=row["inference_ms"],
            postprocess_ms=row["postprocess_ms"],
        ),
        threshold_applied=row["threshold_applied"],
        candidates=_load_candidates(conn, run_id),
        diagnostics=diagnostics,
        error=error,
    )

    provenance = Provenance(
        git_sha=row["git_sha"],
        method_version=row["method_version"],
        config_hash=row["config_hash"],
        model_hashes=json.loads(row["model_hashes_json"]),
        python_version=row["python_version"],
        numpy_version=row["numpy_version"],
        cv2_version=row["cv2_version"],
        onnxruntime_version=row["onnxruntime_version"],
        ort_providers=row["ort_providers"],
        pixi_lock_sha256=row["pixi_lock_sha256"],
        created_at=datetime.fromisoformat(row["provenance_created_at"]),
    )

    return RunRecord(
        id=run_id,
        image_id=row["image_id"],
        exemplar=ExemplarBox(
            box=BBox(
                x=row["exemplar_x"],
                y=row["exemplar_y"],
                w=row["exemplar_w"],
                h=row["exemplar_h"],
            ),
            label=row["exemplar_label"],
        ),
        method=row["method"],
        exploration=row["exploration"],
        config_json=row["config_json"],
        config_hash=row["config_hash"],
        result=result,
        slice_metadata=SliceMetadata(
            true_instance_count=row["slice_true_instance_count"],
            instance_scale_min=row["slice_scale_min"],
            instance_scale_max=row["slice_scale_max"],
            rotation_min_deg=row["slice_rotation_min"],
            rotation_max_deg=row["slice_rotation_max"],
            clutter_level=row["slice_clutter"],
            exemplar_keypoint_count=row["slice_exemplar_keypoint_count"],
        ),
        provenance=provenance,
    )
