"""Real-object insertion benchmark set: real photos, real objects, exact ground truth.

Every other benchmark set in :mod:`object_search.synthetic` is drawn or rendered by this repo.
That is honest but incomplete -- it says nothing about real photographic texture, lighting, or
backgrounds. This module closes that gap **without hand-labelling**: real object photos are
segmented with FastSAM (already Trial-approved, `docs/library-reviews/fastsam.md`) into clean RGBA
cutouts, which are then pasted onto real background photos at known, non-overlapping positions --
so ground truth stays exact **by construction**, exactly like `chipset.py` / `textured.py`, just
with real pixels on both sides of the paste.

Three properties are load-bearing here, same discipline as the other synthetic sets:

1. **Every source photo is individually licence-tracked.** :data:`REAL_OBJECT_MANIFEST` /
   :data:`REAL_BACKGROUND_MANIFEST` record title/author/licence/source-URL per Wikimedia Commons
   file -- this repo's `assets/demo/LICENSES.md` requires per-file provenance, not a blanket
   dataset licence (see the "Generic repeated-instance photos" TODO it closes).
2. **Cutout extraction never guesses silently.** FastSAM's automatic ("everything") mode has no
   notion of "the" subject; :func:`extract_cutout` picks the proposal most likely to be the
   photographed object (large AND centred) and returns ``None`` -- logged, never raised -- when
   nothing usable is found. A bad cutout is skipped, not shipped.
3. **The recorded count is the ACHIEVED count, never the requested N**, and the ground-truth box
   is the AABB of the *actually pasted* (scaled, rotated) alpha mask -- not the cutout's nominal
   size -- exactly the two invariants `chipset.py`/`textured.py` already enforce.

Two artifacts are deliberately gitignored (regenerable, and depend on network + the gitignored
FastSAM weight): the raw Commons downloads and the cached per-object cutouts. Only the final
composited JPEGs + ``.gt.json`` sidecars are committed -- same rule as chipset/textured: small,
exact, and the eval harness must work with no weights and no network.

ROBUSTNESS BACKLOG
-------------------
- Flat alpha-paste has no shadow/lighting harmonization, so composites read as "cut-and-paste"
  rather than a genuinely photographed scene. Acceptable for this set's purpose (real *texture*,
  not photorealistic compositing); revisit with Poisson blending if that gap matters later.
- :func:`_select_object_proposal`'s area-times-centrality heuristic can pick the wrong region on a
  cluttered or low-contrast source photo. Mitigated by curating source photos where the object is
  visually dominant and by manually spot-checking every generated cutout before committing, but a
  wrong pick would silently ship a bad "object" (e.g. a shadow or the wrong item) -- there is no
  automatic sanity check on cutout content today.
- Background photos are resized to a single working resolution (:data:`_WORKING_LONG_EDGE`) with
  no attempt to preserve their native scale relative to the pasted objects, so absolute object size
  is not physically meaningful across images (fine for this harness -- it scores boxes, not
  real-world scale).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NamedTuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import cv2
import numpy as np
import numpy.typing as npt
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from object_search.inference import FastSAMConfig, Proposal
from object_search.schemas.geometry import BBox
from object_search.schemas.records import SliceMetadata
from object_search.search.proposals import ProposalBackend, default_backend, propose
from object_search.synthetic.generator import SyntheticImage

# Master seed for the whole set: every per-image seed derives from this, so REAL_INSERTION_SPECS
# is fixed and regenerates identically (mirrors chipset.py / textured.py).
_MASTER_SEED = 20260806
_N_INSTANCE_CHOICES = (4, 5, 6, 7, 8)

Regime = Literal["plain", "varied", "cluttered"]

# Per-regime jitter knobs, mirroring textured.py's _regime_knobs so the same orientation/scale
# exploration exists on real pixels: plain is the fixed-pose NCC-favourable baseline, varied adds
# scale + rotation jitter, cluttered adds a pasted-but-unrecorded distractor on top of varied.
# scale is relative to the cutout's already-capped _CUTOUT_LONG_EDGE (220px), itself pasted onto
# a _WORKING_LONG_EDGE (1024px) canvas -- so scale_min=0.25 / scale_max=1.6 spans roughly 55px to
# 350px long edge, ~5% to ~34% of the canvas. `varied`/`cluttered` deliberately span large-to-small
# in one range (not a separate size regime) so every image already exercises scale-invariance, not
# just orientation -- a size-only ramp would need yet another regime dimension for a gain this
# range already gives for free.
_REGIME_KNOBS: Mapping[Regime, Mapping[str, float | int]] = {
    "plain": {"scale_min": 1.0, "scale_max": 1.0, "rotation_deg": 0.0, "n_distractors": 0},
    "varied": {"scale_min": 0.25, "scale_max": 1.6, "rotation_deg": 30.0, "n_distractors": 0},
    "cluttered": {"scale_min": 0.3, "scale_max": 1.5, "rotation_deg": 20.0, "n_distractors": 2},
}
_PLACEMENT_MAX_ATTEMPTS = 3000
_INSET = 3  # keep every pasted instance fully inside the frame.
_PLACEMENT_GAP = 3  # a small guaranteed gap so no two instances even touch.
_WORKING_LONG_EDGE = 1024  # background canvases are downscaled to this long edge before pasting.
# A raw Commons object photo can be several thousand pixels on a side even after cropping to the
# mask -- pasted at scale ~1.0 that would already exceed the 1024px background canvas, so EVERY
# placement attempt would fail the "does it fit" check and burn the full attempt cap on expensive
# warpAffine calls over a multi-megapixel image (measured: this hung real-objects generation).
# Capping the cutout's long edge here, once, keeps every downstream resize/rotate/paste cheap and
# keeps pasted objects a plausible fraction of the scene (~20% of the working canvas at scale 1.0,
# the same "modest object on a real background" proportion chipset/textured use synthetically).
_CUTOUT_LONG_EDGE = 220
# Composited output is JPEG, not PNG (unlike chipset/textured's flat/gradient synthetic canvases):
# real photographic backgrounds carry genuine high-frequency detail that barely compresses
# losslessly, so PNG output routinely exceeded the repo's 2 MB pre-commit large-file gate.
# Quality 92 mirrors the basketball frames' documented committed-real-photo convention
# (`assets/demo/LICENSES.md`) and keeps every composite well under the limit.
_COMPOSITE_JPEG_QUALITY = 92
_ALPHA_THRESHOLD = 0.5  # FastSAM's soft mask is thresholded to a hard alpha at this cut.
# A real photo's edge is never perfectly sharp, so a hard threshold at _ALPHA_THRESHOLD routinely
# leaves a 1-2px fringe of the ORIGINAL background colour clinging to the object (measured: every
# ball cutout showed a thin grey halo from its source photo's backdrop). Eroding the binary mask by
# one 3x3 pass trims exactly that fringe; it costs a pixel or two of the object's true edge, which
# is a far smaller error than pasting a visible ring of the wrong background.
_MASK_ERODE_KERNEL = np.ones((3, 3), dtype=np.uint8)
# A clean single-object mask fills a healthy fraction of its own bounding box; a fragmented/wrong
# one (measured failure mode: a near-white object on a near-white background) does not.
_MIN_CUTOUT_SOLIDITY = 0.20
# The inverse failure, also measured: a product photo whose object nearly fills the frame against a
# plain backdrop gives FastSAM no strong internal edge, so its "proposal" is close to the WHOLE
# photo (backdrop included) rather than a tight object silhouette -- solidity alone does not catch
# this, because a whole-frame mask is itself perfectly solid. A mask covering more of the frame
# than this is rejected even though it passed the solidity floor.
_MAX_MASK_COVERAGE = 0.85
# Neither failure has one universal fix -- fragmentation wants a MORE permissive threshold (recovers
# small suppressed proposals), whole-frame wants a MORE selective one (a higher-confidence proposal
# tends to be tighter). extract_cutout tries this ladder in order and keeps the first attempt that
# clears both gates, rather than a single fallback in one direction.
_CONF_THRES_LADDER = (0.4, 0.65, 0.15)  # FastSAM's own default (0.4) tried first.
_USER_AGENT = (
    "object-search-exploration/1.0 "
    "(local research dataset build; https://github.com/ortizeg/object-search-exploration)"
)


class PhotoProvenance(BaseModel):
    """One Wikimedia Commons source photo: where it came from and under what licence.

    ``category`` is the manifest key used everywhere downstream (raw filename stem, cutout cache
    filename, spec ``target``/``background`` field, exemplar identity) -- a single kebab-case slug,
    e.g. ``"tennis-ball"``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    category: str = Field(min_length=1)
    title: str
    file_url: str
    page_url: str
    author: str
    license: str


# Source manifests -- the single source of truth for what gets fetched and composited, mirroring
# CHIPSET_SPECS / TEXTURED_SPECS. Every entry's licence/author was pulled from the Wikimedia
# Commons API's own `imageinfo.extmetadata` (not assumed from a search snippet), and every
# `file_url` was confirmed to resolve with HTTP 200 / an image content-type before being recorded
# here -- the same "recorded, not re-hosted" provenance discipline as the research datasets
# (`eval/datasets.py`) and the basketball frames' "Exact source paths" table.
REAL_OBJECT_MANIFEST: tuple[PhotoProvenance, ...] = (
    PhotoProvenance(
        category="tennis-ball",
        title="Tennis ball 01.jpg",
        file_url="https://upload.wikimedia.org/wikipedia/commons/1/1f/Tennis_ball_01.jpg",
        page_url="https://commons.wikimedia.org/wiki/File:Tennis_ball_01.jpg",
        author="Fcb981 (English Wikipedia)",
        license="CC-BY-2.5",
    ),
    PhotoProvenance(
        category="claw-hammer",
        title="Stanley graphite claw hammer.jpg",
        file_url=(
            "https://upload.wikimedia.org/wikipedia/commons/8/8c/Stanley_graphite_claw_hammer.jpg"
        ),
        page_url="https://commons.wikimedia.org/wiki/File:Stanley_graphite_claw_hammer.jpg",
        author="J.C. Fields (Typhoon)",
        license="CC-BY-SA-3.0",
    ),
    PhotoProvenance(
        category="screwdriver",
        title="Big flat screwdriver.jpg",
        file_url="https://upload.wikimedia.org/wikipedia/commons/5/5f/Big_flat_screwdriver.jpg",
        page_url="https://commons.wikimedia.org/wiki/File:Big_flat_screwdriver.jpg",
        author="Jiří Sedláček (Frettie)",
        license="CC-BY-SA-4.0",
    ),
    PhotoProvenance(
        category="c-clamp",
        title="Carondelet Foundry Company C-Clamp.jpg",
        file_url=(
            "https://upload.wikimedia.org/wikipedia/commons/a/a4/"
            "Carondelet_Foundry_Company_C-Clamp.jpg"
        ),
        page_url=("https://commons.wikimedia.org/wiki/File:Carondelet_Foundry_Company_C-Clamp.jpg"),
        author="Carondelet Foundry Company",
        license="Public Domain",
    ),
    PhotoProvenance(
        category="apple",
        title="Apple (1).jpg",
        file_url="https://upload.wikimedia.org/wikipedia/commons/b/b1/Apple_%281%29.jpg",
        page_url="https://commons.wikimedia.org/wiki/File:Apple_(1).jpg",
        author="Renee Comet (Photographer)",
        license="Public Domain",
    ),
    PhotoProvenance(
        category="orange",
        title="Orange Fruit Close-up.jpg",
        file_url=("https://upload.wikimedia.org/wikipedia/commons/b/bf/Orange_Fruit_Close-up.jpg"),
        page_url="https://commons.wikimedia.org/wiki/File:Orange_Fruit_Close-up.jpg",
        author="freestock.ca",
        license="CC-BY-SA-3.0",
    ),
    PhotoProvenance(
        category="hockey-puck",
        title="Ice-hockey puck 2.JPG",
        file_url="https://upload.wikimedia.org/wikipedia/commons/3/3a/Ice-hockey_puck_2.JPG",
        page_url="https://commons.wikimedia.org/wiki/File:Ice-hockey_puck_2.JPG",
        author="Santeri Viinamäki",
        license="CC-BY-4.0",
    ),
    PhotoProvenance(
        category="chess-pawn",
        title="Chess piece - White pawn.JPG",
        file_url=(
            "https://upload.wikimedia.org/wikipedia/commons/e/ed/Chess_piece_-_White_pawn.JPG"
        ),
        page_url="https://commons.wikimedia.org/wiki/File:Chess_piece_-_White_pawn.JPG",
        author="MichaelMaggs",
        license="CC-BY-SA-2.5",
    ),
    PhotoProvenance(
        # The deliberate stress object: textureless, rotationally symmetric (see the module
        # docstring / DATASETS.md) -- expected to trip ncc's low-variance guard and sparse-geo's
        # SIFT-keypoint floor on a REAL photo, the case none of the synthetic sets covers.
        category="ping-pong-ball",
        title="Table Tennis Plastic Ball 40+ mm.jpg",
        file_url=(
            "https://upload.wikimedia.org/wikipedia/commons/b/b3/"
            "Table_Tennis_Plastic_Ball_40%2B_mm.jpg"
        ),
        page_url=("https://commons.wikimedia.org/wiki/File:Table_Tennis_Plastic_Ball_40%2B_mm.jpg"),
        author="Peter Porai-Koshits",
        license="CC-BY-SA-4.0",
    ),
    PhotoProvenance(
        category="golf-ball",
        title="Golf-ball.jpg",
        file_url="https://upload.wikimedia.org/wikipedia/commons/f/f5/Golf-ball.jpg",
        page_url="https://commons.wikimedia.org/wiki/File:Golf-ball.jpg",
        author="Paolo Neo",
        license="CC-BY-SA-3.0",
    ),
)

REAL_BACKGROUND_MANIFEST: tuple[PhotoProvenance, ...] = (
    PhotoProvenance(
        category="wood-floor",
        title="SibleySquareBareFloorboards.jpg",
        file_url=(
            "https://upload.wikimedia.org/wikipedia/commons/0/0a/SibleySquareBareFloorboards.jpg"
        ),
        page_url="https://commons.wikimedia.org/wiki/File:SibleySquareBareFloorboards.jpg",
        author="DanielPenfield",
        license="CC-BY-SA-4.0",
    ),
    PhotoProvenance(
        category="concrete",
        title=(
            "Grey moderately dirty worn grubby dusty poured concrete seamless floor texture.jpg"
        ),
        file_url=(
            "https://upload.wikimedia.org/wikipedia/commons/a/a2/Grey_moderately_dirty_worn_"
            "grubby_dusty_poured_concrete_seamless_floor_texture.jpg"
        ),
        page_url=(
            "https://commons.wikimedia.org/wiki/File:Grey_moderately_dirty_worn_grubby_dusty_"
            "poured_concrete_seamless_floor_texture.jpg"
        ),
        author="Sisters.seamless",
        license="CC0-1.0",
    ),
    PhotoProvenance(
        category="grass",
        title="084 Green grass lawn background, green mowed grass free photo.jpg",
        file_url=(
            "https://upload.wikimedia.org/wikipedia/commons/b/b6/084_Green_grass_lawn_"
            "background%2C_green_mowed_grass_free_photo.jpg"
        ),
        page_url=(
            "https://commons.wikimedia.org/wiki/File:084_Green_grass_lawn_background,"
            "_green_mowed_grass_free_photo.jpg"
        ),
        author="Marek Ślusarczyk (Tupungato)",
        license="CC-BY-3.0",
    ),
    PhotoProvenance(
        category="gravel",
        title="Gravel Stones.jpg",
        file_url="https://upload.wikimedia.org/wikipedia/commons/d/d1/Gravel_Stones.jpg",
        page_url="https://commons.wikimedia.org/wiki/File:Gravel_Stones.jpg",
        author="Saral Shots",
        license="CC0-1.0",
    ),
    PhotoProvenance(
        category="brick-wall",
        title="Red-brick-wall-texture-clean.jpg",
        file_url=(
            "https://upload.wikimedia.org/wikipedia/commons/a/a8/Red-brick-wall-texture-clean.jpg"
        ),
        page_url="https://commons.wikimedia.org/wiki/File:Red-brick-wall-texture-clean.jpg",
        author="MartinThoma",
        license="CC0-1.0",
    ),
    PhotoProvenance(
        category="sand",
        title="Gfp-grainy-sand-texture.jpg",
        file_url=(
            "https://upload.wikimedia.org/wikipedia/commons/0/0a/Gfp-grainy-sand-texture.jpg"
        ),
        page_url="https://commons.wikimedia.org/wiki/File:Gfp-grainy-sand-texture.jpg",
        author="Yinan Chen (goodfreephotos.com)",
        license="Public Domain",
    ),
    PhotoProvenance(
        # Substitution (see the sourcing note): a macro/close-up of plain berber-weave carpet --
        # every wide-angle carpet photo found under an allowed licence was ornately patterned or a
        # full-room shot with furniture/clutter.
        category="carpet",
        title="Berber Carpet (macro).jpg",
        file_url=(
            "https://upload.wikimedia.org/wikipedia/commons/8/83/Berber_Carpet_%28macro%29.jpg"
        ),
        page_url="https://commons.wikimedia.org/wiki/File:Berber_Carpet_(macro).jpg",
        author="Pi_Guy_31415",
        license="CC-BY-2.5",
    ),
    PhotoProvenance(
        category="granite",
        title=(
            "Blue Pearl Granite (larvikite) (Larvik Batholith, 292-298 Ma, Early Permian; "
            "near Larvik, Norway) 2.jpg"
        ),
        file_url=(
            "https://upload.wikimedia.org/wikipedia/commons/9/9f/Blue_Pearl_Granite_%28larvikite"
            "%29_%28Larvik_Batholith%2C_292-298_Ma%2C_Early_Permian%3B_near_Larvik%2C_Norway%29_"
            "2.jpg"
        ),
        page_url=(
            "https://commons.wikimedia.org/wiki/File:Blue_Pearl_Granite_(larvikite)_(Larvik_"
            "Batholith,_292-298_Ma,_Early_Permian;_near_Larvik,_Norway)_2.jpg"
        ),
        author="James St. John",
        license="CC-BY-2.0",
    ),
    PhotoProvenance(
        # Substitution (see the sourcing note): the dominant subject is smooth paved asphalt, but
        # the frame borders grass/gravel/trees -- no edge-to-edge asphalt photo under an allowed
        # licence was found. Consider cropping to the asphalt region before pasting.
        category="asphalt",
        title="New pedestrian way in Otaniemi.jpg",
        file_url=(
            "https://upload.wikimedia.org/wikipedia/commons/a/ae/New_pedestrian_way_in_Otaniemi.jpg"
        ),
        page_url="https://commons.wikimedia.org/wiki/File:New_pedestrian_way_in_Otaniemi.jpg",
        author="JIP",
        license="CC-BY-SA-4.0",
    ),
    PhotoProvenance(
        category="cardboard",
        title="Corrugated Cardboard.JPG",
        file_url=("https://upload.wikimedia.org/wikipedia/commons/b/b6/Corrugated_Cardboard.JPG"),
        page_url="https://commons.wikimedia.org/wiki/File:Corrugated_Cardboard.JPG",
        author="Richard Wheeler (Zephyris)",
        license="CC-BY-SA-3.0",
    ),
)

# The busier backgrounds -- used only by the `cluttered` regime (see `_build_specs`), mirroring
# `textured.py`'s `noisy_background` flag: REAL_BACKGROUND_MANIFEST above is deliberately clean/
# uniform so `plain`/`varied` isolate pose variation; these give `cluttered` genuine real-world
# visual noise (fine repeated texture, high edge density) instead of just a pasted distractor,
# since a plain-textured backdrop makes the pasted object trivially separable for every method.
REAL_BUSY_BACKGROUND_MANIFEST: tuple[PhotoProvenance, ...] = (
    PhotoProvenance(
        category="leaf-litter",
        title="Dülmen, Wildpark -- 2020 -- 3415.jpg",
        file_url=(
            "https://upload.wikimedia.org/wikipedia/commons/2/27/"
            "D%C3%BClmen%2C_Wildpark_--_2020_--_3415.jpg"
        ),
        page_url=(
            "https://commons.wikimedia.org/wiki/File:D%C3%BClmen,_Wildpark_--_2020_--_3415.jpg"
        ),
        author="Dietmar Rabich",
        license="CC-BY-SA-4.0",
    ),
    PhotoProvenance(
        # Minor artifact: an archival HABS negative border/handwritten catalogue number sits near
        # one corner of the frame (a photographic-survey scan, not a re-photographed floor). Left
        # as-is rather than cropped, matching the "note it, don't over-engineer" posture already
        # taken for the carpet/asphalt substitutions above.
        category="mosaic-tile",
        title=(
            "Historic American Buildings Survey Verlin Berry, Photographer October 21, 1977 "
            "FIRST FLOOR, VIEW OF MOSAIC TILE FLOOR PATTERN - Kamm Building, 111 North Main "
            "Street, Mishawaka, HABS IND,71-MISH,1C-8.tif"
        ),
        file_url=(
            "https://upload.wikimedia.org/wikipedia/commons/1/1c/"
            "Historic_American_Buildings_Survey_Verlin_Berry%2C_Photographer_October_21%2C_1977_"
            "FIRST_FLOOR%2C_VIEW_OF_MOSAIC_TILE_FLOOR_PATTERN_-_Kamm_Building%2C_111_North_Main_"
            "Street%2C_Mishawaka%2C_HABS_IND%2C71-MISH%2C1C-8.tif"
        ),
        page_url=(
            "https://commons.wikimedia.org/wiki/File:Historic_American_Buildings_Survey_Verlin_"
            "Berry,_Photographer_October_21,_1977_FIRST_FLOOR,_VIEW_OF_MOSAIC_TILE_FLOOR_PATTERN_"
            "-_Kamm_Building,_111_North_Main_Street,_Mishawaka,_HABS_IND,71-MISH,1C-8.tif"
        ),
        author="Verlin Berry (NPS, Historic American Buildings Survey)",
        license="Public Domain",
    ),
)


class RealInsertionImageSpec(BaseModel):
    """One benchmark image: a background, a target object, how many times to paste it, jitter.

    ``regime`` mirrors `textured.py`'s three-regime split so the same orientation/scale
    exploration exists on real pixels: ``plain`` (fixed scale, upright -- the NCC-favourable
    baseline), ``varied`` (scale + rotation jitter -- the deep-feature-favourable case), and
    ``cluttered`` (moderate jitter plus a pasted-but-unrecorded distractor -- a precision stress).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    image_id: str
    regime: Regime = "varied"
    background: str
    target: str
    n_instances: int = Field(ge=1)
    seed: int
    scale_min: float = Field(default=0.6, gt=0.0)
    scale_max: float = Field(default=1.4, gt=0.0)
    rotation_deg: float = Field(default=25.0, ge=0.0)
    n_distractors: int = Field(default=0, ge=0)
    distractor: str | None = None


@dataclass(frozen=True, eq=False)
class Cutout:
    """A single-object RGBA cutout, tight to its alpha mask's own bounding box.

    ``eq=False`` because ``rgba`` is a NumPy array (no single-valued ``==``), matching
    :class:`~object_search.inference.fastsam.Proposal`'s convention.
    """

    rgba: npt.NDArray[np.uint8]  # (h, w, 4) BGR + alpha, uint8.
    category: str


# -- object-selection + extraction (pure enough to unit-test with fabricated proposals) ---------


_EDGE_SPAN_MARGIN = 0.02  # a box within this fraction of BOTH opposite edges is "full-span".


def _spans_full_dimension(box: BBox, image_w: int, image_h: int) -> bool:
    """True if ``box`` touches both left+right edges, or both top+bottom edges, of the frame.

    A real product photo's subject is essentially never cropped so tight it touches both opposite
    edges at once -- photographers leave margin. A proposal that does is almost certainly the
    backdrop (or backdrop-plus-object) rather than a tight object silhouette: measured on a claw
    hammer photo, the highest-area x centrality x objectness^3 candidate had ``box.w == image_w``
    exactly (spanned edge to edge) at objectness 0.71, while a smaller, MORE confident candidate
    (objectness 0.74) that did not span an edge was the actual tight hammer box -- pure score-based
    ranking cannot always tell these apart because objectness alone does not reliably discriminate
    backdrop from object, but this cheap geometric test does.
    """
    margin_x = image_w * _EDGE_SPAN_MARGIN
    margin_y = image_h * _EDGE_SPAN_MARGIN
    spans_width = box.x <= margin_x and (box.x + box.w) >= image_w - margin_x
    spans_height = box.y <= margin_y and (box.y + box.h) >= image_h - margin_y
    return spans_width or spans_height


def _select_object_proposal(
    proposals: list[Proposal], image_w: int, image_h: int
) -> Proposal | None:
    """Pick the proposal most likely to BE the photographed subject.

    FastSAM's automatic mode returns every plausible region with no notion of "the" subject. Area
    and centrality alone are not enough: measured on the real manifest, a plain product-photo
    backdrop routinely comes back as FastSAM's SINGLE LARGEST, well-centred proposal (a c-clamp
    photo's uniform background scored area_frac=0.51 at objectness=0.41), while the actual object
    is a smaller, much more confident proposal (area_frac=0.17 at objectness=0.72) that pure
    area x centrality ranks below it. ``objectness`` is cubed rather than used linearly because a
    linear or even squared weight still let that background blob outscore the real object in the
    measured case -- cubing was the smallest power that reliably flipped the ranking without
    needing per-photo tuning.

    Edge-to-edge-spanning proposals (:func:`_spans_full_dimension`) are excluded before ranking,
    when at least one non-spanning candidate exists -- a second, independent measured failure mode
    that objectness weighting alone did not catch (see that function's docstring).
    """
    if not proposals:
        return None
    non_spanning = [p for p in proposals if not _spans_full_dimension(p.box, image_w, image_h)]
    candidates = non_spanning or proposals

    centre_x, centre_y = image_w / 2.0, image_h / 2.0
    diagonal = float(np.hypot(image_w, image_h))

    def _score(proposal: Proposal) -> float:
        area = float(proposal.box.w * proposal.box.h)
        distance = float(np.hypot(proposal.box.cx - centre_x, proposal.box.cy - centre_y))
        centrality = 1.0 - distance / diagonal
        return area * centrality * (proposal.objectness**3)

    return max(candidates, key=_score)


_MERGE_GAP_FRAC = 0.03  # boxes within this fraction of the image diagonal count as "adjacent".
_MERGE_MIN_OBJECTNESS = 0.5  # only merge in reasonably confident regions, not stray noise.
_MERGE_MAX_AREA_RATIO = 2.0  # do not merge in a region more than this many times the primary's.
# A candidate overlapping the primary by more than this fraction of its own (smaller) area is a
# redundant re-detection of the SAME region, not a separate part of the object -- see the "false
# merge" note in _select_merge_partners's docstring.
_MERGE_MAX_OVERLAP_FRAC = 0.15


def _box_gap(a: BBox, b: BBox) -> float:
    """Euclidean gap between two boxes; ``0.0`` if they touch or overlap."""
    dx = max(a.x - (b.x + b.w), b.x - (a.x + a.w), 0)
    dy = max(a.y - (b.y + b.h), b.y - (a.y + a.h), 0)
    return float(np.hypot(dx, dy))


def _box_overlap_area(a: BBox, b: BBox) -> float:
    """Area of the intersection rectangle of two boxes (``0.0`` if disjoint)."""
    ix = max(0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
    iy = max(0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
    return float(ix * iy)


def _select_merge_partners(
    primary: Proposal, proposals: list[Proposal], image_w: int, image_h: int
) -> list[Proposal]:
    """Other proposals to union into ``primary``'s mask -- the compound-object case.

    Measured failure mode: FastSAM split a screwdriver into two nearly-equally-confident, mutually
    adjacent proposals -- the handle (objectness 0.88) and the thin metal shaft (objectness 0.88,
    touching the handle's box by ~1.4% of the shaft's own area) -- and
    :func:`_select_object_proposal` picked only the handle, since a single-proposal ranking has no
    way to say "these two are one object." A partner must be reasonably confident (not noise), not
    drastically larger than the primary (not the backdrop), and mostly *disjoint* from the primary
    while still touching or nearly touching its box (a separate physical part, not a redundant
    re-detection of the same region). That last condition is load-bearing: measured on a tennis
    ball, FastSAM's own proposals for the ball's brand-logo lettering, its lower half, and
    near-duplicate whole-ball boxes all satisfied "confident, right-sized, gap zero" (their boxes
    overlap the primary's), and without the overlap-fraction check every one of them was wrongly
    unioned in -- silently pulling in the ball's own background because the near-duplicate boxes
    extended slightly past it.
    """
    diagonal = float(np.hypot(image_w, image_h))
    gap_limit = diagonal * _MERGE_GAP_FRAC
    primary_area = float(primary.box.w * primary.box.h)
    partners = []
    for candidate in proposals:
        if candidate is primary or candidate.mask is None:
            continue
        if candidate.objectness < _MERGE_MIN_OBJECTNESS:
            continue
        if _spans_full_dimension(candidate.box, image_w, image_h):
            continue
        candidate_area = float(candidate.box.w * candidate.box.h)
        if candidate_area > primary_area * _MERGE_MAX_AREA_RATIO:
            continue
        overlap = _box_overlap_area(primary.box, candidate.box)
        if overlap > _MERGE_MAX_OVERLAP_FRAC * min(primary_area, candidate_area):
            continue
        if _box_gap(primary.box, candidate.box) > gap_limit:
            continue
        partners.append(candidate)
    return partners


def _keep_largest_component(alpha: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Zero every connected foreground component except the largest.

    Measured failure mode: a c-clamp photo's ruler/scale-bar (alternating light/dark squares)
    crossed the alpha threshold as ~10 tiny disconnected specks alongside the actual clamp body.
    They are individually negligible but visually read as noise in the composited cutout, so drop
    everything but the dominant blob (which, after :func:`_select_merge_partners` has already
    unioned any genuine same-object parts together, is the whole object).
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(alpha, connectivity=8)
    if count <= 2:  # background (0) + at most one foreground component -- nothing to drop.
        return alpha
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == largest_label, alpha, 0).astype(np.uint8)


def _mask_solidity(alpha: npt.NDArray[np.uint8]) -> float:
    """Fraction of ``alpha``'s own tight bounding box that is foreground (``alpha > 0``).

    A clean single-object mask -- even an irregular one like a hammer or a chess pawn -- fills a
    healthy fraction of its own bounding box. A badly fragmented mask (measured failure mode: a
    near-white mug on a near-white background gave FastSAM almost no edge to key off, and the
    thresholded mask came out as scattered patches rather than one blob) does not. This is the
    gate that catches that case rather than shipping it.
    """
    ys, xs = np.nonzero(alpha)
    if ys.size == 0:
        return 0.0
    height = int(ys.max() - ys.min()) + 1
    width = int(xs.max() - xs.min()) + 1
    return float(ys.size) / float(height * width)


class _CutoutAttempt(NamedTuple):
    cutout: Cutout
    solidity: float
    coverage: float


def _attempt_cutout(
    image: npt.NDArray[np.uint8],
    category: str,
    backend: ProposalBackend,
    config: FastSAMConfig,
) -> _CutoutAttempt | None:
    """One FastSAM pass: propose, select, merge same-object parts, threshold, erode, crop.

    Pure w.r.t. both quality gates -- :func:`extract_cutout` decides whether ``solidity``/
    ``coverage`` are good enough, this just measures them, so the retry-ladder loop lives in one
    place.
    """
    image_w, image_h = image.shape[1], image.shape[0]
    proposals = propose(image, config, backend=backend)
    proposal = _select_object_proposal(proposals, image_w, image_h)
    if proposal is None or proposal.mask is None:
        return None

    alpha = (proposal.mask >= _ALPHA_THRESHOLD).astype(np.uint8) * 255
    for partner in _select_merge_partners(proposal, proposals, image_w, image_h):
        if partner.mask is None:  # already excluded by _select_merge_partners; mypy can't see that
            continue
        alpha = np.maximum(alpha, (partner.mask >= _ALPHA_THRESHOLD).astype(np.uint8) * 255)

    alpha = np.asarray(cv2.erode(alpha, _MASK_ERODE_KERNEL, iterations=1), dtype=np.uint8)
    alpha = _keep_largest_component(alpha)
    ys, xs = np.nonzero(alpha)
    if ys.size == 0:
        return None

    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    solidity = _mask_solidity(alpha[y0:y1, x0:x1])
    coverage = float(ys.size) / float(image.shape[0] * image.shape[1])
    rgba = np.dstack([image[y0:y1, x0:x1], alpha[y0:y1, x0:x1]]).astype(np.uint8)
    rgba = _resize_to_long_edge(rgba, _CUTOUT_LONG_EDGE)
    return _CutoutAttempt(Cutout(rgba=rgba, category=category), solidity, coverage)


def extract_cutout(
    image: npt.NDArray[np.uint8],
    category: str,
    *,
    backend: ProposalBackend | None = None,
    config: FastSAMConfig | None = None,
) -> Cutout | None:
    """Segment the single dominant object out of ``image`` via FastSAM; return an RGBA cutout.

    1. Run FastSAM in automatic ("everything") mode with masks decoded.
    2. Select the proposal most likely to be the subject (:func:`_select_object_proposal`).
    3. Threshold its soft mask to a hard alpha channel at :data:`_ALPHA_THRESHOLD`, then erode it
       (:data:`_MASK_ERODE_KERNEL`) to trim the background-colour fringe a real photo's soft edge
       leaves behind.
    4. Crop tight to the eroded mask's own bounding box (which can be smaller than FastSAM's box)
       and pack ``(B, G, R, A)`` uint8.
    5. Gate on :func:`_mask_solidity` (floor :data:`_MIN_CUTOUT_SOLIDITY`) and frame coverage
       (ceiling :data:`_MAX_MASK_COVERAGE`) -- two distinct, measured failure modes (a fragmented
       mask and a whole-frame mask are both "solid" by one measure and not the other). When the
       default attempt fails either gate, :data:`_CONF_THRES_LADDER`'s remaining thresholds are
       tried in order and the first attempt clearing both gates wins.

    Never raises on a bad photo: returns ``None`` (logged) when FastSAM finds nothing, the mask is
    empty after thresholding, or every ladder rung fails a gate -- so the caller skips that one
    photo rather than shipping a fragmented or background-inclusive cutout.

    Args:
        image: The BGR source photo.
        category: The manifest category, for log messages only.
        backend: The FastSAM proposal backend. ``None`` constructs the default (requires the
            gitignored weight); tests inject a stub here.
        config: Decoding config. ``None`` uses FastSAM's default NMS with ``return_masks=True``
            (masks are mandatory for this use, unlike Method 5's default). Every ladder rung
            overrides ``conf_thres`` on this config regardless of what is passed here.
    """
    resolved_config = config if config is not None else FastSAMConfig(return_masks=True)
    resolved_backend = backend if backend is not None else default_backend()

    first_attempt: _CutoutAttempt | None = None
    for conf_thres in _CONF_THRES_LADDER:
        rung_config = resolved_config.model_copy(update={"conf_thres": conf_thres})
        attempt = _attempt_cutout(image, category, resolved_backend, rung_config)
        if first_attempt is None:
            first_attempt = attempt
        if attempt is None:
            continue
        if attempt.solidity >= _MIN_CUTOUT_SOLIDITY and attempt.coverage <= _MAX_MASK_COVERAGE:
            if conf_thres != _CONF_THRES_LADDER[0]:
                logger.info(f"{category}: recovered a usable cutout at conf_thres={conf_thres}")
            return attempt.cutout

    if first_attempt is None:
        logger.warning(f"{category}: FastSAM found no usable proposal; skipping cutout")
    else:
        logger.warning(
            f"{category}: no attempt cleared both quality gates (default-pass solidity="
            f"{first_attempt.solidity:.2f}, coverage={first_attempt.coverage:.2f}); "
            "skipping cutout"
        )
    return None


def _resize_to_long_edge(image: npt.NDArray[np.uint8], max_long_edge: int) -> npt.NDArray[np.uint8]:
    """Downscale ``image`` (any channel count) so its long edge is at most ``max_long_edge``.

    Never upscales -- a cutout already smaller than the cap is returned unchanged. ``INTER_AREA``
    is used for the shrink (correct for downsizing, and fine on an alpha channel: a resized alpha
    can land at any value in ``[0, 255]``, which every downstream alpha consumer already treats as
    a soft weight, not a hard 0/255 flag).
    """
    height, width = image.shape[:2]
    long_edge = max(height, width)
    if long_edge <= max_long_edge:
        return image
    scale = max_long_edge / long_edge
    resized = cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return np.ascontiguousarray(resized, dtype=np.uint8)


# -- compositing (pure numpy/opencv, unit-testable with fabricated cutouts) ---------------------


def _warp_cutout(
    cutout: Cutout, scale: float, angle_deg: float
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """Scale and rotate an RGBA cutout; return ``(warped_bgr, warped_alpha)`` at the new size."""
    height, width = cutout.rgba.shape[:2]
    scaled_w, scaled_h = max(1, round(width * scale)), max(1, round(height * scale))
    resized = cv2.resize(cutout.rgba, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)

    rot = cv2.getRotationMatrix2D((scaled_w / 2.0, scaled_h / 2.0), angle_deg, 1.0)
    cos_t, sin_t = abs(float(rot[0, 0])), abs(float(rot[0, 1]))
    out_w = int(scaled_w * cos_t + scaled_h * sin_t)
    out_h = int(scaled_w * sin_t + scaled_h * cos_t)
    rot[0, 2] += (out_w - scaled_w) / 2.0
    rot[1, 2] += (out_h - scaled_h) / 2.0
    warped = cv2.warpAffine(
        resized, rot, (out_w, out_h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0)
    )
    return warped[:, :, :3].astype(np.uint8), warped[:, :, 3].astype(np.uint8)


def _paste_rgba(
    canvas: npt.NDArray[np.uint8],
    bgr: npt.NDArray[np.uint8],
    alpha: npt.NDArray[np.uint8],
    x: int,
    y: int,
) -> None:
    """Alpha-blend ``bgr``/``alpha`` onto ``canvas`` in place at ``(x, y)``."""
    region = canvas[y : y + bgr.shape[0], x : x + bgr.shape[1]]
    weight = (alpha.astype(np.float32) / 255.0)[..., None]
    blended = weight * bgr.astype(np.float32) + (1.0 - weight) * region.astype(np.float32)
    region[:] = blended.astype(np.uint8)


def _alpha_bbox(alpha: npt.NDArray[np.uint8], x: int, y: int) -> BBox | None:
    """AABB of the non-zero alpha pixels, offset into canvas coordinates by ``(x, y)``."""
    ys, xs = np.nonzero(alpha > 0)
    if ys.size == 0:
        return None
    return BBox(
        x=x + int(xs.min()),
        y=y + int(ys.min()),
        w=int(xs.max() - xs.min()) + 1,
        h=int(ys.max() - ys.min()) + 1,
    )


def _overlaps(box: BBox, placed: list[BBox]) -> bool:
    gap = _PLACEMENT_GAP
    for other in placed:
        if not (
            box.x + box.w + gap <= other.x
            or other.x + other.w + gap <= box.x
            or box.y + box.h + gap <= other.y
            or other.y + other.h + gap <= box.y
        ):
            return True
    return False


def generate_real_insertion_image(
    spec: RealInsertionImageSpec,
    background: npt.NDArray[np.uint8],
    cutouts: Mapping[str, Cutout],
) -> SyntheticImage:
    """Paste ``spec.target`` onto ``background`` ``spec.n_instances`` times; exact ground truth.

    Rejection-samples non-overlapping positions (same strict-non-overlap + attempt-cap-with-
    achieved-count-honesty pattern as `chipset.py`/`textured.py`), scaling and rotating the cutout
    per instance. The ground-truth box is the AABB of the *warped* alpha mask, not the cutout's
    nominal size -- a rotated instance is boxed by its true extent.

    Distractors (a *different* manifest object, when ``spec.distractor`` is set) are pasted but
    never recorded -- genuine false-positive bait, same convention as the other synthetic sets.
    """
    rng = np.random.default_rng(spec.seed)
    canvas = background.copy()
    height, width = canvas.shape[:2]
    cutout = cutouts[spec.target]

    placed: list[BBox] = []
    scales: list[float] = []
    rotations: list[float] = []
    attempts = 0
    while len(placed) < spec.n_instances and attempts < _PLACEMENT_MAX_ATTEMPTS:
        attempts += 1
        scale = float(rng.uniform(spec.scale_min, spec.scale_max))
        angle = float(rng.uniform(-spec.rotation_deg, spec.rotation_deg))
        bgr, alpha = _warp_cutout(cutout, scale, angle)
        out_h, out_w = bgr.shape[:2]
        if out_w + 2 * _INSET >= width or out_h + 2 * _INSET >= height:
            continue
        x = int(rng.integers(_INSET, width - out_w - _INSET + 1))
        y = int(rng.integers(_INSET, height - out_h - _INSET + 1))
        box = _alpha_bbox(alpha, x, y)
        if box is None or _overlaps(box, placed):
            continue
        _paste_rgba(canvas, bgr, alpha, x, y)
        placed.append(box)
        scales.append(scale)
        rotations.append(angle)

    if spec.n_distractors and spec.distractor is not None and spec.distractor in cutouts:
        distractor_cutout = cutouts[spec.distractor]
        d_placed = 0
        d_attempts = 0
        while d_placed < spec.n_distractors and d_attempts < _PLACEMENT_MAX_ATTEMPTS:
            d_attempts += 1
            scale = float(rng.uniform(spec.scale_min, spec.scale_max))
            angle = float(rng.uniform(-spec.rotation_deg, spec.rotation_deg))
            bgr, alpha = _warp_cutout(distractor_cutout, scale, angle)
            out_h, out_w = bgr.shape[:2]
            if out_w + 2 * _INSET >= width or out_h + 2 * _INSET >= height:
                continue
            x = int(rng.integers(_INSET, width - out_w - _INSET + 1))
            y = int(rng.integers(_INSET, height - out_h - _INSET + 1))
            box = _alpha_bbox(alpha, x, y)
            if box is None or _overlaps(box, placed):  # keep clear of the real instances only
                continue
            _paste_rgba(canvas, bgr, alpha, x, y)
            placed.append(box)  # only for the non-overlap check against later distractors/reals
            d_placed += 1

    if len(placed) < spec.n_instances:
        logger.warning(
            f"{spec.image_id}: placed {len(placed)}/{spec.n_instances} instances after "
            f"{attempts} attempts; recording the achieved count as ground truth"
        )

    # Distractors were appended to `placed` only to keep later placements clear of them; drop them
    # before building ground truth -- distractors are bait, never boxes (EVAL-19/20 convention).
    real_boxes = placed[: len(scales)]
    order = sorted(range(len(real_boxes)), key=lambda i: (real_boxes[i].y, real_boxes[i].x))
    real_boxes = [real_boxes[i] for i in order]
    scales = [scales[i] for i in order]
    rotations = [rotations[i] for i in order]

    slice_metadata = SliceMetadata(
        true_instance_count=len(real_boxes),
        instance_scale_min=min(scales) if scales else None,
        instance_scale_max=max(scales) if scales else None,
        rotation_min_deg=min(rotations) if rotations else None,
        rotation_max_deg=max(rotations) if rotations else None,
    )
    return SyntheticImage(
        image=canvas, boxes=tuple(real_boxes), spec=None, slice_metadata=slice_metadata
    )


def _build_specs() -> tuple[RealInsertionImageSpec, ...]:
    """Three regime images per manifest object (plain/varied/cluttered), cycled across backgrounds.

    Mirrors `textured.py`'s regime stratification (EVAL-20) so the same orientation/scale
    exploration -- fixed pose, then scale+rotation jitter, then jitter plus a distractor -- exists
    on real pixels, not just rendered emblems. With 10 objects this is ``10 x 3 == 30`` images:
    enough to make the plain/varied/cluttered comparison meaningful without the sweep cost
    ballooning (comparable to the existing chipset(10) + textured(48) full-sweep footprint).
    `plain`/`varied` draw from the clean :data:`REAL_BACKGROUND_MANIFEST`; `cluttered` draws from
    the busy :data:`REAL_BUSY_BACKGROUND_MANIFEST` instead, so it stacks a genuinely harder
    background on top of its jitter and distractor, not just the same clean backdrop.
    """
    rng = np.random.default_rng(_MASTER_SEED)
    objects = [entry.category for entry in REAL_OBJECT_MANIFEST]
    clean_backgrounds = [entry.category for entry in REAL_BACKGROUND_MANIFEST]
    busy_backgrounds = [entry.category for entry in REAL_BUSY_BACKGROUND_MANIFEST]
    if not objects or not clean_backgrounds or not busy_backgrounds:
        return ()

    specs: list[RealInsertionImageSpec] = []
    clean_cycle = 0
    busy_cycle = 0
    for target in objects:
        distractor_pool = [name for name in objects if name != target]
        for regime in ("plain", "varied", "cluttered"):
            if regime == "cluttered":
                background = busy_backgrounds[busy_cycle % len(busy_backgrounds)]
                busy_cycle += 1
            else:
                background = clean_backgrounds[clean_cycle % len(clean_backgrounds)]
                clean_cycle += 1
            knobs = _REGIME_KNOBS[regime]
            n_distractors = int(knobs["n_distractors"])
            distractor = None
            if n_distractors and distractor_pool:
                distractor = distractor_pool[int(rng.integers(0, len(distractor_pool)))]
            else:
                n_distractors = 0
            specs.append(
                RealInsertionImageSpec(
                    image_id=f"real-{regime}-{target}",
                    regime=regime,
                    background=background,
                    target=target,
                    n_instances=int(rng.choice(_N_INSTANCE_CHOICES)),
                    seed=int(rng.integers(0, 2**31)),
                    scale_min=float(knobs["scale_min"]),
                    scale_max=float(knobs["scale_max"]),
                    rotation_deg=float(knobs["rotation_deg"]),
                    n_distractors=n_distractors,
                    distractor=distractor,
                )
            )
    return tuple(specs)


REAL_INSERTION_SPECS: tuple[RealInsertionImageSpec, ...] = _build_specs()


# -- fetching + orchestration (network / FastSAM boundary -- not exercised by the automated suite)


def _download(url: str) -> bytes | None:
    """``GET`` one Commons URL; ``None`` (logged) on any failure, never raises."""
    request = Request(url, headers={"User-Agent": _USER_AGENT})  # noqa: S310 - fixed Commons URLs
    try:
        with urlopen(request, timeout=30) as response:  # noqa: S310 - fixed https Commons URLs
            data: bytes = response.read()
    except (URLError, HTTPError, TimeoutError, ValueError) as exc:
        logger.warning(f"download failed for {url}: {exc}")
        return None
    return data


def fetch_real_photos(raw_dir: Path, *, force: bool = False) -> list[Path]:
    """Download every manifest photo from Wikimedia Commons into ``raw_dir/{objects,backgrounds}/``.

    Never leaves a partial file (downloads land at a ``.part`` path, renamed on success, mirroring
    `inference/models.py::fetch`). A download failure is logged and skipped -- one missing/renamed
    Commons file degrades only itself, the same graceful-degradation posture as `fetch-datasets`.
    Records each successfully-downloaded file's SHA-256 alongside its manifest provenance in
    ``<raw_dir>/provenance.json``, keyed by category so a partial re-run only updates what changed.
    """
    written: list[Path] = []
    entries: dict[str, dict[str, object]] = {}
    prov_path = raw_dir / "provenance.json"
    if prov_path.is_file():
        entries = {entry["category"]: entry for entry in json.loads(prov_path.read_text())}

    for kind, manifest, subdir in (
        ("object", REAL_OBJECT_MANIFEST, "objects"),
        ("background", REAL_BACKGROUND_MANIFEST, "backgrounds"),
        ("background", REAL_BUSY_BACKGROUND_MANIFEST, "backgrounds"),
    ):
        dest_dir = raw_dir / subdir
        dest_dir.mkdir(parents=True, exist_ok=True)
        for entry in manifest:
            suffix = Path(entry.file_url).suffix or ".jpg"
            dest = dest_dir / f"{entry.category}{suffix}"
            if dest.is_file() and not force:
                logger.info(f"{entry.category}: exists, skipping (use --force to re-download)")
                written.append(dest)
                continue
            data = _download(entry.file_url)
            if data is None:
                logger.warning(f"{kind} {entry.category!r}: download failed, skipping")
                continue
            part = dest.with_suffix(dest.suffix + ".part")
            part.write_bytes(data)
            part.replace(dest)
            written.append(dest)
            entries[entry.category] = {
                "category": entry.category,
                "kind": kind,
                "title": entry.title,
                "file_url": entry.file_url,
                "page_url": entry.page_url,
                "author": entry.author,
                "license": entry.license,
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            logger.info(f"{entry.category}: downloaded to {dest}")

    if entries:
        prov_path.write_text(
            json.dumps(sorted(entries.values(), key=lambda e: str(e["category"])), indent=2) + "\n",
            encoding="utf-8",
        )
    return written


def _raw_photo_path(directory: Path, category: str) -> Path | None:
    """The first file matching ``<category>.*`` in ``directory``, or ``None`` if absent."""
    if not directory.is_dir():
        return None
    matches = sorted(directory.glob(f"{category}.*"))
    return matches[0] if matches else None


def _load_background(entry: PhotoProvenance, raw_dir: Path) -> npt.NDArray[np.uint8] | None:
    """Read and downscale one raw background photo to :data:`_WORKING_LONG_EDGE`."""
    raw_path = _raw_photo_path(raw_dir / "backgrounds", entry.category)
    if raw_path is None:
        logger.warning(f"{entry.category}: raw background missing under {raw_dir}; skipping")
        return None
    raw_image = cv2.imread(str(raw_path))
    if raw_image is None:
        logger.warning(f"{entry.category}: failed to read {raw_path}; skipping")
        return None
    image = _resize_to_long_edge(np.asarray(raw_image, dtype=np.uint8), _WORKING_LONG_EDGE)
    return np.ascontiguousarray(image, dtype=np.uint8)


def _load_or_build_cutout(
    entry: PhotoProvenance,
    raw_dir: Path,
    cutouts_dir: Path,
    *,
    force: bool,
    backend: ProposalBackend | None,
) -> Cutout | None:
    """Load a cached cutout, or build (and cache) one from the raw object photo via FastSAM."""
    cache_path = cutouts_dir / f"{entry.category}.png"
    if cache_path.is_file() and not force:
        cached = cv2.imread(str(cache_path), cv2.IMREAD_UNCHANGED)
        if cached is not None and cached.ndim == 3 and cached.shape[2] == 4:
            return Cutout(rgba=np.asarray(cached, dtype=np.uint8), category=entry.category)
        logger.warning(f"{entry.category}: cached cutout unreadable, rebuilding")

    raw_path = _raw_photo_path(raw_dir / "objects", entry.category)
    if raw_path is None:
        logger.warning(f"{entry.category}: raw object photo missing under {raw_dir}; skipping")
        return None
    raw_image = cv2.imread(str(raw_path))
    if raw_image is None:
        logger.warning(f"{entry.category}: failed to read {raw_path}; skipping")
        return None
    image = np.asarray(raw_image, dtype=np.uint8)

    cutout = extract_cutout(image, entry.category, backend=backend)
    if cutout is None:
        return None
    cutouts_dir.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(cache_path), cutout.rgba):
        raise OSError(f"failed to write cutout cache to {cache_path}")
    return cutout


def write_real_insertion(
    out_dir: Path,
    raw_dir: Path,
    cutouts_dir: Path,
    *,
    force: bool = False,
    backend: ProposalBackend | None = None,
) -> list[Path]:
    """Build (or reuse cached) cutouts, composite every :data:`REAL_INSERTION_SPECS`, write GT.

    Requires ``fetch_real_photos`` to have already populated ``raw_dir``. Any manifest entry whose
    raw photo is missing, or whose FastSAM cutout fails, degrades only the image(s) that depend on
    it -- logged, not raised -- mirroring the achieved-count-honesty and graceful-degradation
    conventions used throughout ``synthetic/`` and ``eval/datasets.py``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    cutouts: dict[str, Cutout] = {}
    for entry in REAL_OBJECT_MANIFEST:
        cutout = _load_or_build_cutout(entry, raw_dir, cutouts_dir, force=force, backend=backend)
        if cutout is not None:
            cutouts[entry.category] = cutout

    backgrounds: dict[str, npt.NDArray[np.uint8]] = {}
    for entry in (*REAL_BACKGROUND_MANIFEST, *REAL_BUSY_BACKGROUND_MANIFEST):
        image = _load_background(entry, raw_dir)
        if image is not None:
            backgrounds[entry.category] = image

    written: list[Path] = []
    for spec in REAL_INSERTION_SPECS:
        image_path = out_dir / f"{spec.image_id}.jpg"
        if image_path.is_file() and not force:
            logger.info(f"{spec.image_id}: exists, skipping (use --force to overwrite)")
            written.append(image_path)
            continue
        if spec.target not in cutouts:
            logger.warning(f"{spec.image_id}: no cutout for target {spec.target!r}; skipping")
            continue
        if spec.background not in backgrounds:
            logger.warning(f"{spec.image_id}: no background {spec.background!r}; skipping")
            continue

        result = generate_real_insertion_image(spec, backgrounds[spec.background], cutouts)
        if not result.boxes:
            logger.warning(f"{spec.image_id}: zero instances placed; skipping (no valid GT)")
            continue
        jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, _COMPOSITE_JPEG_QUALITY]
        if not cv2.imwrite(str(image_path), result.image, jpeg_params):
            raise OSError(f"failed to write real-insertion image to {image_path}")

        sidecar = out_dir / f"{spec.image_id}.gt.json"
        payload = {
            "image": image_path.name,
            "width": int(result.image.shape[1]),
            "height": int(result.image.shape[0]),
            "regime": spec.regime,
            "target": spec.target,
            "background": spec.background,
            "seed": spec.seed,
            "requested_n": spec.n_instances,
            "achieved_n": len(result.boxes),
            "exemplar_index": 0,
            "slice_metadata": result.slice_metadata.model_dump(mode="json"),
            "boxes": [box.model_dump(mode="json") for box in result.boxes],
        }
        sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        logger.info(
            f"{spec.image_id}: {len(result.boxes)} instances (requested {spec.n_instances})"
        )
        written.append(image_path)
    return written
