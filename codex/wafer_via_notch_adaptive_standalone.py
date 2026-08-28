"""Full copy-paste Wafer_Via pipeline with adaptive-background V5 notch angle.

The die-map, YOLO, pitch, locate_die, coordinate, image-alignment, and
return APIs are inherited unchanged from wafer_via_notch_standalone.
Only detect_wafer_notch is overridden at the end of this file.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import cv2
import numpy as np


ImageInput = Union[str, Path, np.ndarray]
Point = Tuple[float, float]
PointPair = Tuple[Point, Point]
DetectionFormat = Literal[
    "auto", "point", "point_conf", "xyxy", "xywh", "yolo_txt", "xyxy_conf_class"
]
RefinementMode = Literal["auto", "gradient", "corner_color"]
AngleMode = Literal["robust", "local"]

__all__ = [
    "GridEstimate",
    "WaferBoundary",
    "WaferDieMap",
    "inspect_yolo_results",
    "parse_yolo_points",
    "refine_cross_point",
    "estimate_grid_from_yolo",
    "detect_wafer_boundary",
    "generate_die_map",
    "build_die_map_from_yolo",
    "build_die_map",
    "locate_die",
    "align_wafer_image",
    "transform_point_to_aligned",
    "transform_point_to_original",
    "make_clip_overlay",
    "make_wafer_overlay",
]


def _load_bgr(image: ImageInput) -> np.ndarray:
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] == 3:
            return image
        if image.ndim == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        raise ValueError(f"Unsupported image shape: {image.shape}")
    loaded = cv2.imread(str(image), cv2.IMREAD_COLOR)
    if loaded is None:
        raise FileNotFoundError(str(image))
    return loaded


def _point(value: Sequence[float]) -> Point:
    return float(value[0]), float(value[1])


@dataclass(frozen=True)
class GridEstimate:
    """Grid geometry measured in centre-clip pixel coordinates.

    ``angle_deg`` is both the clockwise-down image-space grid tilt and the
    positive OpenCV correction angle required to make that grid horizontal.
    """

    points_clip: Tuple[Point, ...]
    center_corner_clip: Point
    side_corner_clip: Point
    below_corner_clip: Point
    pitch_x: float
    pitch_y: float
    angle_deg: float
    angle_x_deg: float
    angle_y_deg: float
    angle_confidence: float
    refined: bool = False
    raw_points_clip: Tuple[Point, ...] = ()
    refinement_confidences: Tuple[float, ...] = ()
    refinement_mode: str = "none"
    center_corner_raw_clip: Optional[Point] = None
    side_corner_raw_clip: Optional[Point] = None
    below_corner_raw_clip: Optional[Point] = None
    angle_mode: str = "local"
    robust_angle_deg: Optional[float] = None
    local_angle_deg: Optional[float] = None
    angle_pairs_clip: Tuple[PointPair, ...] = ()
    angle_pairs_raw_clip: Tuple[PointPair, ...] = ()
    angle_pair_axes: Tuple[str, ...] = ()
    angle_pair_angles_deg: Tuple[float, ...] = ()
    angle_pair_residuals_deg: Tuple[float, ...] = ()
    angle_candidate_count: int = 0

    @property
    def pitch_x_points_clip(self) -> PointPair:
        """Refined/accepted centre and side points used for ``pitch_x``."""

        return self.center_corner_clip, self.side_corner_clip

    @property
    def pitch_y_points_clip(self) -> PointPair:
        """Refined/accepted centre and below points used for ``pitch_y``."""

        return self.center_corner_clip, self.below_corner_clip

    @property
    def pitch_x_points_raw_clip(self) -> PointPair:
        """Original YOLO centres corresponding to the X-pitch point pair."""

        return (
            self.center_corner_raw_clip or self.center_corner_clip,
            self.side_corner_raw_clip or self.side_corner_clip,
        )

    @property
    def pitch_y_points_raw_clip(self) -> PointPair:
        """Original YOLO centres corresponding to the Y-pitch point pair."""

        return (
            self.center_corner_raw_clip or self.center_corner_clip,
            self.below_corner_raw_clip or self.below_corner_clip,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "points_clip": list(self.points_clip),
            "center_corner_clip": self.center_corner_clip,
            "side_corner_clip": self.side_corner_clip,
            "below_corner_clip": self.below_corner_clip,
            "pitch_x": self.pitch_x,
            "pitch_y": self.pitch_y,
            "angle_deg": self.angle_deg,
            "angle_x_deg": self.angle_x_deg,
            "angle_y_deg": self.angle_y_deg,
            "angle_confidence": self.angle_confidence,
            "refined": self.refined,
            "raw_points_clip": list(self.raw_points_clip),
            "refinement_confidences": list(self.refinement_confidences),
            "refinement_mode": self.refinement_mode,
            "center_corner_raw_clip": self.center_corner_raw_clip,
            "side_corner_raw_clip": self.side_corner_raw_clip,
            "below_corner_raw_clip": self.below_corner_raw_clip,
            "pitch_x_points_clip": self.pitch_x_points_clip,
            "pitch_y_points_clip": self.pitch_y_points_clip,
            "pitch_x_points_raw_clip": self.pitch_x_points_raw_clip,
            "pitch_y_points_raw_clip": self.pitch_y_points_raw_clip,
            "angle_mode": self.angle_mode,
            "robust_angle_deg": self.robust_angle_deg,
            "local_angle_deg": self.local_angle_deg,
            "angle_pairs_clip": self.angle_pairs_clip,
            "angle_pairs_raw_clip": self.angle_pairs_raw_clip,
            "angle_pair_axes": self.angle_pair_axes,
            "angle_pair_angles_deg": self.angle_pair_angles_deg,
            "angle_pair_residuals_deg": self.angle_pair_residuals_deg,
            "angle_candidate_count": self.angle_candidate_count,
        }


@dataclass(frozen=True)
class WaferBoundary:
    center_px: Point
    radius_px: float
    contour_px: np.ndarray = field(repr=False)
    area_px: float = 0.0
    bbox_px: Tuple[int, int, int, int] = (0, 0, 0, 0)
    method: str = ""


@dataclass
class WaferDieMap:
    wafer_cx: int
    wafer_cy: int
    wafer_r: int
    pitch_x: float
    pitch_y: float
    x0: float
    y0: float
    die_w: int
    die_h: int
    pixel_per_unit: float
    dies: List[Dict[str, Any]] = field(default_factory=list)
    dies_by_index: Dict[Tuple[int, int], Dict[str, Any]] = field(default_factory=dict)
    image_shape: Tuple[int, int] = (0, 0)
    rotation_deg: float = 0.0
    grid_angle_deg: float = 0.0
    angle_confidence: float = 1.0
    edge_mode: str = "circle"
    wafer_boundary: Optional[WaferBoundary] = field(default=None, repr=False)
    grid_estimate: Optional[GridEstimate] = field(default=None, repr=False)
    aligned_image: Optional[np.ndarray] = field(default=None, repr=False)
    original_to_aligned_matrix: Optional[np.ndarray] = field(default=None, repr=False)
    aligned_to_original_matrix: Optional[np.ndarray] = field(default=None, repr=False)
    pitch_x_points_full: Optional[PointPair] = None
    pitch_y_points_full: Optional[PointPair] = None
    pitch_x_points_raw_full: Optional[PointPair] = None
    pitch_y_points_raw_full: Optional[PointPair] = None
    detected_pitch_x: Optional[float] = None
    detected_pitch_y: Optional[float] = None
    pitch_source: str = "direct"
    angle_pairs_full: Tuple[PointPair, ...] = ()
    angle_pairs_raw_full: Tuple[PointPair, ...] = ()
    # Coordinate-space and notch diagnostics. The base builder keeps the
    # historical ``original_image`` defaults. The notch builder returns an
    # axis-aligned map in ``aligned_image`` coordinates and records the source
    # angle/origin separately.
    coordinate_space: str = "original_image"
    source_grid_angle_deg: Optional[float] = None
    image_rotation_deg: float = 0.0
    source_x0: Optional[float] = None
    source_y0: Optional[float] = None
    original_wafer_boundary: Optional[WaferBoundary] = field(default=None, repr=False)
    notch_result: Optional[Any] = field(default=None, repr=False)
    notch_overlay_image: Optional[np.ndarray] = field(default=None, repr=False)
    notch_zoom_image: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def num_dies(self) -> int:
        return len(self.dies)

    @property
    def axis_x(self) -> Point:
        angle = math.radians(self.grid_angle_deg)
        return math.cos(angle), math.sin(angle)

    @property
    def axis_y(self) -> Point:
        ux, uy = self.axis_x
        return -uy, ux

    def get_die(self, ix: int, iy: int) -> Optional[Dict[str, Any]]:
        return self.dies_by_index.get((ix, iy))


# [SECTOR: 10_YOLO_COORDINATES] ----------------------------------------------
def _tensor_to_numpy(value: Any) -> Optional[np.ndarray]:
    """Convert a torch/numpy-like value to CPU numpy without importing torch."""

    if value is None:
        return None
    converted = value
    if hasattr(converted, "detach"):
        converted = converted.detach()
    if hasattr(converted, "cpu"):
        converted = converted.cpu()
    if hasattr(converted, "numpy"):
        converted = converted.numpy()
    return np.asarray(converted)


def inspect_yolo_results(results: Any, *, max_rows: int = 10) -> Dict[str, Any]:
    """Print and return a compact description of Ultralytics YOLO results.

    Accepts either the full ``results`` list returned by ``model(...)`` or one
    ``Results`` object. Tensor values are detached, moved to CPU, and previewed
    without requiring this module to import Ultralytics or torch.
    """

    if max_rows < 0:
        raise ValueError("max_rows must be zero or greater.")
    if hasattr(results, "boxes"):
        result_items = [results]
    elif isinstance(results, Sequence) and not isinstance(results, (str, bytes, bytearray)):
        result_items = list(results)
    else:
        raise TypeError("Pass the results list from model(...) or one Results object.")

    summary: Dict[str, Any] = {
        "results_type": type(results).__name__,
        "results_count": len(result_items),
        "items": [],
    }
    print(f"results type: {type(results).__module__}.{type(results).__name__}")
    print(f"results length: {len(result_items)}")

    for result_index, result in enumerate(result_items):
        item: Dict[str, Any] = {
            "index": result_index,
            "result_type": type(result).__name__,
            "orig_shape": getattr(result, "orig_shape", None),
            "path": str(getattr(result, "path", "")),
            "boxes": None,
        }
        print(f"\n[result {result_index}]")
        print(f"result type: {type(result).__module__}.{type(result).__name__}")
        print(f"original shape: {item['orig_shape']}")
        if item["path"]:
            print(f"path: {item['path']}")

        boxes = getattr(result, "boxes", None)
        if boxes is None:
            print("boxes: None")
            summary["items"].append(item)
            continue

        try:
            detection_count = len(boxes)
        except TypeError:
            detection_count = None
        box_summary: Dict[str, Any] = {
            "boxes_type": type(boxes).__name__,
            "detection_count": detection_count,
            "is_track": bool(getattr(boxes, "is_track", False)),
            "orig_shape": getattr(boxes, "orig_shape", None),
            "arrays": {},
        }
        item["boxes"] = box_summary
        print(f"boxes type: {type(boxes).__module__}.{type(boxes).__name__}")
        print(f"detection count: {detection_count}")
        print(f"boxes original shape: {box_summary['orig_shape']}")
        print(f"tracking boxes: {box_summary['is_track']}")

        for attribute in ("data", "xywh", "xywhn", "xyxy", "xyxyn", "conf", "cls", "id"):
            try:
                raw_value = getattr(boxes, attribute, None)
                array = _tensor_to_numpy(raw_value)
            except Exception as exc:  # diagnostic output must continue for other attributes
                box_summary["arrays"][attribute] = {"error": str(exc)}
                print(f"{attribute}: ERROR {exc}")
                continue
            if array is None:
                box_summary["arrays"][attribute] = None
                print(f"{attribute}: None")
                continue
            if array.ndim == 0:
                preview = array.copy()
            else:
                preview = array[:max_rows].copy()
            metadata = {
                "shape": tuple(int(value) for value in array.shape),
                "dtype": str(array.dtype),
                "preview": preview,
            }
            box_summary["arrays"][attribute] = metadata
            print(f"{attribute}: shape={metadata['shape']}, dtype={metadata['dtype']}")
            print(preview)
            if array.ndim > 0 and len(array) > max_rows:
                print(f"... {len(array) - max_rows} more row(s)")

        summary["items"].append(item)
    return summary


def _load_yolo_rows(path: Union[str, Path]) -> List[List[float]]:
    rows: List[List[float]] = []
    for line_number, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            rows.append([float(value) for value in line.replace(",", " ").split()])
        except ValueError as exc:
            raise ValueError(f"Invalid YOLO row at {path}:{line_number}: {raw!r}") from exc
    return rows


def parse_yolo_points(
    detections: Union[str, Path, np.ndarray, Sequence[Any]],
    image_size: Tuple[int, int] = (512, 512),
    *,
    detection_format: DetectionFormat = "auto",
    normalized: Optional[bool] = None,
    confidence_threshold: float = 0.25,
) -> List[Point]:
    """Convert common YOLO points/boxes to pixel centres.

    Supported rows are ``(x,y)``, ``(x,y,confidence)``, ``(x1,y1,x2,y2)``,
    standard YOLO txt ``(class,cx,cy,w,h[,confidence])``, and Ultralytics
    ``(x1,y1,x2,y2,confidence,class)``. Dictionaries may contain ``point``,
    ``center``, ``xyxy``, ``xywh``, or ``bbox``.
    """

    width, height = int(image_size[0]), int(image_size[1])
    if width <= 0 or height <= 0:
        raise ValueError("image_size must contain positive width and height.")
    if isinstance(detections, (str, Path)):
        detections = _load_yolo_rows(detections)
        if detection_format == "auto":
            detection_format = "yolo_txt"
    if hasattr(detections, "xyxy"):
        detections = np.asarray(getattr(detections, "xyxy"))
        if detection_format == "auto":
            detection_format = "xyxy"
    if isinstance(detections, np.ndarray):
        detections = detections.tolist()
    if not isinstance(detections, Sequence):
        raise TypeError("detections must be a path, ndarray, or sequence.")

    points: List[Point] = []
    for item in detections:
        item_format: DetectionFormat = detection_format
        confidence = 1.0
        values: Sequence[float]
        if isinstance(item, Mapping):
            confidence = float(item.get("confidence", item.get("conf", item.get("score", 1.0))))
            if "point" in item:
                values, item_format = item["point"], "point"
            elif "center" in item:
                values, item_format = item["center"], "point"
            elif "x" in item and "y" in item:
                values, item_format = (item["x"], item["y"]), "point"
            elif "xyxy" in item:
                values, item_format = item["xyxy"], "xyxy"
            elif "xywh" in item:
                values, item_format = item["xywh"], "xywh"
            elif "bbox" in item:
                values = item["bbox"]
                item_format = str(item.get("bbox_format", "xyxy"))  # type: ignore[assignment]
            else:
                raise ValueError(f"Unsupported detection dictionary keys: {sorted(item)}")
        else:
            values = item

        row = [float(value) for value in values]
        if item_format == "auto":
            if len(row) == 2:
                item_format = "point"
            elif len(row) == 3:
                item_format = "point_conf"
            elif len(row) == 4:
                item_format = "xyxy"
            elif len(row) == 5:
                item_format = "yolo_txt"
            elif len(row) == 6:
                # Resolve the two common six-column layouts. Normalized YOLO
                # labels start with an integer class and keep cx/cy/w/h in
                # [0,1]. Pixel-space six-column rows remain explicitly
                # selectable with detection_format when their layout is
                # ambiguous.
                looks_like_normalized_yolo = (
                    row[0] >= 0.0
                    and abs(row[0] - round(row[0])) < 1e-6
                    and all(0.0 <= value <= 1.0 for value in row[1:6])
                )
                item_format = "yolo_txt" if looks_like_normalized_yolo else "xyxy_conf_class"
            elif len(row) > 6:
                item_format = "xyxy_conf_class"
            else:
                raise ValueError(f"Cannot infer detection row format: {row}")

        if item_format == "point":
            if len(row) < 2:
                raise ValueError(f"Point needs two values: {row}")
            cx, cy = row[:2]
            coord_values = row[:2]
        elif item_format == "point_conf":
            if len(row) < 3:
                raise ValueError(f"point_conf needs x,y,confidence: {row}")
            cx, cy, confidence = row[:3]
            coord_values = row[:2]
        elif item_format == "xyxy":
            if len(row) < 4:
                raise ValueError(f"xyxy needs four values: {row}")
            x1, y1, x2, y2 = row[:4]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            coord_values = row[:4]
        elif item_format == "xywh":
            if len(row) < 4:
                raise ValueError(f"xywh needs four values: {row}")
            cx, cy = row[0], row[1]
            coord_values = row[:4]
        elif item_format == "yolo_txt":
            if len(row) < 5:
                raise ValueError(f"YOLO txt needs class,cx,cy,w,h: {row}")
            cx, cy = row[1], row[2]
            coord_values = row[1:5]
            if len(row) >= 6:
                confidence = row[5]
        elif item_format == "xyxy_conf_class":
            if len(row) < 6:
                raise ValueError(f"xyxy_conf_class needs six values: {row}")
            x1, y1, x2, y2, confidence = row[:5]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            coord_values = row[:4]
        else:
            raise ValueError(f"Unsupported detection_format: {item_format!r}")

        if confidence < confidence_threshold:
            continue
        is_normalized = normalized
        if is_normalized is None:
            is_normalized = bool(coord_values and max(abs(v) for v in coord_values) <= 1.000001)
        if is_normalized:
            cx *= width
            cy *= height
        if -1.0 <= cx <= width and -1.0 <= cy <= height:
            points.append((float(cx), float(cy)))

    deduplicated: List[Point] = []
    for candidate in points:
        if not any(math.hypot(candidate[0] - p[0], candidate[1] - p[1]) < 0.75 for p in deduplicated):
            deduplicated.append(candidate)
    return deduplicated


# [SECTOR: 20_COLOR_INVARIANT_REFINEMENT] ------------------------------------
def _profile_street_center(profile: np.ndarray, approximate: float, max_width: int) -> Tuple[float, float]:
    values = np.asarray(profile, dtype=np.float64)
    if values.size < 7:
        return approximate, 0.0
    values = cv2.GaussianBlur(values.reshape(1, -1), (0, 0), 1.0).ravel().astype(np.float32)
    dilated = cv2.dilate(values.reshape(1, -1), np.ones((1, 3), np.uint8)).ravel()
    local_max = values >= dilated
    peaks = np.flatnonzero(local_max)
    peaks = peaks[np.argsort(values[peaks])[-min(18, len(peaks)):]]
    baseline = float(np.median(values))
    best_center, best_score = float(approximate), 0.0
    min_width = 3
    for left in peaks:
        for right in peaks:
            width = int(right - left)
            if width < min_width or width > max_width:
                continue
            center = (float(left) + float(right)) / 2.0
            if abs(center - approximate) > max(4.0, max_width * 0.65):
                continue
            edge_strength = math.sqrt(max(0.0, values[left] - baseline) * max(0.0, values[right] - baseline))
            symmetry = math.exp(-abs(center - approximate) / max(2.0, max_width * 0.35))
            score = edge_strength * symmetry
            if score > best_score:
                best_center, best_score = center, score
    scale = float(np.percentile(values, 90) - baseline) + 1e-9
    confidence = float(np.clip(best_score / scale, 0.0, 1.0))
    return (best_center, confidence) if confidence >= 0.08 else (float(approximate), 0.0)


def _odd_kernel(value: int, maximum: int) -> int:
    kernel = max(1, int(value))
    if kernel % 2 == 0:
        kernel += 1
    largest = max(1, int(maximum))
    if largest % 2 == 0:
        largest -= 1
    return min(kernel, largest)


def _profile_colour_band_center(
    profile: np.ndarray,
    approximate: float,
    max_shift: float,
) -> Tuple[float, float]:
    """Find the centre of a high colour-distance band near an approximate point."""

    values = np.asarray(profile, dtype=np.float32).reshape(-1)
    if values.size < 7:
        return float(approximate), 0.0
    values = cv2.GaussianBlur(values.reshape(1, -1), (0, 0), 1.2).ravel()
    baseline = float(np.percentile(values, 25))
    high = float(np.percentile(values, 95))
    contrast = high - baseline
    if contrast <= 1e-6:
        return float(approximate), 0.0

    coordinates = np.arange(values.size, dtype=np.float32)
    allowed = np.abs(coordinates - float(approximate)) <= float(max_shift)
    proximity = np.exp(-np.abs(coordinates - float(approximate)) / max(2.0, max_shift * 0.45))
    score = (values - baseline) * proximity
    score[~allowed] = -np.inf
    peak_index = int(np.argmax(score))
    if not np.isfinite(score[peak_index]) or values[peak_index] <= baseline:
        return float(approximate), 0.0

    threshold = baseline + 0.45 * (float(values[peak_index]) - baseline)
    left = peak_index
    right = peak_index
    while left > 0 and values[left - 1] >= threshold:
        left -= 1
    while right + 1 < values.size and values[right + 1] >= threshold:
        right += 1
    band_coordinates = coordinates[left:right + 1]
    weights = np.maximum(values[left:right + 1] - baseline, 1e-6)
    center = float(np.average(band_coordinates, weights=weights))
    peak_contrast = max(0.0, float(values[peak_index]) - baseline)
    confidence = float(np.clip(peak_contrast / contrast, 0.0, 1.0))
    confidence *= float(math.exp(-abs(center - approximate) / max(2.0, max_shift)))
    return center, confidence


def _corner_colour_candidate(
    roi_bgr: np.ndarray,
    approximate_local: Point,
    *,
    corner_patch_ratio: float,
    noise_kernel: int,
) -> Tuple[Point, float]:
    """Find a cross band unlike all four local corner-die reference colours."""

    height, width = roi_bgr.shape[:2]
    if min(height, width) < 9:
        return approximate_local, 0.0
    kernel = _odd_kernel(noise_kernel, min(height, width))
    filtered = cv2.medianBlur(roi_bgr, kernel) if kernel >= 3 else roi_bgr
    lab = cv2.cvtColor(filtered, cv2.COLOR_BGR2LAB).astype(np.float32)
    patch = int(round(min(height, width) * float(corner_patch_ratio)))
    patch = int(np.clip(patch, 3, max(3, min(height, width) // 3)))
    corner_patches = (
        lab[:patch, :patch],
        lab[:patch, width - patch:],
        lab[height - patch:, :patch],
        lab[height - patch:, width - patch:],
    )
    references = np.asarray(
        [np.median(corner.reshape(-1, 3), axis=0) for corner in corner_patches],
        dtype=np.float32,
    )
    difference = lab[:, :, None, :] - references[None, None, :, :]
    distance_to_die = np.sqrt(np.sum(difference * difference, axis=3)).min(axis=2)
    # OpenCV supports large median kernels for uint8 images, while float32
    # medianBlur is limited to small kernels. The source image already receives
    # the requested denoising above, so a maximum 5x5 response filter is enough.
    response_kernel = _odd_kernel(min(noise_kernel, 5), min(height, width))
    if response_kernel >= 3:
        distance_to_die = cv2.medianBlur(distance_to_die.astype(np.float32), response_kernel)

    vertical_profile = np.median(distance_to_die, axis=0)
    horizontal_profile = np.median(distance_to_die, axis=1)
    max_shift = max(4.0, min(height, width) * 0.35)
    center_x, confidence_x = _profile_colour_band_center(
        vertical_profile, float(approximate_local[0]), max_shift
    )
    center_y, confidence_y = _profile_colour_band_center(
        horizontal_profile, float(approximate_local[1]), max_shift
    )
    confidence = math.sqrt(confidence_x * confidence_y)
    return (float(center_x), float(center_y)), float(confidence)


def refine_cross_point(
    clip_image: ImageInput,
    approximate_point: Point,
    *,
    search_radius: int = 18,
    max_street_width: Optional[int] = None,
    mode: RefinementMode = "auto",
    corner_patch_ratio: float = 0.22,
    corner_reference_weight: float = 0.70,
    noise_kernel: int = 5,
) -> Tuple[Point, float]:
    """Optionally refine a model point to the centre of the crossing streets.

    ``gradient`` uses paired Lab-gradient boundaries. ``corner_color`` learns
    the four local corner-die colours and selects bands unlike all of them.
    ``auto`` combines both, preferring the corner-colour cue under noise.
    """

    if mode not in ("auto", "gradient", "corner_color"):
        raise ValueError("mode must be 'auto', 'gradient', or 'corner_color'.")
    if not (0.05 <= float(corner_patch_ratio) <= 0.33):
        raise ValueError("corner_patch_ratio must be between 0.05 and 0.33.")
    if not (0.0 <= float(corner_reference_weight) <= 1.0):
        raise ValueError("corner_reference_weight must be between 0.0 and 1.0.")
    image = _load_bgr(clip_image)
    height, width = image.shape[:2]
    ax, ay = float(approximate_point[0]), float(approximate_point[1])
    radius = max(6, int(search_radius))
    x1, x2 = max(0, int(round(ax)) - radius), min(width, int(round(ax)) + radius + 1)
    y1, y2 = max(0, int(round(ay)) - radius), min(height, int(round(ay)) + radius + 1)
    roi = image[y1:y2, x1:x2]
    if min(roi.shape[:2]) < 7:
        return (ax, ay), 0.0

    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    gx_energy = np.zeros(lab.shape[:2], np.float32)
    gy_energy = np.zeros(lab.shape[:2], np.float32)
    for channel in cv2.split(lab):
        centered = channel - float(np.median(channel))
        robust_scale = float(np.percentile(np.abs(centered), 95)) + 1e-6
        normalized_channel = centered / robust_scale
        gx_energy += cv2.Scharr(normalized_channel, cv2.CV_32F, 1, 0) ** 2
        gy_energy += cv2.Scharr(normalized_channel, cv2.CV_32F, 0, 1) ** 2
    gx = np.sqrt(gx_energy)
    gy = np.sqrt(gy_energy)
    vertical_profile = np.median(gx, axis=0)
    horizontal_profile = np.median(gy, axis=1)
    street_width = int(max_street_width or max(8, radius))
    local_x, confidence_x = _profile_street_center(vertical_profile, ax - x1, street_width)
    local_y, confidence_y = _profile_street_center(horizontal_profile, ay - y1, street_width)
    gradient_confidence = math.sqrt(confidence_x * confidence_y)
    gradient_point = (float(x1 + local_x), float(y1 + local_y))

    colour_local, colour_confidence = _corner_colour_candidate(
        roi,
        (ax - x1, ay - y1),
        corner_patch_ratio=corner_patch_ratio,
        noise_kernel=noise_kernel,
    )
    colour_point = (float(x1 + colour_local[0]), float(y1 + colour_local[1]))

    if mode == "gradient":
        return (gradient_point, float(gradient_confidence)) if gradient_confidence > 0.0 else ((ax, ay), 0.0)
    if mode == "corner_color":
        return (colour_point, float(colour_confidence)) if colour_confidence > 0.0 else ((ax, ay), 0.0)
    if gradient_confidence <= 0.0 and colour_confidence <= 0.0:
        return (ax, ay), 0.0
    if colour_confidence <= 0.0:
        return gradient_point, float(gradient_confidence)
    if gradient_confidence <= 0.0:
        return colour_point, float(colour_confidence)

    disagreement = math.dist(gradient_point, colour_point)
    colour_weight = float(corner_reference_weight) * colour_confidence
    gradient_weight = (1.0 - float(corner_reference_weight)) * gradient_confidence
    if disagreement > max(4.0, radius * 0.55):
        if colour_weight >= gradient_weight:
            return colour_point, float(colour_confidence)
        return gradient_point, float(gradient_confidence)
    total_weight = colour_weight + gradient_weight
    if total_weight <= 1e-9:
        return colour_point, float(colour_confidence)
    combined = (
        (colour_point[0] * colour_weight + gradient_point[0] * gradient_weight) / total_weight,
        (colour_point[1] * colour_weight + gradient_point[1] * gradient_weight) / total_weight,
    )
    combined_confidence = float(np.clip(
        max(colour_confidence, gradient_confidence) * math.exp(-disagreement / max(4.0, radius)),
        0.0,
        1.0,
    ))
    return combined, combined_confidence


# [SECTOR: 30_GRID_ESTIMATION] ------------------------------------------------
def _fold_grid_angle(angle_deg: float) -> float:
    return (float(angle_deg) + 45.0) % 90.0 - 45.0


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    sorted_values, sorted_weights = values[order], weights[order]
    index = int(np.searchsorted(np.cumsum(sorted_weights), sorted_weights.sum() * 0.5))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


@dataclass(frozen=True)
class _RobustAngleEstimate:
    angle_deg: float
    confidence: float
    pair_indices: Tuple[Tuple[int, int], ...]
    pair_axes: Tuple[str, ...]
    pair_angles_deg: Tuple[float, ...]
    pair_residuals_deg: Tuple[float, ...]
    candidate_count: int


def _estimate_grid_orientation(
    points: np.ndarray,
    max_rotation_deg: float,
    inlier_tolerance_deg: float,
) -> _RobustAngleEstimate:
    angles: List[float] = []
    weights: List[float] = []
    pair_indices: List[Tuple[int, int]] = []
    pair_axes: List[str] = []
    seen_pairs: set[Tuple[int, int]] = set()
    neighbour_count = min(8, len(points) - 1)
    for index, point in enumerate(points):
        delta = points - point
        distances = np.linalg.norm(delta, axis=1)
        neighbours = np.argsort(distances)[1:neighbour_count + 1]
        for neighbour_value in neighbours:
            neighbour = int(neighbour_value)
            pair = (min(index, neighbour), max(index, neighbour))
            if pair in seen_pairs or distances[neighbour] < 3.0:
                continue
            seen_pairs.add(pair)
            vector = points[pair[1]] - points[pair[0]]
            folded_angle = _fold_grid_angle(
                math.degrees(math.atan2(float(vector[1]), float(vector[0])))
            )
            if abs(folded_angle) > max_rotation_deg:
                continue
            pair_indices.append(pair)
            pair_axes.append("x" if abs(float(vector[0])) >= abs(float(vector[1])) else "y")
            angles.append(folded_angle)
            # Longer grid-aligned spans are less sensitive to a 1~2 px point
            # localisation error. sqrt prevents a long span from dominating.
            weights.append(math.sqrt(float(distances[neighbour])))
    if not angles:
        raise ValueError("No horizontal/vertical YOLO point pairs were found within max_rotation_deg.")

    values = np.asarray(angles, dtype=np.float64)
    weight_array = np.asarray(weights, dtype=np.float64)
    first = _weighted_median(values, weight_array)
    broad_keep = np.abs(values - first) <= max(4.0, inlier_tolerance_deg * 2.0)
    robust_angle = _weighted_median(values[broad_keep], weight_array[broad_keep])
    residuals = np.abs(values - robust_angle)
    inliers = residuals <= inlier_tolerance_deg
    if not np.any(inliers):
        inliers[int(np.argmin(residuals))] = True
    robust_angle = _weighted_median(values[inliers], weight_array[inliers])
    residuals = np.abs(values - robust_angle)
    inliers = residuals <= inlier_tolerance_deg

    inlier_indices = np.flatnonzero(inliers)
    inlier_ratio = float(len(inlier_indices) / len(values))
    inlier_spread = _weighted_median(
        residuals[inliers], weight_array[inliers]
    ) if np.any(inliers) else inlier_tolerance_deg
    axes_used = {pair_axes[int(index)] for index in inlier_indices}
    axis_coverage = 1.0 if axes_used == {"x", "y"} else 0.70
    confidence = float(np.clip(
        inlier_ratio
        * math.exp(-inlier_spread / max(inlier_tolerance_deg, 1e-6))
        * axis_coverage,
        0.0,
        1.0,
    ))
    return _RobustAngleEstimate(
        angle_deg=float(robust_angle),
        confidence=confidence,
        pair_indices=tuple(pair_indices[int(index)] for index in inlier_indices),
        pair_axes=tuple(pair_axes[int(index)] for index in inlier_indices),
        pair_angles_deg=tuple(float(values[int(index)]) for index in inlier_indices),
        pair_residuals_deg=tuple(float(residuals[int(index)]) for index in inlier_indices),
        candidate_count=len(values),
    )


def _select_axis_neighbour(
    delta: np.ndarray,
    primary: np.ndarray,
    secondary: np.ndarray,
    *,
    prefer_positive: bool,
    axis_tolerance: float,
    perpendicular_tolerance_px: float,
) -> Tuple[int, np.ndarray]:
    along = delta @ primary
    across = delta @ secondary
    valid = (np.abs(along) >= 3.0) & (
        np.abs(across) <= np.maximum(perpendicular_tolerance_px, np.abs(along) * axis_tolerance)
    )
    candidates = np.flatnonzero(valid & ((along > 0) if prefer_positive else (along < 0)))
    if not len(candidates):
        candidates = np.flatnonzero(valid)
    if not len(candidates):
        raise ValueError("The centre corner has no usable neighbour on one grid axis.")
    selected = int(candidates[np.argmin(np.abs(along[candidates]))])
    vector = delta[selected].copy()
    if float(vector @ primary) < 0:
        vector *= -1.0
    return selected, vector


def estimate_grid_from_yolo(
    clip_image: ImageInput,
    detections: Union[str, Path, np.ndarray, Sequence[Any]],
    *,
    reference_point_clip: Optional[Point] = None,
    detection_format: DetectionFormat = "auto",
    normalized: Optional[bool] = None,
    confidence_threshold: float = 0.25,
    refine: bool = True,
    refine_radius: int = 18,
    refine_mode: RefinementMode = "auto",
    refine_max_street_width: Optional[int] = None,
    refine_corner_patch_ratio: float = 0.22,
    refine_corner_reference_weight: float = 0.70,
    refine_noise_kernel: int = 5,
    refine_min_confidence: float = 0.15,
    angle_mode: AngleMode = "robust",
    angle_inlier_tolerance_deg: float = 2.5,
    max_rotation_deg: float = 20.0,
    axis_tolerance: float = 0.18,
    perpendicular_tolerance_px: float = 5.0,
    max_axis_disagreement_deg: float = 3.0,
    strict: bool = True,
) -> GridEstimate:
    """Select the reference-nearest corner and its side/below neighbours.

    ``reference_point_clip`` is normally the detected full-wafer centre
    transformed into clip coordinates. Standalone calls fall back to the clip
    image centre for backwards compatibility.
    """

    if not (0.0 <= float(refine_min_confidence) <= 1.0):
        raise ValueError("refine_min_confidence must be between 0.0 and 1.0.")
    if angle_mode not in ("robust", "local"):
        raise ValueError("angle_mode must be 'robust' or 'local'.")
    if not (0.05 <= float(angle_inlier_tolerance_deg) <= 10.0):
        raise ValueError("angle_inlier_tolerance_deg must be between 0.05 and 10.0.")
    image = _load_bgr(clip_image)
    height, width = image.shape[:2]
    points = parse_yolo_points(
        detections,
        (width, height),
        detection_format=detection_format,
        normalized=normalized,
        confidence_threshold=confidence_threshold,
    )
    if len(points) < 3:
        raise ValueError(f"At least three YOLO cross-points are required; received {len(points)}.")
    raw_points = list(points)
    refinement_confidences = [0.0] * len(points)
    if refine:
        refined_points: List[Point] = []
        refinement_confidences = []
        for point in raw_points:
            candidate, candidate_confidence = refine_cross_point(
                image,
                point,
                search_radius=refine_radius,
                max_street_width=refine_max_street_width,
                mode=refine_mode,
                corner_patch_ratio=refine_corner_patch_ratio,
                corner_reference_weight=refine_corner_reference_weight,
                noise_kernel=refine_noise_kernel,
            )
            confidence_value = float(candidate_confidence)
            refinement_confidences.append(confidence_value)
            refined_points.append(
                candidate if confidence_value >= float(refine_min_confidence) else point
            )
        points = refined_points
    array = np.asarray(points, dtype=np.float64)
    selection_reference = np.asarray(
        reference_point_clip if reference_point_clip is not None else (width / 2.0, height / 2.0),
        dtype=np.float64,
    ).reshape(-1)
    if selection_reference.size != 2 or not np.all(np.isfinite(selection_reference)):
        raise ValueError("reference_point_clip must contain two finite coordinates.")
    center_index = int(np.argmin(np.linalg.norm(array - selection_reference, axis=1)))
    center = array[center_index]
    robust_angle = _estimate_grid_orientation(
        array, max_rotation_deg, angle_inlier_tolerance_deg
    )
    angle = math.radians(robust_angle.angle_deg)
    axis_x = np.array((math.cos(angle), math.sin(angle)))
    axis_y = np.array((-math.sin(angle), math.cos(angle)))
    delta = array - center
    delta[center_index] = 0.0

    side_index, side_vector = _select_axis_neighbour(
        delta, axis_x, axis_y,
        prefer_positive=True,
        axis_tolerance=axis_tolerance,
        perpendicular_tolerance_px=perpendicular_tolerance_px,
    )
    below_index, below_vector = _select_axis_neighbour(
        delta, axis_y, axis_x,
        prefer_positive=True,
        axis_tolerance=axis_tolerance,
        perpendicular_tolerance_px=perpendicular_tolerance_px,
    )
    if side_index == below_index:
        raise ValueError("The same YOLO point was selected for both pitch axes.")

    pitch_x = float(np.linalg.norm(side_vector))
    pitch_y = float(np.linalg.norm(below_vector))
    angle_x = _fold_grid_angle(math.degrees(math.atan2(side_vector[1], side_vector[0])))
    angle_y = _fold_grid_angle(math.degrees(math.atan2(-below_vector[0], below_vector[1])))
    disagreement = abs(angle_x - angle_y)
    if angle_mode == "local" and strict and disagreement > max_axis_disagreement_deg:
        raise ValueError(
            f"X/Y angle disagreement is {disagreement:.3f} deg, above "
            f"{max_axis_disagreement_deg:.3f} deg. Check YOLO false positives."
        )
    local_angle = float((angle_x + angle_y) / 2.0)
    local_confidence = float(np.clip(
        1.0 - disagreement / max(max_axis_disagreement_deg * 2.0, 1e-6),
        0.0,
        1.0,
    ))
    combined_angle = robust_angle.angle_deg if angle_mode == "robust" else local_angle
    confidence = robust_angle.confidence if angle_mode == "robust" else local_confidence
    angle_pairs = tuple(
        (_point(array[first]), _point(array[second]))
        for first, second in robust_angle.pair_indices
    )
    raw_angle_pairs = tuple(
        (_point(raw_points[first]), _point(raw_points[second]))
        for first, second in robust_angle.pair_indices
    )
    return GridEstimate(
        points_clip=tuple(_point(point) for point in array),
        center_corner_clip=_point(center),
        side_corner_clip=_point(array[side_index]),
        below_corner_clip=_point(array[below_index]),
        pitch_x=pitch_x,
        pitch_y=pitch_y,
        angle_deg=combined_angle,
        angle_x_deg=float(angle_x),
        angle_y_deg=float(angle_y),
        angle_confidence=confidence,
        refined=bool(refine),
        raw_points_clip=tuple(_point(point) for point in raw_points),
        refinement_confidences=tuple(refinement_confidences),
        refinement_mode=refine_mode if refine else "none",
        center_corner_raw_clip=_point(raw_points[center_index]),
        side_corner_raw_clip=_point(raw_points[side_index]),
        below_corner_raw_clip=_point(raw_points[below_index]),
        angle_mode=angle_mode,
        robust_angle_deg=robust_angle.angle_deg,
        local_angle_deg=local_angle,
        angle_pairs_clip=angle_pairs,
        angle_pairs_raw_clip=raw_angle_pairs,
        angle_pair_axes=robust_angle.pair_axes,
        angle_pair_angles_deg=robust_angle.pair_angles_deg,
        angle_pair_residuals_deg=robust_angle.pair_residuals_deg,
        angle_candidate_count=robust_angle.candidate_count,
    )


# [SECTOR: 40_WAFER_BOUNDARY] -------------------------------------------------
def detect_wafer_boundary(
    image: ImageInput,
    *,
    max_dimension: int = 2048,
    min_area_ratio: float = 0.08,
    max_area_ratio: float = 0.98,
) -> WaferBoundary:
    """Find the largest plausible central wafer contour without fixed colour."""

    source = _load_bgr(image)
    full_height, full_width = source.shape[:2]
    scale = min(1.0, float(max_dimension) / max(full_height, full_width))
    if scale < 1.0:
        small = cv2.resize(source, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small = source
    height, width = small.shape[:2]
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    border = max(2, int(round(min(height, width) * 0.015)))
    border_pixels = np.concatenate(
        (lab[:border].reshape(-1, 3), lab[-border:].reshape(-1, 3),
         lab[:, :border].reshape(-1, 3), lab[:, -border:].reshape(-1, 3)), axis=0
    )
    background = np.median(border_pixels, axis=0)
    distance = np.linalg.norm(lab - background, axis=2)
    distance_u8 = cv2.normalize(distance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, distance_mask = cv2.threshold(distance_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, gray_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks = (("lab_border_distance", distance_mask), ("gray_otsu", gray_mask),
             ("gray_otsu_inverse", cv2.bitwise_not(gray_mask)))
    kernel_size = max(3, int(round(min(height, width) * 0.006)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    image_area = float(height * width)
    best: Optional[Tuple[float, str, np.ndarray]] = None
    for method, raw_mask in masks:
        mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            area = float(cv2.contourArea(contour))
            area_ratio = area / image_area
            if not (min_area_ratio <= area_ratio <= max_area_ratio):
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            center_distance = math.hypot(cx - width / 2.0, cy - height / 2.0) / max(1.0, math.hypot(width, height) / 2.0)
            perimeter = float(cv2.arcLength(contour, True))
            circularity = 4.0 * math.pi * area / max(perimeter * perimeter, 1.0)
            fill = area / max(math.pi * radius * radius, 1.0)
            bx, by, bw, bh = cv2.boundingRect(contour)
            touches = sum((bx <= 1, by <= 1, bx + bw >= width - 1, by + bh >= height - 1))
            score = 2.0 * area_ratio + max(0.0, 1.0 - center_distance) + circularity + fill - 0.7 * touches
            if best is None or score > best[0]:
                best = score, method, contour
    if best is None:
        raise RuntimeError("Wafer boundary was not found; inspect background contrast or area limits.")

    _, method, contour_small = best
    contour = np.rint(contour_small.astype(np.float64) / scale).astype(np.int32)
    (cx, cy), radius = cv2.minEnclosingCircle(contour)
    bx, by, bw, bh = cv2.boundingRect(contour)
    return WaferBoundary(
        center_px=(float(cx), float(cy)),
        radius_px=float(radius),
        contour_px=contour,
        area_px=float(cv2.contourArea(contour)),
        bbox_px=(int(bx), int(by), int(bx + bw), int(by + bh)),
        method=method,
    )


# [SECTOR: 50_DIE_MAP] --------------------------------------------------------
def _normalize_edge_mode(edge_mode: str) -> str:
    value = str(edge_mode).strip().lower()
    aliases = {"partial": "circle", "disc": "circle", "outer": "ring", "grid": "ring", "all": "both"}
    value = aliases.get(value, value)
    if value not in ("circle", "ring", "both"):
        raise ValueError("edge_mode must be 'circle', 'ring', or 'both'.")
    return value


def _edge_flag(partial: bool, ring: bool, mode: str) -> bool:
    return partial if mode == "circle" else ring if mode == "ring" else partial or ring


def _die_polygon(origin: Point, ix: int, iy: int, pitch_x: float, pitch_y: float, angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    axis_x = np.array((math.cos(angle), math.sin(angle)), dtype=np.float64)
    axis_y = np.array((-math.sin(angle), math.cos(angle)), dtype=np.float64)
    base = np.asarray(origin, dtype=np.float64)
    left, right = ix * pitch_x, (ix + 1) * pitch_x
    top, bottom = -(iy + 1) * pitch_y, -iy * pitch_y
    return np.asarray(
        [base + left * axis_x + top * axis_y,
         base + right * axis_x + top * axis_y,
         base + right * axis_x + bottom * axis_y,
         base + left * axis_x + bottom * axis_y],
        dtype=np.float64,
    )


def _convex_intersection(
    first: np.ndarray,
    second: np.ndarray,
) -> Tuple[float, np.ndarray]:
    first_polygon = np.asarray(first, dtype=np.float32).reshape(-1, 2)
    second_polygon = np.asarray(second, dtype=np.float32).reshape(-1, 2)
    if len(first_polygon) < 3 or len(second_polygon) < 3:
        return 0.0, np.empty((0, 2), dtype=np.float64)
    area, intersection = cv2.intersectConvexConvex(first_polygon, second_polygon)
    if intersection is None or float(area) <= 1e-6:
        return 0.0, np.empty((0, 2), dtype=np.float64)
    return float(area), np.asarray(intersection, dtype=np.float64).reshape(-1, 2)


def _polygon_points(polygon: np.ndarray) -> Tuple[Point, ...]:
    return tuple(
        (float(point[0]), float(point[1]))
        for point in np.asarray(polygon).reshape(-1, 2)
    )


def generate_die_map(
    boundary: WaferBoundary,
    image_shape: Tuple[int, int],
    origin_full: Point,
    pitch_x: float,
    pitch_y: float,
    angle_deg: float,
    *,
    pixel_per_unit: float = 32.0,
    include_edge: bool = True,
    edge_margin: float = 1.0,
    edge_mode: str = "circle",
    angle_confidence: float = 1.0,
    grid_estimate: Optional[GridEstimate] = None,
) -> WaferDieMap:
    """Generate a rotated die lattice clipped by the actual wafer contour."""

    if pitch_x <= 0 or pitch_y <= 0:
        raise ValueError("pitch_x and pitch_y must be positive.")
    if pixel_per_unit <= 0:
        raise ValueError("pixel_per_unit must be positive.")
    mode = _normalize_edge_mode(edge_mode)
    contour = np.asarray(boundary.contour_px, dtype=np.int32).reshape(-1, 1, 2)
    contour_points = contour.reshape(-1, 2).astype(np.float64)
    wafer_hull = cv2.convexHull(contour).reshape(-1, 2).astype(np.float64)
    image_height, image_width = int(image_shape[0]), int(image_shape[1])
    image_polygon = np.asarray(
        (
            (0.0, 0.0),
            (float(image_width), 0.0),
            (float(image_width), float(image_height)),
            (0.0, float(image_height)),
        ),
        dtype=np.float64,
    )
    origin = np.asarray(origin_full, dtype=np.float64)
    angle = math.radians(angle_deg)
    axis_x = np.array((math.cos(angle), math.sin(angle)), dtype=np.float64)
    axis_y = np.array((-math.sin(angle), math.cos(angle)), dtype=np.float64)
    relative = contour_points - origin
    projected_x = relative @ axis_x
    projected_y = relative @ axis_y
    ix_min = int(math.floor(projected_x.min() / pitch_x)) - 1
    ix_max = int(math.ceil(projected_x.max() / pitch_x)) + 1
    iy_min = int(math.floor(-projected_y.max() / pitch_y)) - 1
    iy_max = int(math.ceil(-projected_y.min() / pitch_y)) + 1
    wafer_cx, wafer_cy = boundary.center_px
    radius = float(boundary.radius_px)
    margin_distance = max(0.0, (1.0 - float(edge_margin)) * radius)
    dies: List[Dict[str, Any]] = []
    by_index: Dict[Tuple[int, int], Dict[str, Any]] = {}

    for iy in range(iy_min, iy_max + 1):
        for ix in range(ix_min, ix_max + 1):
            polygon = _die_polygon(origin_full, ix, iy, pitch_x, pitch_y, angle_deg)
            center = polygon.mean(axis=0)
            signed_distance = cv2.pointPolygonTest(contour, (float(center[0]), float(center[1])), True)
            if edge_margin < 1.0 and signed_distance < margin_distance:
                continue
            full_area = abs(float(cv2.contourArea(polygon.astype(np.float32))))
            wafer_area, wafer_polygon = _convex_intersection(polygon, wafer_hull)
            if wafer_area <= 1e-6:
                continue
            partial = wafer_area < full_area - max(1e-3, full_area * 1e-6)
            if not include_edge and partial:
                continue
            visible_area, visible_polygon = _convex_intersection(wafer_polygon, image_polygon)
            image_partial = visible_area < wafer_area - max(1e-3, wafer_area * 1e-6)
            x1, y1 = np.floor(polygon.min(axis=0)).astype(int)
            x2, y2 = np.ceil(polygon.max(axis=0)).astype(int)
            if len(visible_polygon) >= 3:
                crop_x1, crop_y1 = np.floor(visible_polygon.min(axis=0)).astype(int)
                crop_x2, crop_y2 = np.ceil(visible_polygon.max(axis=0)).astype(int)
                crop_rect = (
                    int(np.clip(crop_x1, 0, image_width)),
                    int(np.clip(crop_y1, 0, image_height)),
                    int(np.clip(crop_x2, 0, image_width)),
                    int(np.clip(crop_y2, 0, image_height)),
                )
            else:
                crop_rect = (0, 0, 0, 0)
            cx, cy = int(round(float(center[0]))), int(round(float(center[1])))
            entry: Dict[str, Any] = {
                "index": (ix, iy),
                "center_px": (cx, cy),
                "rect_px": (int(x1), int(y1), int(x2), int(y2)),
                "crop_rect_px": crop_rect,
                "polygon_px": _polygon_points(polygon),
                "wafer_polygon_px": _polygon_points(wafer_polygon),
                "visible_polygon_px": _polygon_points(visible_polygon),
                "full_area_px": full_area,
                "wafer_area_px": wafer_area,
                "visible_area_px": visible_area,
                "real_coord": ((cx - wafer_cx) / pixel_per_unit, (wafer_cy - cy) / pixel_per_unit),
                "is_edge_partial": bool(partial),
                "is_image_partial": bool(image_partial),
                "is_outside_image": bool(visible_area <= 1e-6),
                "is_edge_ring": False,
                "is_edge": False,
            }
            dies.append(entry)
            by_index[(ix, iy)] = entry

    present = set(by_index)
    for entry in dies:
        ix, iy = entry["index"]
        ring = any((ix + dx, iy + dy) not in present
                   for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy)
        entry["is_edge_ring"] = bool(ring)
        entry["is_edge"] = bool(_edge_flag(entry["is_edge_partial"], ring, mode))

    return WaferDieMap(
        wafer_cx=int(round(wafer_cx)), wafer_cy=int(round(wafer_cy)), wafer_r=int(round(radius)),
        pitch_x=float(pitch_x), pitch_y=float(pitch_y),
        x0=float(origin_full[0]), y0=float(origin_full[1]),
        die_w=int(round(pitch_x)), die_h=int(round(pitch_y)),
        pixel_per_unit=float(pixel_per_unit), dies=dies, dies_by_index=by_index,
        image_shape=(int(image_shape[0]), int(image_shape[1])),
        rotation_deg=float(angle_deg), grid_angle_deg=float(angle_deg),
        angle_confidence=float(angle_confidence), edge_mode=mode,
        wafer_boundary=boundary, grid_estimate=grid_estimate,
    )


# [SECTOR: 60_LOCATE_DIE] -----------------------------------------------------
def locate_die(
    die_map: WaferDieMap,
    point: Optional[Point] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> Dict[str, Any]:
    """Return the rotated-grid die index and geometry for a point or bbox."""

    if (point is None) == (bbox is None):
        raise ValueError("Specify exactly one of point or bbox.")
    if bbox is not None:
        qx, qy = (float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0
        input_type = "bbox"
    else:
        qx, qy = float(point[0]), float(point[1])  # type: ignore[index]
        input_type = "point"
    relative = np.array((qx - die_map.x0, qy - die_map.y0), dtype=np.float64)
    axis_x = np.asarray(die_map.axis_x)
    axis_y = np.asarray(die_map.axis_y)
    ix = int(math.floor(float(relative @ axis_x) / die_map.pitch_x))
    iy = int(math.floor(-float(relative @ axis_y) / die_map.pitch_y))
    entry = die_map.get_die(ix, iy)
    polygon = _die_polygon((die_map.x0, die_map.y0), ix, iy,
                           die_map.pitch_x, die_map.pitch_y, die_map.grid_angle_deg)
    center = polygon.mean(axis=0)
    x1, y1 = np.floor(polygon.min(axis=0)).astype(int)
    x2, y2 = np.ceil(polygon.max(axis=0)).astype(int)
    contour = None if die_map.wafer_boundary is None else die_map.wafer_boundary.contour_px.reshape(-1, 1, 2)
    in_wafer = bool(contour is not None and cv2.pointPolygonTest(
        contour, (qx, qy), False) >= 0)
    if entry is not None:
        partial = bool(entry["is_edge_partial"])
        ring = bool(entry["is_edge_ring"])
        die_rect = entry["rect_px"]
        crop_rect = entry["crop_rect_px"]
        wafer_polygon = entry.get("wafer_polygon_px", ())
        visible_polygon = entry.get("visible_polygon_px", ())
        image_partial = bool(entry.get("is_image_partial", False))
        outside_image = bool(entry.get("is_outside_image", False))
    elif contour is not None:
        partial = not all(cv2.pointPolygonTest(contour, (float(p[0]), float(p[1])), False) >= 0 for p in polygon)
        ring = True
        die_rect = (int(x1), int(y1), int(x2), int(y2))
        crop_rect = (
            int(np.clip(x1, 0, die_map.image_shape[1])),
            int(np.clip(y1, 0, die_map.image_shape[0])),
            int(np.clip(x2, 0, die_map.image_shape[1])),
            int(np.clip(y2, 0, die_map.image_shape[0])),
        )
        wafer_polygon, visible_polygon = (), ()
        image_partial = bool(
            x1 < 0 or y1 < 0
            or x2 > die_map.image_shape[1] or y2 > die_map.image_shape[0]
        )
        outside_image = bool(
            x2 <= 0 or y2 <= 0
            or x1 >= die_map.image_shape[1] or y1 >= die_map.image_shape[0]
        )
    else:
        partial, ring = True, True
        die_rect = (int(x1), int(y1), int(x2), int(y2))
        crop_rect = die_rect
        wafer_polygon, visible_polygon = (), ()
        image_partial, outside_image = False, False
    rx = (qx - die_map.wafer_cx) / die_map.pixel_per_unit
    ry = (die_map.wafer_cy - qy) / die_map.pixel_per_unit
    cx, cy = int(round(float(center[0]))), int(round(float(center[1])))
    return {
        "input_type": input_type,
        "query_px": (qx, qy),
        "die_index": (ix, iy),
        "die_center_px": (cx, cy),
        "die_rect_px": die_rect,
        "crop_rect_px": crop_rect,
        "die_polygon_px": tuple((float(p[0]), float(p[1])) for p in polygon),
        "wafer_polygon_px": wafer_polygon,
        "visible_polygon_px": visible_polygon,
        "real_coord": (rx, ry),
        "real_distance": math.hypot(rx, ry),
        "die_real_coord": ((cx - die_map.wafer_cx) / die_map.pixel_per_unit,
                           (die_map.wafer_cy - cy) / die_map.pixel_per_unit),
        "wafer_center_px": (die_map.wafer_cx, die_map.wafer_cy),
        "corner_px": (die_map.x0, die_map.y0),
        "pitch_x": die_map.pitch_x,
        "pitch_y": die_map.pitch_y,
        "angle_deg": die_map.grid_angle_deg,
        "is_edge": bool(_edge_flag(partial, ring, die_map.edge_mode)),
        "is_edge_partial": partial,
        "is_image_partial": image_partial,
        "is_outside_image": outside_image,
        "is_edge_ring": ring,
        "edge_mode": die_map.edge_mode,
        "in_wafer": in_wafer,
    }


# [SECTOR: 65_ANGLE_ALIGNED_IMAGE] --------------------------------------------
def _alignment_matrices(center_px: Point, angle_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    matrix = cv2.getRotationMatrix2D(
        (float(center_px[0]), float(center_px[1])), float(angle_deg), 1.0
    ).astype(np.float64)
    inverse = cv2.invertAffineTransform(matrix).astype(np.float64)
    return matrix, inverse


def align_wafer_image(
    image: ImageInput,
    center_px: Point,
    angle_deg: float,
    *,
    interpolation: int = cv2.INTER_CUBIC,
    border_value: Tuple[int, int, int] = (0, 0, 0),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate a wafer image to horizontal and return image plus both transforms.

    Returns ``(aligned_image, original_to_aligned, aligned_to_original)``.
    Output size is identical to the input and rotation is around ``center_px``.
    """

    source = _load_bgr(image)
    height, width = source.shape[:2]
    matrix, inverse = _alignment_matrices(center_px, angle_deg)
    if abs(float(angle_deg)) < 1e-12:
        return source.copy(), matrix, inverse
    aligned = cv2.warpAffine(
        source,
        matrix,
        (width, height),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=tuple(int(value) for value in border_value),
    )
    return aligned, matrix, inverse


def _transform_point(matrix: np.ndarray, point: Point) -> Point:
    homogeneous = np.array((float(point[0]), float(point[1]), 1.0), dtype=np.float64)
    transformed = np.asarray(matrix, dtype=np.float64) @ homogeneous
    return float(transformed[0]), float(transformed[1])


def transform_point_to_aligned(die_map: WaferDieMap, point: Point) -> Point:
    """Convert one original-image point to ``die_map.aligned_image`` coordinates."""

    matrix = die_map.original_to_aligned_matrix
    if matrix is None:
        matrix, _ = _alignment_matrices(
            (die_map.wafer_cx, die_map.wafer_cy), die_map.grid_angle_deg
        )
    return _transform_point(matrix, point)


def transform_point_to_original(die_map: WaferDieMap, point: Point) -> Point:
    """Convert one aligned-image point back to original-image coordinates."""

    matrix = die_map.aligned_to_original_matrix
    if matrix is None:
        _, matrix = _alignment_matrices(
            (die_map.wafer_cx, die_map.wafer_cy), die_map.grid_angle_deg
        )
    return _transform_point(matrix, point)


# [SECTOR: 70_OVERLAY] --------------------------------------------------------
def make_clip_overlay(clip_image: ImageInput, estimate: GridEstimate) -> np.ndarray:
    overlay = _load_bgr(clip_image).copy()
    if estimate.angle_pairs_clip:
        angle_layer = overlay.copy()
        for pair, axis in zip(estimate.angle_pairs_clip, estimate.angle_pair_axes):
            first = tuple(np.rint(pair[0]).astype(int))
            second = tuple(np.rint(pair[1]).astype(int))
            colour = (255, 170, 40) if axis == "x" else (210, 80, 255)
            cv2.line(angle_layer, first, second, colour, 1, cv2.LINE_AA)
        overlay = cv2.addWeighted(angle_layer, 0.45, overlay, 0.55, 0.0)
    if estimate.raw_points_clip:
        for raw_point, refined_point in zip(estimate.raw_points_clip, estimate.points_clip):
            raw = tuple(np.rint(raw_point).astype(int))
            refined = tuple(np.rint(refined_point).astype(int))
            cv2.circle(overlay, raw, 3, (255, 255, 255), 1, cv2.LINE_AA)
            if raw != refined:
                cv2.line(overlay, raw, refined, (190, 190, 190), 1, cv2.LINE_AA)
    for point in estimate.points_clip:
        cv2.circle(overlay, tuple(np.rint(point).astype(int)), 4, (0, 215, 255), -1, cv2.LINE_AA)
    center = tuple(np.rint(estimate.center_corner_clip).astype(int))
    side = tuple(np.rint(estimate.side_corner_clip).astype(int))
    below = tuple(np.rint(estimate.below_corner_clip).astype(int))
    cv2.circle(overlay, center, 7, (0, 255, 0), -1, cv2.LINE_AA)
    cv2.arrowedLine(overlay, center, side, (255, 120, 0), 2, cv2.LINE_AA, tipLength=0.12)
    cv2.arrowedLine(overlay, center, below, (255, 0, 255), 2, cv2.LINE_AA, tipLength=0.12)
    for text, point, colour in (
        ("P0", center, (0, 255, 0)),
        ("PX", side, (255, 120, 0)),
        ("PY", below, (255, 0, 255)),
    ):
        cv2.putText(
            overlay, text, (point[0] + 6, point[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA,
        )
        cv2.putText(
            overlay, text, (point[0] + 6, point[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA,
        )
    label = (
        f"Px={estimate.pitch_x:.2f} Py={estimate.pitch_y:.2f} "
        f"A={estimate.angle_deg:.3f}deg({estimate.angle_mode}) "
        f"N={len(estimate.angle_pairs_clip)}/{estimate.angle_candidate_count} "
        f"R={estimate.refinement_mode}"
    )
    cv2.putText(overlay, label, (8, max(20, overlay.shape[0] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(overlay, label, (8, max(20, overlay.shape[0] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return overlay


def make_wafer_overlay(
    image: ImageInput,
    die_map: WaferDieMap,
    *,
    draw_dies: bool = True,
    thickness: int = 1,
) -> np.ndarray:
    overlay = _load_bgr(image).copy()
    if die_map.wafer_boundary is not None:
        cv2.drawContours(overlay, [die_map.wafer_boundary.contour_px], -1, (0, 255, 255), max(2, thickness * 2))
    if draw_dies:
        for die in die_map.dies:
            polygon_points = die.get("visible_polygon_px", die["polygon_px"])
            if not polygon_points:
                continue
            polygon = np.rint(np.asarray(polygon_points)).astype(np.int32)
            if len(polygon) < 3:
                continue
            colour = (0, 80, 255) if die["is_edge"] else (0, 220, 0)
            cv2.polylines(overlay, [polygon], True, colour, thickness, cv2.LINE_AA)
    cv2.circle(overlay, (die_map.wafer_cx, die_map.wafer_cy), max(4, thickness * 3), (0, 0, 255), -1)
    cv2.circle(overlay, (int(round(die_map.x0)), int(round(die_map.y0))),
               max(5, thickness * 4), (0, 255, 0), -1)
    return overlay


# [SECTOR: 80_PIPELINE] -------------------------------------------------------
def _legacy_build_die_map_from_yolo(
    wafer_image: ImageInput,
    clip_image: ImageInput,
    detections: Union[str, Path, np.ndarray, Sequence[Any]],
    *,
    clip_origin: Optional[Point] = None,
    detection_format: DetectionFormat = "auto",
    normalized: Optional[bool] = None,
    confidence_threshold: float = 0.25,
    refine: bool = True,
    refine_radius: int = 18,
    refine_mode: RefinementMode = "auto",
    refine_max_street_width: Optional[int] = None,
    refine_corner_patch_ratio: float = 0.22,
    refine_corner_reference_weight: float = 0.70,
    refine_noise_kernel: int = 5,
    refine_min_confidence: float = 0.15,
    angle_mode: AngleMode = "robust",
    angle_inlier_tolerance_deg: float = 2.5,
    pitch_size: Optional[Tuple[float, float]] = None,
    pixel_per_unit: float = 32.0,
    include_edge: bool = True,
    edge_margin: float = 1.0,
    edge_mode: str = "circle",
    boundary_max_dimension: int = 2048,
    return_aligned_image: bool = True,
    alignment_interpolation: int = cv2.INTER_CUBIC,
    alignment_border_value: Tuple[int, int, int] = (0, 0, 0),
) -> WaferDieMap:
    """End-to-end entry point for a full wafer image and centre-clip YOLO output.

    ``wafer_image`` and ``clip_image`` accept either an image path or an
    already-decoded OpenCV/numpy image. For production, uint8 BGR arrays are
    recommended. ``detections`` accepts memory arrays/lists as well as a YOLO
    text path. See ``[SECTOR: 90_USAGE_REFERENCE]`` below for detailed examples.
    """

    wafer = _load_bgr(wafer_image)
    clip = _load_bgr(clip_image)
    full_height, full_width = wafer.shape[:2]
    clip_height, clip_width = clip.shape[:2]
    if clip_origin is None:
        clip_origin = ((full_width - clip_width) / 2.0, (full_height - clip_height) / 2.0)
    boundary = detect_wafer_boundary(wafer, max_dimension=boundary_max_dimension)
    wafer_center_clip = (
        float(boundary.center_px[0]) - float(clip_origin[0]),
        float(boundary.center_px[1]) - float(clip_origin[1]),
    )
    estimate = estimate_grid_from_yolo(
        clip, detections,
        reference_point_clip=wafer_center_clip,
        detection_format=detection_format,
        normalized=normalized,
        confidence_threshold=confidence_threshold,
        refine=refine,
        refine_radius=refine_radius,
        refine_mode=refine_mode,
        refine_max_street_width=refine_max_street_width,
        refine_corner_patch_ratio=refine_corner_patch_ratio,
        refine_corner_reference_weight=refine_corner_reference_weight,
        refine_noise_kernel=refine_noise_kernel,
        refine_min_confidence=refine_min_confidence,
        angle_mode=angle_mode,
        angle_inlier_tolerance_deg=angle_inlier_tolerance_deg,
    )
    origin_full = (clip_origin[0] + estimate.center_corner_clip[0],
                   clip_origin[1] + estimate.center_corner_clip[1])
    if pitch_size is None:
        map_pitch_x, map_pitch_y = estimate.pitch_x, estimate.pitch_y
        pitch_source = "detected"
    else:
        pitch_values = np.asarray(pitch_size, dtype=np.float64).reshape(-1)
        if (
            pitch_values.size != 2
            or not np.all(np.isfinite(pitch_values))
            or np.any(pitch_values <= 0.0)
        ):
            raise ValueError("pitch_size must be a positive finite (pitch_x, pitch_y) pair.")
        map_pitch_x, map_pitch_y = float(pitch_values[0]), float(pitch_values[1])
        pitch_source = "manual"
    die_map = generate_die_map(
        boundary, (full_height, full_width), origin_full,
        map_pitch_x, map_pitch_y, estimate.angle_deg,
        pixel_per_unit=pixel_per_unit,
        include_edge=include_edge,
        edge_margin=edge_margin,
        edge_mode=edge_mode,
        angle_confidence=estimate.angle_confidence,
        grid_estimate=estimate,
    )

    def pair_to_full(pair: PointPair) -> PointPair:
        return (
            (
                float(clip_origin[0]) + float(pair[0][0]),
                float(clip_origin[1]) + float(pair[0][1]),
            ),
            (
                float(clip_origin[0]) + float(pair[1][0]),
                float(clip_origin[1]) + float(pair[1][1]),
            ),
        )

    die_map.pitch_x_points_full = pair_to_full(estimate.pitch_x_points_clip)
    die_map.pitch_y_points_full = pair_to_full(estimate.pitch_y_points_clip)
    die_map.pitch_x_points_raw_full = pair_to_full(estimate.pitch_x_points_raw_clip)
    die_map.pitch_y_points_raw_full = pair_to_full(estimate.pitch_y_points_raw_clip)
    die_map.angle_pairs_full = tuple(pair_to_full(pair) for pair in estimate.angle_pairs_clip)
    die_map.angle_pairs_raw_full = tuple(
        pair_to_full(pair) for pair in estimate.angle_pairs_raw_clip
    )
    die_map.detected_pitch_x = float(estimate.pitch_x)
    die_map.detected_pitch_y = float(estimate.pitch_y)
    die_map.pitch_source = pitch_source
    matrix, inverse = _alignment_matrices(
        (die_map.wafer_cx, die_map.wafer_cy), die_map.grid_angle_deg
    )
    die_map.original_to_aligned_matrix = matrix
    die_map.aligned_to_original_matrix = inverse
    if return_aligned_image:
        die_map.aligned_image, _, _ = align_wafer_image(
            wafer,
            (die_map.wafer_cx, die_map.wafer_cy),
            die_map.grid_angle_deg,
            interpolation=alignment_interpolation,
            border_value=alignment_border_value,
        )
    return die_map


build_die_map = _legacy_build_die_map_from_yolo


# [SECTOR: 90_USAGE_REFERENCE] ------------------------------------------------
# =============================================================================
# 상세 사용법 (전부 주석이므로 이 파일을 통째로 복사해도 자동 실행되지 않습니다.)
# =============================================================================
#
# 이 모듈의 가장 일반적인 처리 순서는 다음과 같습니다.
#
#   1) 10000x10000 전체 wafer 이미지를 메모리에 준비합니다.
#   2) 전체 이미지의 중앙에서 512x512 clip을 만듭니다.
#   3) 학습한 YOLO 모델로 clip 안의 십자점들을 검출합니다.
#   4) 전체 이미지, clip 이미지, YOLO 좌표를 build_die_map_from_yolo()에 넣습니다.
#   5) 반환된 dm으로 pitch/angle을 읽고 locate_die()를 호출합니다.
#
# 중요:
#   - 함수 하나만 복사하면 안 됩니다. 이 wafer_via.py 파일 전체를 복사해야 합니다.
#   - 이 파일 위쪽에 있는 numpy/cv2 import와 모든 class/helper 함수가 필요합니다.
#   - 이미지 색상으로 십자점을 새로 찾는 구조가 아닙니다. YOLO 좌표가 기준입니다.
#   - Python OpenCV 이미지는 일반적으로 dtype=uint8, shape=(H,W,3), BGR 순서입니다.
#   - JPEG/PNG 압축 bytes는 cv2.imdecode()로 ndarray로 바꾼 뒤 전달합니다.
#
# -----------------------------------------------------------------------------
# 예제 0. Ultralytics results/list/boxes 구조 먼저 출력하기
# -----------------------------------------------------------------------------
#
# results = model(center_clip_bgr)
# summary = inspect_yolo_results(results, max_rows=10)
#
# # list 길이가 1이면 일반적으로 입력 이미지가 1장이라는 뜻입니다.
# # 실제 십자점 검출 개수는 len(results[0].boxes)로 확인합니다.
# # 출력된 boxes.xywh 또는 boxes.data를 아래 예제처럼 detections로 변환합니다.
#
# -----------------------------------------------------------------------------
# 예제 1. 전체 wafer 이미지와 중앙 512x512 clip이 모두 메모리에 있을 때
# -----------------------------------------------------------------------------
#
# # wafer_bgr: 전체 원본 이미지, 예: shape=(10000, 10000, 3), dtype=uint8
# full_h, full_w = wafer_bgr.shape[:2]
# clip_w = 512
# clip_h = 512
# clip_x = (full_w - clip_w) // 2
# clip_y = (full_h - clip_h) // 2
#
# # copy()는 선택입니다. view를 그대로 전달해도 현재 로직은 정상 동작합니다.
# center_clip_bgr = wafer_bgr[
#     clip_y:clip_y + clip_h,
#     clip_x:clip_x + clip_w,
# ].copy()
#
# # exact center clip이면 clip_origin을 생략해도 같은 값이 자동 계산됩니다.
# clip_origin = (clip_x, clip_y)
#
# -----------------------------------------------------------------------------
# 예제 2. YOLO 검출 좌표를 메모리 배열로 준비하는 방법
# -----------------------------------------------------------------------------
#
# 아래 형식 중 하나를 사용합니다. 모든 좌표는 512x512 clip 기준입니다.
# confidence_threshold보다 작은 detection은 자동 제외됩니다.
#
# 2-A) 십자점 중심만 있는 Nx2 형식: [x, y]
# yolo_points = np.array([
#     [166.2, 164.8],
#     [256.1, 169.5],
#     [345.9, 174.1],
#     [161.5, 256.3],
#     [251.4, 260.9],  # 검출된 wafer 중심에 가장 가까운 점 -> center corner 후보
#     [341.3, 265.6],  # center corner 옆 점 -> pitch_x 계산
#     [246.7, 352.8],  # center corner 아래 점 -> pitch_y 계산
# ], dtype=np.float32)
#
# dm = build_die_map_from_yolo(
#     wafer_image=wafer_bgr,
#     clip_image=center_clip_bgr,
#     detections=yolo_points,
#     detection_format="point",   # auto도 가능
#     normalized=False,           # x,y가 pixel 좌표이므로 False
#     clip_origin=clip_origin,
# )
#
# 2-B) 점 + 신뢰도 Nx3 형식: [x, y, confidence]
# yolo_points_conf = np.array([
#     [251.4, 260.9, 0.98],
#     [341.3, 265.6, 0.96],
#     [246.7, 352.8, 0.97],
# ], dtype=np.float32)
#
# dm = build_die_map_from_yolo(
#     wafer_bgr,
#     center_clip_bgr,
#     yolo_points_conf,
#     detection_format="point_conf",  # auto도 가능
#     normalized=False,
#     confidence_threshold=0.25,
#     clip_origin=clip_origin,
# )
#
# 2-C) Ultralytics boxes.xyxy Nx4 형식: [x1, y1, x2, y2]
# yolo_xyxy = results[0].boxes.xyxy.cpu().numpy()
#
# dm = build_die_map_from_yolo(
#     wafer_bgr,
#     center_clip_bgr,
#     yolo_xyxy,
#     detection_format="xyxy",
#     normalized=False,
#     clip_origin=clip_origin,
# )
#
# 2-D) Ultralytics boxes.data Nx6 형식:
#      [x1, y1, x2, y2, confidence, class]
# yolo_data = results[0].boxes.data.cpu().numpy()
#
# dm = build_die_map_from_yolo(
#     wafer_bgr,
#     center_clip_bgr,
#     yolo_data,
#     detection_format="xyxy_conf_class",  # auto도 가능
#     normalized=False,
#     confidence_threshold=0.25,
#     clip_origin=clip_origin,
# )
#
# 2-E) 정규화 YOLO Nx5/Nx6 형식:
#      [class, center_x, center_y, width, height, (optional confidence)]
# yolo_normalized = np.array([
#     [0, 0.4910, 0.5096, 0.0200, 0.0200, 0.98],
#     [0, 0.6666, 0.5188, 0.0200, 0.0200, 0.96],
#     [0, 0.4818, 0.6891, 0.0200, 0.0200, 0.97],
# ], dtype=np.float32)
#
# dm = build_die_map_from_yolo(
#     wafer_bgr,
#     center_clip_bgr,
#     yolo_normalized,
#     detection_format="yolo_txt",  # normalized 6열은 auto 판별도 가능
#     normalized=True,
#     clip_origin=clip_origin,
# )
#
# 주의: pixel 단위의 [class,cx,cy,w,h,confidence] 6열은 다른 6열 형식과
#       모호하므로 detection_format="yolo_txt", normalized=False를 명시합니다.
#
# -----------------------------------------------------------------------------
# 예제 3. 가장 권장하는 전체 호출 형태
# -----------------------------------------------------------------------------
#
# dm = build_die_map_from_yolo(
#     wafer_image=wafer_bgr,          # 전체 wafer BGR ndarray
#     clip_image=center_clip_bgr,     # YOLO에 넣었던 512x512 BGR ndarray
#     detections=yolo_data,           # YOLO 결과 ndarray/list
#     clip_origin=(clip_x, clip_y),   # clip 왼쪽 위의 full-image 좌표
#     detection_format="xyxy_conf_class",
#     normalized=False,
#     confidence_threshold=0.25,
#     refine=True,                    # 기본값: 코너 die 색상 + Lab 경계로 중심 보정
#     refine_mode="auto",            # auto | corner_color | gradient
#     refine_radius=18,               # YOLO 중심 주변 탐색 반경(px)
#     refine_min_confidence=0.15,     # 이보다 낮으면 YOLO 원좌표 유지
#     angle_mode="robust",           # robust=전체 점, local=기존 P0/PX/PY
#     angle_inlier_tolerance_deg=2.5,
#     pitch_size=None,               # None=자동, 또는 (pitch_x, pitch_y)
#     pixel_per_unit=32.0,            # 실좌표 환산용 px/unit
#     include_edge=True,              # wafer 외곽의 partial die도 map에 포함
#     edge_margin=1.0,
#     edge_mode="circle",            # circle | ring | both
#     boundary_max_dimension=2048,    # 외곽선 검출용 downscale 상한
#     return_aligned_image=True,      # angle 보정된 full image를 dm에 저장
# )
#
# exact center clip이면 clip_origin은 생략할 수 있습니다.
# 하지만 생산 코드에서는 clip 위치 실수를 방지하기 위해 명시하는 것을 권장합니다.
# 전체 image 중심과 실제 wafer 중심이 달라도 괜찮습니다. build_die_map_from_yolo()는
# 먼저 wafer 외곽선을 검출하고, wafer 중심을 clip 좌표로 변환한 뒤 가장 가까운
# YOLO 십자점을 (0,0) grid origin으로 선택합니다.
#
# -----------------------------------------------------------------------------
# 예제 4. 반환값에서 center corner, pitch, angle 확인
# -----------------------------------------------------------------------------
#
# print("wafer center:", (dm.wafer_cx, dm.wafer_cy))
# print("wafer radius:", dm.wafer_r)
# print("center corner(full image):", (dm.x0, dm.y0))
# print("pitch_x:", dm.pitch_x)
# print("pitch_y:", dm.pitch_y)
# print("pitch source:", dm.pitch_source)  # detected / manual
# print("detected pitch:", (dm.detected_pitch_x, dm.detected_pitch_y))
# print("pitch_x points(full):", dm.pitch_x_points_full)
# print("pitch_y points(full):", dm.pitch_y_points_full)
# print("pitch_x raw points(full):", dm.pitch_x_points_raw_full)
# print("pitch_y raw points(full):", dm.pitch_y_points_raw_full)
# print("grid angle(deg):", dm.grid_angle_deg)
# print("angle confidence:", dm.angle_confidence)
# print("angle pairs(full):", dm.angle_pairs_full)
# print("angle pairs raw(full):", dm.angle_pairs_raw_full)
# print("number of dies:", dm.num_dies)
# print("full image shape:", dm.image_shape)
# print("aligned image:", None if dm.aligned_image is None else dm.aligned_image.shape)
#
# # 512 clip 안에서 실제 선택된 세 점도 확인할 수 있습니다.
# estimate = dm.grid_estimate
# if estimate is not None:
#     print("center corner in clip:", estimate.center_corner_clip)
#     print("side corner in clip:", estimate.side_corner_clip)
#     print("below corner in clip:", estimate.below_corner_clip)
#     print("pitch_x points(clip):", estimate.pitch_x_points_clip)
#     print("pitch_y points(clip):", estimate.pitch_y_points_clip)
#     print("pitch_x raw points(clip):", estimate.pitch_x_points_raw_clip)
#     print("pitch_y raw points(clip):", estimate.pitch_y_points_raw_clip)
#     print("angle from X vector:", estimate.angle_x_deg)
#     print("angle from Y vector:", estimate.angle_y_deg)
#     print("robust angle:", estimate.robust_angle_deg)
#     print("local angle:", estimate.local_angle_deg)
#     print("angle pairs(clip):", estimate.angle_pairs_clip)
#     print("angle pair axes:", estimate.angle_pair_axes)
#     print("angle pair residuals:", estimate.angle_pair_residuals_deg)
#
# angle 규칙:
#   - angle > 0이면 오른쪽 이웃으로 갈수록 영상의 Y가 증가하는 기울기입니다.
#   - 같은 +angle 값을 cv2.getRotationMatrix2D에 사용하면 수평 보정할 수 있습니다.
#   - die lattice와 locate_die()는 원본 이미지 좌표계를 유지합니다.
#   - dm.aligned_image에는 angle이 보정된 full image가 별도로 들어 있습니다.
#   - 보정 이미지의 좌표는 transform_point_to_original()로 되돌린 뒤 locate_die()에 넣습니다.
#
# -----------------------------------------------------------------------------
# 예제 5. full-image의 point 좌표로 die index 찾기
# -----------------------------------------------------------------------------
#
# result = locate_die(dm, point=(5499, 4700))
#
# print("die index:", result["die_index"])
# print("query point:", result["query_px"])
# print("die center:", result["die_center_px"])
# print("die polygon:", result["die_polygon_px"])
# print("die bounding rect:", result["die_rect_px"])
# print("real coordinate:", result["real_coord"])
# print("inside wafer:", result["in_wafer"])
# print("edge die:", result["is_edge"])
#
# index 규칙:
#   - ix + 방향 = 영상 오른쪽
#   - iy + 방향 = 영상 위쪽
#   - 회전각이 있어도 위 규칙은 grid 축 기준으로 유지됩니다.
#   - include_edge=True이면 중심이 wafer/image 밖이어도 일부가 wafer에 걸치는
#     die index는 유지됩니다.
#   - polygon_px는 index용 전체 polygon, wafer_polygon_px는 wafer 외곽 절단 결과,
#     visible_polygon_px는 wafer와 이미지 범위로 모두 절단한 결과입니다.
#
# -----------------------------------------------------------------------------
# 예제 6. 검사/불량 BBox 중심이 어느 die인지 찾기
# -----------------------------------------------------------------------------
#
# defect_bbox = (4880, 5080, 4980, 5180)  # full-image x1,y1,x2,y2
# result = locate_die(dm, bbox=defect_bbox)
# print(result["die_index"])
#
# point와 bbox를 동시에 넣으면 안 됩니다. 둘 중 정확히 하나만 지정합니다.
# bbox는 내부적으로 중심 좌표를 계산해서 해당 die를 찾습니다.
#
# -----------------------------------------------------------------------------
# 예제 7. index로 이미 생성된 die 정보 직접 조회
# -----------------------------------------------------------------------------
#
# die = dm.get_die(ix=2, iy=-3)
# if die is not None:
#     print(die["center_px"])
#     print(die["polygon_px"])
#     print(die["is_edge_partial"])
#     print(die["is_edge_ring"])
#
# -----------------------------------------------------------------------------
# 예제 8. 512 clip에서 center/side/below 선택 결과 오버레이
# -----------------------------------------------------------------------------
#
# if dm.grid_estimate is not None:
#     clip_overlay = make_clip_overlay(center_clip_bgr, dm.grid_estimate)
#     cv2.imwrite("clip_grid_overlay.png", clip_overlay)
#
# overlay 색상:
#   - 초록점: 선택된 center corner
#   - 파란 화살표: pitch_x를 만든 옆 점 방향
#   - 자홍 화살표: pitch_y를 만든 아래 점 방향
#   - 흰색 빈 원: 보정 전 YOLO bbox 중심
#   - 회색 선: 보정 전 중심에서 보정 후 중심까지의 이동량
#   - 노란점: 실제 pitch/angle 계산에 사용한 보정 후 십자점
#
# -----------------------------------------------------------------------------
# 예제 8-B. angle 보정된 full image 저장과 좌표 변환
# -----------------------------------------------------------------------------
#
# # return_aligned_image=True가 기본값입니다.
# if dm.aligned_image is not None:
#     cv2.imwrite("wafer_aligned.png", dm.aligned_image)
#
# # 원본 좌표 -> angle 보정 이미지 좌표
# original_point = (5499.0, 4700.0)
# aligned_point = transform_point_to_aligned(dm, original_point)
#
# # angle 보정 이미지 좌표 -> 원본 좌표
# restored_point = transform_point_to_original(dm, aligned_point)
#
# # locate_die()는 원본 좌표를 받으므로 aligned image에서 검출한 점은 되돌립니다.
# aligned_defect = (5503.2, 4691.8)
# original_defect = transform_point_to_original(dm, aligned_defect)
# result = locate_die(dm, point=original_defect)
#
# # 10000x10000 BGR 보정 이미지의 추가 메모리가 부담되면 다음 옵션을 사용합니다.
# dm_without_aligned = build_die_map_from_yolo(
#     wafer_bgr,
#     center_clip_bgr,
#     yolo_data,
#     clip_origin=clip_origin,
#     detection_format="xyxy_conf_class",
#     return_aligned_image=False,
# )
# # 이 경우 aligned_image만 None이며 좌표 변환 matrix는 계속 반환됩니다.
#
# -----------------------------------------------------------------------------
# 예제 9. full wafer 외곽선 + 전체 die map 오버레이
# -----------------------------------------------------------------------------
#
# wafer_overlay = make_wafer_overlay(
#     wafer_bgr,
#     dm,
#     draw_dies=True,
#     thickness=1,
# )
# cv2.imwrite("wafer_die_map_overlay.png", wafer_overlay)
#
# 주의: 10000x10000 uint8 BGR 원본은 약 300MB입니다.
# 기본 aligned_image가 약 300MB, make_wafer_overlay() 출력도 약 300MB가 추가됩니다.
# 메모리가 부족하면 draw_dies=False로 외곽선/기준점만 확인하거나 작은 preview를 씁니다.
#
# -----------------------------------------------------------------------------
# 예제 10. 노이즈가 심하고 YOLO 중심좌표가 street 중앙에서 벗어난 경우
# -----------------------------------------------------------------------------
#
# dm = build_die_map_from_yolo(
#     wafer_bgr,
#     center_clip_bgr,
#     yolo_data,
#     clip_origin=clip_origin,
#     detection_format="xywh",
#     refine=True,
#     refine_mode="auto",
#     refine_radius=24,               # 예상 bbox 오차보다 조금 크게 설정
#     refine_corner_patch_ratio=0.22,
#     refine_corner_reference_weight=0.70,
#     refine_noise_kernel=5,
#     refine_min_confidence=0.15,
# )
#
# refine=False:
#   - 학습 모델의 bbox 중심을 신뢰하고 색상 영향을 전혀 받지 않습니다.
#   - 보정 전/후 결과 비교 또는 영상에 실제 street가 보이지 않을 때 사용합니다.
#
# refine=True (기본값), refine_mode="auto":
#   1) 각 YOLO 점 주변 ROI의 네 꼭짓점 patch에서 die 대표색을 각각 median으로 학습합니다.
#   2) 네 die 대표색 중 어느 것과도 다른 픽셀을 street 후보로 만듭니다.
#   3) median filter와 X/Y projection으로 강한 점 노이즈를 억제하고 교차 중심을 찾습니다.
#   4) 동시에 Lab 양방향 경계쌍으로 구한 중심과 결합합니다.
#   5) 보정 confidence가 refine_min_confidence보다 낮으면 YOLO 원좌표를 그대로 씁니다.
#
# 색상 관련 중요 사항:
#   - 특정 빨강/초록/파랑 hue나 고정 RGB/HSV threshold를 사용하지 않습니다.
#   - 네 corner die가 서로 다른 색이어도 각각을 reference로 사용합니다.
#   - street 색이 corner die 중 하나와 비슷하면 corner_color confidence가 낮아지고
#     auto가 Lab gradient 결과를 사용합니다.
#
# 모드 선택:
#   - auto: 기본 권장. corner_color와 gradient를 함께 사용합니다.
#   - corner_color: 네 corner die 색상과 다른 band만 사용합니다.
#   - gradient: 기존 Lab 경계쌍 방식만 사용합니다.
#
# 튜닝 순서:
#   - 실제 중심이 탐색 범위 밖이면 refine_radius를 키웁니다(예: 18 -> 24/30).
#   - salt-and-pepper 노이즈가 강하면 refine_noise_kernel을 5 또는 7로 둡니다.
#   - die 내부 무늬가 corner patch를 많이 차지하면 refine_corner_patch_ratio를
#     0.15~0.28 범위에서 조정합니다.
#   - make_clip_overlay()로 흰 원 -> 노란 점 이동 방향을 반드시 확인합니다.
#
# 보정 상세값:
# estimate = dm.grid_estimate
# print(estimate.raw_points_clip)          # 보정 전 YOLO 중심들
# print(estimate.points_clip)              # 실제 계산에 사용된 보정 후 중심들
# print(estimate.refinement_confidences)   # 점별 보정 confidence
# print(estimate.refinement_mode)          # auto / corner_color / gradient / none
#
# -----------------------------------------------------------------------------
# 예제 11. 중앙이 아닌 위치에서 512 clip을 만든 경우
# -----------------------------------------------------------------------------
#
# clip_x = 4200
# clip_y = 4650
# center_clip_bgr = wafer_bgr[clip_y:clip_y + 512, clip_x:clip_x + 512]
#
# dm = build_die_map_from_yolo(
#     wafer_bgr,
#     center_clip_bgr,
#     yolo_data,
#     clip_origin=(clip_x, clip_y),  # 반드시 실제 clip 왼쪽 위 좌표
#     detection_format="xyxy_conf_class",
# )
#
# clip_origin을 잘못 넣으면 pitch와 angle은 맞아도 full-image center corner와 모든
# die 좌표가 같은 양만큼 이동하므로 반드시 실제 crop 위치와 일치시킵니다.
#
# -----------------------------------------------------------------------------
# 예제 12. JPEG/PNG 압축 bytes가 메모리에 있을 때
# -----------------------------------------------------------------------------
#
# # encoded_wafer_bytes / encoded_clip_bytes가 bytes 또는 bytearray라고 가정합니다.
# wafer_buffer = np.frombuffer(encoded_wafer_bytes, dtype=np.uint8)
# clip_buffer = np.frombuffer(encoded_clip_bytes, dtype=np.uint8)
# wafer_bgr = cv2.imdecode(wafer_buffer, cv2.IMREAD_COLOR)
# center_clip_bgr = cv2.imdecode(clip_buffer, cv2.IMREAD_COLOR)
# if wafer_bgr is None or center_clip_bgr is None:
#     raise ValueError("이미지 bytes decode 실패")
#
# dm = build_die_map_from_yolo(wafer_bgr, center_clip_bgr, yolo_data)
#
# -----------------------------------------------------------------------------
# 예제 13. 자주 발생하는 오류와 확인할 값
# -----------------------------------------------------------------------------
#
# ValueError: At least three YOLO cross-points are required
#   -> confidence_threshold 이후 남은 점이 center/side/below 최소 3개보다 적습니다.
#
# ValueError: The centre corner has no usable neighbour on one grid axis
#   -> center 기준 옆 또는 아래 점이 없거나 false positive 때문에 축 방향이 깨졌습니다.
#
# ValueError: X/Y angle disagreement ...
#   -> 옆 벡터와 아래 벡터가 직교 grid로 설명되지 않습니다.
#      YOLO 좌표, class filtering, 중복 검출을 확인합니다.
#
# RuntimeError: Wafer boundary was not found
#   -> wafer/background 대비가 너무 낮거나 wafer 면적이 기본 허용 범위를 벗어났습니다.
#
# 결과 승인 전 확인 권장:
#   1) dm.grid_estimate.center_corner_clip이 실제 중앙 십자점인지
#   2) side_corner_clip이 같은 row의 바로 옆 점인지
#   3) below_corner_clip이 같은 column의 바로 아래 점인지
#   4) pitch_x/pitch_y가 실제 die 간격과 일치하는지
#   5) robust/local angle 차이와 angle_pair_residuals_deg가 작은지
#   6) clip overlay와 wafer overlay가 실제 street/grid에 맞는지

# [SECTOR: 85_NOTCH_ANGLE] ---------------------------------------------------
# The notch detector and notch-only DM builder are embedded below. The legacy
# YOLO angle helpers above are never called by the exported builder.
__all__.extend([
    "AlignedNotchGuideResult",
    "NotchAngleResult",
    "detect_wafer_notch",
    "align_wafer_by_notch",
    "draw_aligned_wafer_notch_guide",
    "make_notch_overlay",
    "make_notch_zoom",
    "estimate_grid_from_yolo_notch",
])

@dataclass(frozen=True)
class NotchAngleResult:
    found: bool
    wafer_center_px: Point
    wafer_radius_px: float
    notch_point_px: Point
    notch_deepest_point_px: Point
    notch_angle_deg: float
    reference_angle_deg: float
    correction_angle_deg: float
    notch_depth_px: float
    notch_width_deg: float
    notch_width_px: float
    confidence: float
    radial_noise_px: float
    candidate_arc_px: Tuple[Point, ...] = field(repr=False)
    wafer_contour_px: np.ndarray
    segmentation_threshold: float
    scale: float
    failure_mode: str
    detection_method: str
    search_center_angle_deg: float
    search_half_width_deg: float
    edge_support: float
    circle_fit_residual_px: float


@dataclass(frozen=True)
class AlignedNotchGuideResult:
    """V5-style geometry and a writable overlay for one aligned wafer image.

    Every point is expressed in the input ``aligned_image`` coordinate system.
    ``overlay_image`` is a full-resolution BGR copy, so callers may draw their
    own ground-truth marks on it with ordinary OpenCV functions.
    """

    overlay_image: np.ndarray = field(repr=False)
    found: bool
    wafer_center_px: Point
    wafer_radius_px: float
    notch_center_px: Optional[Point]
    notch_point_px: Optional[Point]
    notch_left_px: Optional[Point]
    notch_right_px: Optional[Point]
    notch_angle_deg: Optional[float]
    reference_angle_deg: float
    residual_angle_deg: float
    notch_depth_px: float
    notch_width_deg: float
    effective_depth_threshold_px: float
    candidate_arc_px: Tuple[Point, ...]
    wafer_contour_px: np.ndarray = field(repr=False)
    search_center_angle_deg: float
    search_half_width_deg: float
    detection_method: str


def _load_bgr(image: ImageInput) -> np.ndarray:
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] == 3:
            return image
        if image.ndim == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        raise ValueError(f"Unsupported image shape: {image.shape}")
    loaded = cv2.imread(str(image), cv2.IMREAD_COLOR)
    if loaded is None:
        raise FileNotFoundError(str(image))
    return loaded


def _normalise_angle(angle_deg: float) -> float:
    return float((float(angle_deg) + 180.0) % 360.0 - 180.0)


def _lab_edge_strength(image_bgr: np.ndarray):
    """Return colour-transition strength without choosing either side's colour."""

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    lab = cv2.GaussianBlur(lab, (5, 5), 0).astype(np.float32)
    squared = np.zeros(lab.shape[:2], dtype=np.float32)
    for channel_index in range(3):
        channel = lab[:, :, channel_index]
        gx = cv2.Scharr(channel, cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(channel, cv2.CV_32F, 0, 1)
        squared += gx * gx + gy * gy
    edge = np.sqrt(squared).astype(np.float32)
    normaliser = float(np.percentile(edge, 99.5))
    if not np.isfinite(normaliser) or normaliser <= 1e-6:
        raise RuntimeError("Wafer outer edge was not found: image has no usable colour edge.")
    edge = np.clip(edge / normaliser, 0.0, 1.0)
    edge = cv2.GaussianBlur(edge, (3, 3), 0)
    return edge.astype(np.float32), normaliser


def _angle_distance_deg(angles_deg: np.ndarray, centre_deg: float) -> np.ndarray:
    return np.abs((angles_deg - float(centre_deg) + 180.0) % 360.0 - 180.0)


def _polar_sample(
    image: np.ndarray,
    center: Point,
    radii: np.ndarray,
    angles: np.ndarray,
) -> np.ndarray:
    map_x = (
        float(center[0]) + np.cos(angles)[:, None] * radii[None, :]
    ).astype(np.float32)
    map_y = (
        float(center[1]) + np.sin(angles)[:, None] * radii[None, :]
    ).astype(np.float32)
    return cv2.remap(
        image,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _initial_outer_radius(
    edge: np.ndarray,
    center: Point,
    angles: np.ndarray,
) -> Tuple[float, float]:
    """Find the outer circular edge supported by many unrelated angles."""

    height, width = edge.shape
    max_radius = min(
        float(center[0]),
        float(center[1]),
        float(width - 1) - float(center[0]),
        float(height - 1) - float(center[1]),
    )
    if max_radius < min(height, width) * 0.25:
        raise RuntimeError("Wafer centre hint leaves too little room for an outer circle.")
    radii = np.linspace(max_radius * 0.55, max_radius * 0.995, max(128, int(max_radius * 0.50)))
    polar = _polar_sample(edge, center, radii, angles)
    # A true circumference is present at the same radius over many angles.
    # The 65th percentile rejects isolated die/street edges.
    radial_score = np.percentile(polar, 65.0, axis=0).astype(np.float32)
    radial_score = cv2.GaussianBlur(radial_score.reshape(1, -1), (11, 1), 0).reshape(-1)
    outer_bias = 0.70 + 0.30 * (radii - radii[0]) / max(1e-6, radii[-1] - radii[0])
    scored = radial_score * outer_bias
    index = int(np.argmax(scored))
    return float(radii[index]), float(radial_score[index])


def _track_outer_edge(
    edge: np.ndarray,
    center: Point,
    radius: float,
    angles: np.ndarray,
    *,
    inward_px: float,
    outward_px: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Track a continuous edge close to the predicted circumference."""

    height, width = edge.shape
    max_corner_radius = math.hypot(width, height)
    radial_count = max(80, int(round(inward_px + outward_px)) * 2)
    radii = np.linspace(
        max(2.0, float(radius) - float(inward_px)),
        min(max_corner_radius, float(radius) + float(outward_px)),
        radial_count,
    )
    polar = _polar_sample(edge, center, radii, angles).astype(np.float32)
    samples_per_degree = len(angles) / 360.0
    angular_kernel = max(3, int(round(0.45 * samples_per_degree)) | 1)
    polar = cv2.GaussianBlur(polar, (3, angular_kernel), 0)

    # Only a weak distance prior is used. At a notch the outer-circle edge is
    # absent, so the actual inner arc must still be allowed to win.
    distance = np.abs(radii - float(radius))
    prior = np.exp(-distance / max(2.0, float(inward_px) * 0.55))
    scored = polar * (0.82 + 0.18 * prior[None, :])
    indices = np.argmax(scored, axis=1)
    boundary = radii[indices].astype(np.float32)
    support = polar[np.arange(len(angles)), indices].astype(np.float32)
    boundary = _circular_median(
        boundary, max(3, int(round(0.25 * samples_per_degree)) | 1)
    )
    return boundary, support


def _fit_circle_from_radial_profile(
    boundary: np.ndarray,
    support: np.ndarray,
    angles: np.ndarray,
    valid: np.ndarray,
) -> Tuple[float, float, float, float]:
    """Fit radius and centre offset from r(theta)=R+dx*cos+dy*sin."""

    design = np.column_stack((np.ones(len(angles)), np.cos(angles), np.sin(angles)))
    base = valid & np.isfinite(boundary) & np.isfinite(support)
    if int(base.sum()) < 60:
        raise RuntimeError("Wafer circle fit has insufficient supported edge angles.")
    support_floor = float(np.percentile(support[base], 35.0))
    base &= support >= support_floor
    keep = base.copy()
    coefficients = np.asarray((float(np.median(boundary[base])), 0.0, 0.0))
    residual_noise = float("inf")
    for _ in range(7):
        weights = np.clip(support[keep], 0.03, 1.0)
        weighted_design = design[keep] * np.sqrt(weights)[:, None]
        weighted_values = boundary[keep] * np.sqrt(weights)
        coefficients = np.linalg.lstsq(weighted_design, weighted_values, rcond=None)[0]
        residual = boundary - design @ coefficients
        centre = float(np.median(residual[keep]))
        residual_noise = float(1.4826 * np.median(np.abs(residual[keep] - centre)))
        new_keep = base & (np.abs(residual - centre) <= max(1.25, 3.0 * residual_noise))
        if int(new_keep.sum()) < 60 or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    return (
        float(coefficients[0]),
        float(coefficients[1]),
        float(coefficients[2]),
        residual_noise,
    )


def _circular_gaussian(values: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel_size = max(3, int(kernel_size) | 1)
    half = kernel_size // 2
    extended = np.concatenate((values[-half:], values, values[:half]))
    blurred = cv2.GaussianBlur(
        extended.reshape(1, -1), (kernel_size, 1), 0
    ).reshape(-1)
    return blurred[half:half + len(values)]


def _circular_median(values: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel_size = max(3, int(kernel_size) | 1)
    half = kernel_size // 2
    extended = np.concatenate((values[-half:], values, values[:half]))
    windows = np.lib.stride_tricks.sliding_window_view(extended, kernel_size)
    return np.median(windows, axis=1).astype(np.float32)


def _circular_candidate_groups(active: np.ndarray):
    if not np.any(active):
        return []
    if np.all(active):
        return [np.arange(len(active), dtype=np.int64)]
    starts = np.flatnonzero(active & ~np.roll(active, 1))
    groups = []
    for start_value in starts:
        start = int(start_value)
        values = [start]
        index = (start + 1) % len(active)
        while active[index] and index != start:
            values.append(index)
            index = (index + 1) % len(active)
        groups.append(np.asarray(values, dtype=np.int64))
    return groups


def detect_wafer_notch(
    image: ImageInput,
    *,
    reference_angle_deg: float = 90.0,
    max_dimension: int = 3072,
    angle_samples: int = 3600,
    baseline_window_deg: float = 10.0,
    radial_inner_ratio: float = 0.85,
    min_notch_depth_px: Optional[float] = None,
    min_notch_depth_ratio: float = 0.001,
    min_wide_notch_deg: float = 2.0,
    search_center_angle_deg: float = 90.0,
    search_half_width_deg: float = 45.0,
    wafer_center_hint_px: Optional[Point] = None,
    wafer_radius_hint_px: Optional[float] = None,
    failure_mode: Literal["error", "zero"] = "error",
    require_notch: Optional[bool] = None,
) -> NotchAngleResult:
    """Find a local inward deviation of the wafer's geometric outer edge.

    ``notch_angle_deg`` uses image coordinates (right=0, down=90). The returned
    ``correction_angle_deg`` is suitable for ``cv2.getRotationMatrix2D`` and
    moves the detected notch to ``reference_angle_deg``.

    No foreground/background colour is assumed. LAB colour-gradient magnitude
    supplies edge evidence only. The circle is fitted outside the configured
    bottom search sector, and the notch is the supported inward edge deviation
    within that sector. Hints are full-resolution image coordinates.

    ``failure_mode="error"`` raises when no notch is reliable.
    ``failure_mode="zero"`` returns ``found=False`` and a zero correction.
    ``require_notch`` remains as a backwards-compatible alias.
    """

    mode = str(failure_mode).strip().lower()
    if require_notch is not None:
        mode = "error" if bool(require_notch) else "zero"
    if mode not in ("error", "zero"):
        raise ValueError("failure_mode must be 'error' or 'zero'.")
    if not 2.0 <= float(search_half_width_deg) <= 120.0:
        raise ValueError("search_half_width_deg must be between 2 and 120 degrees.")
    if not 0.50 <= float(radial_inner_ratio) < 1.0:
        raise ValueError("radial_inner_ratio must be in [0.50, 1.0).")

    source = _load_bgr(image)
    full_height, full_width = source.shape[:2]
    scale = min(1.0, float(max_dimension) / max(full_height, full_width))
    if scale < 1.0:
        work = cv2.resize(source, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        work = source

    edge, edge_normaliser = _lab_edge_strength(work)
    work_height, work_width = work.shape[:2]

    angle_samples = max(720, int(angle_samples))
    angles = np.arange(angle_samples, dtype=np.float64) * (
        2.0 * math.pi / angle_samples
    )
    angles_deg = np.degrees(angles)
    search_distance = _angle_distance_deg(angles_deg, search_center_angle_deg)
    search_mask = search_distance <= float(search_half_width_deg)
    fit_mask = search_distance >= float(search_half_width_deg) + 5.0

    if wafer_center_hint_px is None:
        center = (work_width / 2.0, work_height / 2.0)
    else:
        center = (
            float(wafer_center_hint_px[0]) * scale,
            float(wafer_center_hint_px[1]) * scale,
        )
    if wafer_radius_hint_px is None:
        radius, _ = _initial_outer_radius(edge, center, angles)
    else:
        radius = float(wafer_radius_hint_px) * scale
    if radius <= min(work_height, work_width) * 0.20:
        raise RuntimeError("Estimated wafer radius is implausibly small.")

    circle_fit_noise = float("inf")
    # Re-centre using the first harmonic of the tracked radius. The notch
    # sector is excluded, so a wide or deep notch cannot pull the fitted circle.
    for _ in range(4):
        fit_window = max(12.0, radius * 0.08)
        boundary, support = _track_outer_edge(
            edge,
            center,
            radius,
            angles,
            inward_px=fit_window,
            outward_px=fit_window,
        )
        fitted_radius, offset_x, offset_y, circle_fit_noise = _fit_circle_from_radial_profile(
            boundary, support, angles, fit_mask
        )
        max_step = max(3.0, radius * 0.06)
        offset_length = math.hypot(offset_x, offset_y)
        if offset_length > max_step:
            factor = max_step / offset_length
            offset_x *= factor
            offset_y *= factor
        center = (center[0] + offset_x, center[1] + offset_y)
        radius = float(fitted_radius)
        if not np.isfinite(radius) or radius <= min(work_height, work_width) * 0.20:
            raise RuntimeError("Wafer circle fit produced an invalid radius.")
        if abs(offset_x) + abs(offset_y) < 0.20:
            break
    cx, cy = center
    if not (-0.10 * work_width <= cx <= 1.10 * work_width and -0.10 * work_height <= cy <= 1.10 * work_height):
        raise RuntimeError("Estimated wafer centre is outside the image.")

    inward_range = max(8.0, radius * (1.0 - float(radial_inner_ratio)))
    outward_range = max(5.0, radius * 0.018)
    boundary, support = _track_outer_edge(
        edge,
        center,
        radius,
        angles,
        inward_px=inward_range,
        outward_px=outward_range,
    )

    raw_depth = float(radius) - boundary
    supported_fit = fit_mask & (support >= np.percentile(support[fit_mask], 30.0))
    baseline_shift = float(np.median(raw_depth[supported_fit]))
    deficit = raw_depth - baseline_shift
    samples_per_degree = angle_samples / 360.0
    deficit = _circular_gaussian(
        deficit, max(3, int(round(0.30 * samples_per_degree)) | 1)
    )
    fit_residual = deficit[supported_fit]
    fit_residual_median = float(np.median(fit_residual))
    radial_noise = float(
        1.4826 * np.median(np.abs(fit_residual - fit_residual_median))
    )

    # ``baseline_window_deg`` remains accepted so old copy/paste calls do not
    # break. Geometry fitting outside the search sector replaces that baseline.
    _ = baseline_window_deg
    depth_limit = (
        float(min_notch_depth_px) * scale
        if min_notch_depth_px is not None
        else max(1.25, float(radius) * float(min_notch_depth_ratio))
    )
    candidate_threshold = max(
        0.50, depth_limit * 0.40, 2.5 * radial_noise
    )
    # Do not reject a notch because its edge is weak: low contrast is exactly
    # the difficult production case. Edge support contributes to confidence,
    # while geometry (depth/width/area) decides whether the depression exists.
    active = search_mask & (deficit >= candidate_threshold)
    # The notch edge can be interrupted by glare, die streets, or texture.
    # Join only short angular gaps; this preserves one physical depression
    # without requiring every ray to contain a strong gradient.
    bridge_kernel = max(3, int(round(0.80 * samples_per_degree)) | 1)
    half_bridge = bridge_kernel // 2
    extended_active = np.concatenate(
        (active[-half_bridge:], active, active[:half_bridge])
    ).astype(np.uint8).reshape(1, -1)
    extended_active = cv2.morphologyEx(
        extended_active,
        cv2.MORPH_CLOSE,
        np.ones((1, bridge_kernel), np.uint8),
    ).reshape(-1)
    active = extended_active[half_bridge:half_bridge + angle_samples].astype(bool)
    active &= search_mask
    groups = _circular_candidate_groups(active)
    degree_step = 360.0 / angle_samples
    candidates = []
    for indices in groups:
        width_deg = float(len(indices) * degree_step)
        if width_deg > 2.0 * float(search_half_width_deg):
            continue
        values = np.maximum(deficit[indices], 0.0)
        peak = float(values.max())
        area = float(values.sum() * degree_step)
        candidate_support = float(np.mean(support[indices]))
        score = area * math.sqrt(max(peak, 0.0)) * max(candidate_support, 0.02)
        candidates.append((score, peak, area, width_deg, candidate_support, indices))
    if candidates:
        _, peak_depth, candidate_area, notch_width_deg, candidate_support, candidate_indices = max(
            candidates, key=lambda item: item[0]
        )
        peak_index = int(candidate_indices[np.argmax(deficit[candidate_indices])])
    else:
        search_indices = np.flatnonzero(search_mask)
        peak_index = int(search_indices[np.argmax(deficit[search_indices])])
        peak_depth = float(deficit[peak_index])
        candidate_area = 0.0
        notch_width_deg = degree_step
        candidate_support = float(support[peak_index])
        candidate_indices = np.asarray((peak_index,), dtype=np.int64)

    minimum_width = degree_step
    strong_notch = bool(
        peak_depth >= max(depth_limit, 0.70, 4.5 * radial_noise)
        and notch_width_deg >= minimum_width
    )
    wide_shallow_notch = bool(
        peak_depth >= max(depth_limit * 0.45, 0.60, 3.0 * radial_noise)
        and notch_width_deg >= float(min_wide_notch_deg)
        and candidate_area >= depth_limit * max(1.5, float(min_wide_notch_deg))
    )
    found = bool(strong_notch or wide_shallow_notch)

    # The requested reference is the angular midpoint of the separated notch
    # region, not the depth-weighted apex. Unwrap group indices so a notch
    # crossing 0/360 degrees is handled correctly.
    unwrapped = np.unwrap(
        candidate_indices.astype(np.float64) * 2.0 * math.pi / angle_samples
    )
    notch_center_index = float(np.mean(unwrapped) * angle_samples / (2.0 * math.pi))
    notch_angle_rad = float(
        (notch_center_index % angle_samples) * 2.0 * math.pi / angle_samples
    )
    notch_angle_deg = float(math.degrees(notch_angle_rad) % 360.0)
    deepest_angle = float(angles[peak_index])
    notch_radius = float(boundary[peak_index])
    notch_deepest_point = (
        float(cx + math.cos(deepest_angle) * notch_radius),
        float(cy + math.sin(deepest_angle) * notch_radius),
    )
    # This is the user-confirmed red point: the notch centre direction at the
    # fitted wafer outer circle, i.e. where the circle would be without a cut.
    notch_point = (
        float(cx + math.cos(notch_angle_rad) * radius),
        float(cy + math.sin(notch_angle_rad) * radius),
    )

    notch_width_px = float(
        2.0 * radius * math.sin(math.radians(notch_width_deg) / 2.0)
    )
    snr = peak_depth / max(0.25, radial_noise)
    depth_score = (peak_depth - depth_limit) / max(depth_limit * 2.0, 1.0)
    area_score = candidate_area / max(depth_limit * 6.0, 1.0)
    confidence = float(np.clip(
        0.35 * min(1.0, snr / 8.0)
        + 0.25 * np.clip(depth_score, 0.0, 1.0)
        + 0.20 * np.clip(area_score, 0.0, 1.0)
        + 0.20 * np.clip(candidate_support, 0.0, 1.0),
        0.0,
        1.0,
    ))
    if not found:
        confidence = 0.0
        if mode == "error":
            raise RuntimeError(
                f"Wafer notch was not found: peak_depth={peak_depth / scale:.2f}px, "
                f"width={notch_width_deg:.2f}deg, required_depth={depth_limit / scale:.2f}px. "
                f"search={float(search_center_angle_deg):.1f}+/-{float(search_half_width_deg):.1f}deg. "
                "Use failure_mode='zero' to return angle 0, or provide wafer centre/radius hints."
            )
        notch_angle_deg = float(reference_angle_deg) % 360.0
        notch_angle_rad = math.radians(notch_angle_deg)
        notch_point = (
            float(cx + math.cos(notch_angle_rad) * radius),
            float(cy + math.sin(notch_angle_rad) * radius),
        )
        notch_deepest_point = notch_point
        candidate_indices = np.asarray((), dtype=np.int64)

    inv_scale = 1.0 / scale
    full_center = (float(cx * inv_scale), float(cy * inv_scale))
    full_radius = float(radius * inv_scale)
    full_notch_point = (
        float(notch_point[0] * inv_scale),
        float(notch_point[1] * inv_scale),
    )
    full_deepest_point = (
        float(notch_deepest_point[0] * inv_scale),
        float(notch_deepest_point[1] * inv_scale),
    )
    arc = tuple(
        (
            float((cx + math.cos(angles[index]) * boundary[index]) * inv_scale),
            float((cy + math.sin(angles[index]) * boundary[index]) * inv_scale),
        )
        for index in candidate_indices[::max(1, len(candidate_indices) // 48)]
    )
    contour_stride = max(1, angle_samples // 1440)
    contour_indices = np.arange(0, angle_samples, contour_stride, dtype=np.int64)
    contour_points = np.column_stack((
        cx + np.cos(angles[contour_indices]) * boundary[contour_indices],
        cy + np.sin(angles[contour_indices]) * boundary[contour_indices],
    ))
    contour_full = np.rint(contour_points * inv_scale).astype(np.int32).reshape(-1, 1, 2)
    return NotchAngleResult(
        found=found,
        wafer_center_px=full_center,
        wafer_radius_px=full_radius,
        notch_point_px=full_notch_point,
        notch_deepest_point_px=full_deepest_point,
        notch_angle_deg=notch_angle_deg,
        reference_angle_deg=float(reference_angle_deg),
        correction_angle_deg=_normalise_angle(
            notch_angle_deg - float(reference_angle_deg)
        ),
        notch_depth_px=float(peak_depth * inv_scale),
        notch_width_deg=notch_width_deg,
        notch_width_px=float(notch_width_px * inv_scale),
        confidence=confidence,
        radial_noise_px=float(radial_noise * inv_scale),
        candidate_arc_px=arc,
        wafer_contour_px=contour_full,
        segmentation_threshold=float(edge_normaliser),
        scale=scale,
        failure_mode=mode,
        detection_method="geometry_edge_bottom_sector",
        search_center_angle_deg=float(search_center_angle_deg) % 360.0,
        search_half_width_deg=float(search_half_width_deg),
        edge_support=float(candidate_support),
        circle_fit_residual_px=float(circle_fit_noise * inv_scale),
    )


def align_wafer_by_notch(
    image: ImageInput,
    result: Optional[NotchAngleResult] = None,
    *,
    reference_angle_deg: float = 90.0,
    failure_mode: Literal["error", "zero"] = "error",
    interpolation: int = cv2.INTER_CUBIC,
    border_value: Tuple[int, int, int] = (0, 0, 0),
):
    """Return ``(aligned_image, matrix, inverse_matrix, notch_result)``."""

    source = _load_bgr(image)
    if result is None:
        result = detect_wafer_notch(
            source,
            reference_angle_deg=reference_angle_deg,
            failure_mode=failure_mode,
        )
    height, width = source.shape[:2]
    matrix = cv2.getRotationMatrix2D(
        result.wafer_center_px, result.correction_angle_deg, 1.0
    )
    aligned = cv2.warpAffine(
        source,
        matrix,
        (width, height),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    inverse = cv2.invertAffineTransform(matrix)
    return aligned, matrix, inverse, result


def _transform_result_for_visual(
    result: NotchAngleResult,
    *,
    scale: float = 1.0,
    offset: Point = (0.0, 0.0),
) -> NotchAngleResult:
    def transform_point(point: Point) -> Point:
        return (
            float(point[0]) * float(scale) + float(offset[0]),
            float(point[1]) * float(scale) + float(offset[1]),
        )

    contour = result.wafer_contour_px.astype(np.float64) * float(scale)
    contour[:, :, 0] += float(offset[0])
    contour[:, :, 1] += float(offset[1])
    return replace(
        result,
        wafer_center_px=transform_point(result.wafer_center_px),
        wafer_radius_px=float(result.wafer_radius_px) * float(scale),
        notch_point_px=transform_point(result.notch_point_px),
        notch_deepest_point_px=transform_point(result.notch_deepest_point_px),
        notch_depth_px=float(result.notch_depth_px) * float(scale),
        notch_width_px=float(result.notch_width_px) * float(scale),
        radial_noise_px=float(result.radial_noise_px) * float(scale),
        candidate_arc_px=tuple(transform_point(point) for point in result.candidate_arc_px),
        wafer_contour_px=np.rint(contour).astype(np.int32),
        circle_fit_residual_px=float(result.circle_fit_residual_px) * float(scale),
    )


def make_notch_overlay(
    image: ImageInput,
    result: NotchAngleResult,
    *,
    thickness: int = 2,
    max_dimension: Optional[int] = None,
) -> np.ndarray:
    """Visualise the user-confirmed outer reference and deepest diagnostic."""

    source = _load_bgr(image)
    if max_dimension is not None and int(max_dimension) > 0:
        visual_scale = min(
            1.0,
            float(max_dimension) / max(source.shape[0], source.shape[1]),
        )
    else:
        visual_scale = 1.0
    if visual_scale < 1.0:
        source = cv2.resize(
            source,
            None,
            fx=visual_scale,
            fy=visual_scale,
            interpolation=cv2.INTER_AREA,
        )
        result = _transform_result_for_visual(result, scale=visual_scale)
        thickness = max(1, int(round(float(thickness) * visual_scale)))
    overlay = source.copy()
    center = tuple(int(round(v)) for v in result.wafer_center_px)
    notch = tuple(int(round(v)) for v in result.notch_point_px)
    deepest = tuple(int(round(v)) for v in result.notch_deepest_point_px)
    radius = int(round(result.wafer_radius_px))
    cv2.circle(overlay, center, radius, (255, 255, 0), thickness, cv2.LINE_AA)
    cv2.drawContours(
        overlay, [result.wafer_contour_px], -1, (120, 120, 120), 1, cv2.LINE_AA
    )
    for boundary_angle in (
        result.search_center_angle_deg - result.search_half_width_deg,
        result.search_center_angle_deg + result.search_half_width_deg,
    ):
        angle_rad = math.radians(boundary_angle)
        endpoint = (
            int(round(center[0] + math.cos(angle_rad) * radius)),
            int(round(center[1] + math.sin(angle_rad) * radius)),
        )
        cv2.line(overlay, center, endpoint, (255, 0, 255), 1, cv2.LINE_AA)
    if result.candidate_arc_px:
        arc = np.rint(np.asarray(result.candidate_arc_px)).astype(np.int32)
        cv2.polylines(overlay, [arc], False, (0, 255, 255), max(2, thickness), cv2.LINE_AA)
    cv2.arrowedLine(
        overlay, center, notch, (0, 220, 0), max(2, thickness), cv2.LINE_AA, tipLength=0.025
    )
    cv2.circle(overlay, center, max(5, thickness * 3), (255, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(overlay, deepest, max(4, thickness * 2), (0, 255, 0), -1, cv2.LINE_AA)
    cv2.circle(overlay, notch, max(6, thickness * 4), (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(overlay, notch, max(10, thickness * 6), (255, 255, 255), thickness, cv2.LINE_AA)
    text = (
        f"found={result.found}  notch={result.notch_angle_deg:.3f} deg  "
        f"correction={result.correction_angle_deg:+.3f} deg  "
        f"depth={result.notch_depth_px:.1f}px  conf={result.confidence:.2f}"
    )
    cv2.putText(
        overlay, text, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
        (0, 0, 0), 4, cv2.LINE_AA
    )
    cv2.putText(
        overlay, text, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
        (255, 255, 255), 1, cv2.LINE_AA
    )
    diagnostic_text = (
        f"method={result.detection_method}  edge={result.edge_support:.3f}  "
        f"circle_residual={result.circle_fit_residual_px:.2f}px"
    )
    cv2.putText(
        overlay, diagnostic_text, (24, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
        (0, 0, 0), 4, cv2.LINE_AA
    )
    cv2.putText(
        overlay, diagnostic_text, (24, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
        (255, 255, 255), 1, cv2.LINE_AA
    )
    return overlay


def make_notch_zoom(
    image: ImageInput,
    result: NotchAngleResult,
    *,
    size_px: Optional[int] = None,
    scale: float = 4.0,
) -> np.ndarray:
    """Return an enlarged annotated crop around the selected notch point."""

    source = _load_bgr(image)
    height, width = source.shape[:2]
    crop_size = int(size_px or max(80, round(result.wafer_radius_px * 0.13)))
    # Centre the crop between the outer reference and inner apex so both remain
    # visible even when the outer reference is very close to the image border.
    cx = int(round((result.notch_point_px[0] + result.notch_deepest_point_px[0]) / 2.0))
    cy = int(round((result.notch_point_px[1] + result.notch_deepest_point_px[1]) / 2.0))
    x0, x1 = max(0, cx - crop_size), min(width, cx + crop_size)
    y0, y1 = max(0, cy - crop_size), min(height, cy + crop_size)
    crop = source[y0:y1, x0:x1]
    local_result = _transform_result_for_visual(
        result,
        offset=(-float(x0), -float(y0)),
    )
    crop = make_notch_overlay(crop, local_result, thickness=2)
    return cv2.resize(
        crop, None, fx=float(scale), fy=float(scale), interpolation=cv2.INTER_NEAREST
    )


def draw_aligned_wafer_notch_guide(
    aligned_image: ImageInput,
    *,
    reference_angle_deg: float = 90.0,
    search_half_width_deg: float = 70.0,
    bg_threshold: int = 20,
    wafer_morph_kernel: int = 25,
    silhouette_open_kernel: int = 3,
    angle_samples: int = 14400,
    radial_samples: int = 200,
    min_notch_depth_px: float = 4.0,
    noise_margin_px: float = 3.0,
    min_notch_span_deg: float = 0.06,
    smooth_deg: float = 0.25,
    failure_mode: Literal["error", "zero"] = "zero",
    thickness: Optional[int] = None,
    draw_text: bool = True,
) -> AlignedNotchGuideResult:
    """Detect and draw V5 wafer/notch geometry on an already aligned image.

    This diagnostic deliberately mirrors the proven V5 approach: the largest
    non-black component is closed/opened, its minimum enclosing circle supplies
    the wafer ring, and a dense radial scan searches only the lower sector for
    an inward notch. It does not rotate the image or build a die map.

    The returned ``overlay_image`` has the same shape as the input and is a
    writable BGR copy. All returned coordinates refer to that image. The green
    line is the requested alignment reference, the red line is the measured
    notch direction, and the small orange arc between them is the residual
    angle after alignment.

    ``failure_mode="zero"`` still returns the wafer ring/search lines when the
    notch is absent. ``failure_mode="error"`` raises ``RuntimeError`` instead.
    """

    mode = str(failure_mode).strip().lower()
    if mode not in ("error", "zero"):
        raise ValueError("failure_mode must be 'error' or 'zero'.")
    if not 1.0 <= float(search_half_width_deg) <= 170.0:
        raise ValueError("search_half_width_deg must be between 1 and 170 degrees.")
    if int(angle_samples) < 720:
        raise ValueError("angle_samples must be at least 720.")
    if int(radial_samples) < 32:
        raise ValueError("radial_samples must be at least 32.")
    if float(min_notch_depth_px) < 0.0 or float(noise_margin_px) < 0.0:
        raise ValueError("notch depth thresholds must be non-negative.")

    source = _load_bgr(aligned_image)
    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    # V5 detect_wafer(): largest threshold component after a 25x25 close/open,
    # followed by minEnclosingCircle. Kernel sizes remain configurable only so
    # production images can be tuned without changing the implementation.
    _, wafer_mask = cv2.threshold(
        gray, int(bg_threshold), 255, cv2.THRESH_BINARY
    )
    morph_size = max(1, int(wafer_morph_kernel))
    if morph_size > 1:
        morph_size |= 1
        morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_size, morph_size)
        )
        wafer_mask = cv2.morphologyEx(
            wafer_mask, cv2.MORPH_CLOSE, morph_kernel
        )
        wafer_mask = cv2.morphologyEx(
            wafer_mask, cv2.MORPH_OPEN, morph_kernel
        )
    contours, _ = cv2.findContours(
        wafer_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise RuntimeError(
            "V5 wafer ring was not found. Adjust bg_threshold for this image."
        )
    wafer_contour = max(contours, key=cv2.contourArea)
    (wafer_cx, wafer_cy), wafer_radius = cv2.minEnclosingCircle(wafer_contour)
    wafer_cx = float(round(wafer_cx))
    wafer_cy = float(round(wafer_cy))
    wafer_radius = float(round(wafer_radius))
    if wafer_radius <= 0.0:
        raise RuntimeError("V5 wafer ring radius is invalid.")

    # V5 _wafer_silhouette(): optional light opening and largest connected
    # contour. Unlike the closed mask above, this keeps the notch concavity.
    _, silhouette_mask = cv2.threshold(
        gray, int(bg_threshold), 255, cv2.THRESH_BINARY
    )
    open_size = max(1, int(silhouette_open_kernel))
    if open_size >= 3:
        open_size |= 1
        open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_size, open_size)
        )
        silhouette_mask = cv2.morphologyEx(
            silhouette_mask, cv2.MORPH_OPEN, open_kernel
        )
    silhouette_contours, _ = cv2.findContours(
        silhouette_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    silhouette = np.zeros_like(gray, dtype=np.uint8)
    if silhouette_contours:
        silhouette_contour = max(silhouette_contours, key=cv2.contourArea)
        cv2.drawContours(silhouette, [silhouette_contour], -1, 1, -1)

    sample_count = int(angle_samples)
    angles = np.linspace(0.0, 2.0 * math.pi, sample_count, endpoint=False)
    radii_axis = np.linspace(
        wafer_radius * 0.93,
        wafer_radius * 1.015,
        int(radial_samples),
    )
    xs = (
        wafer_cx + radii_axis[None, :] * np.cos(angles)[:, None]
    ).astype(np.int32)
    ys = (
        wafer_cy + radii_axis[None, :] * np.sin(angles)[:, None]
    ).astype(np.int32)
    np.clip(xs, 0, width - 1, out=xs)
    np.clip(ys, 0, height - 1, out=ys)
    on_wafer = silhouette[ys, xs] > 0
    last_indices = np.where(
        on_wafer.any(axis=1),
        on_wafer.shape[1] - 1 - np.argmax(on_wafer[:, ::-1], axis=1),
        0,
    )
    boundary_radii = radii_axis[last_indices]
    depth = np.median(boundary_radii) - boundary_radii

    smooth_window = max(
        3, int(round(float(smooth_deg) / 360.0 * sample_count))
    )
    if smooth_window >= 3:
        smooth_kernel = np.ones(smooth_window, dtype=np.float64) / smooth_window
        padded = np.concatenate(
            (depth[-smooth_window:], depth, depth[:smooth_window])
        )
        depth = np.convolve(padded, smooth_kernel, mode="same")[
            smooth_window:smooth_window + sample_count
        ]

    angles_deg = np.degrees(angles)
    distance_from_reference = np.abs(
        (angles_deg - float(reference_angle_deg) + 180.0) % 360.0 - 180.0
    )
    in_sector = distance_from_reference <= float(search_half_width_deg)
    active = (depth > float(min_notch_depth_px)) & in_sector
    groups = _circular_candidate_groups(active)
    degree_step = 360.0 / sample_count
    groups = [
        group for group in groups
        if len(group) * degree_step >= float(min_notch_span_deg)
    ]
    candidate_indices = (
        max(groups, key=lambda group: float(depth[group].sum()))
        if groups else np.asarray((), dtype=np.int64)
    )
    outside_depth = depth[~in_sector]
    noise_floor = (
        float(np.percentile(outside_depth, 99.5))
        if outside_depth.size else 0.0
    )
    effective_threshold = max(
        float(min_notch_depth_px), noise_floor + float(noise_margin_px)
    )
    found = bool(
        candidate_indices.size
        and float(np.max(depth[candidate_indices])) >= effective_threshold
    )

    notch_center: Optional[Point] = None
    notch_point: Optional[Point] = None
    notch_left: Optional[Point] = None
    notch_right: Optional[Point] = None
    notch_angle: Optional[float] = None
    residual_angle = 0.0
    notch_depth = 0.0
    notch_width = 0.0
    candidate_arc: Tuple[Point, ...] = ()

    if found:
        candidate_depth = depth[candidate_indices]
        candidate_angles = angles[candidate_indices]
        weight_sum = float(candidate_depth.sum())
        notch_angle = float(
            math.degrees(
                math.atan2(
                    float((np.sin(candidate_angles) * candidate_depth).sum()),
                    float((np.cos(candidate_angles) * candidate_depth).sum()),
                )
            ) % 360.0
        )
        residual_angle = _normalise_angle(
            notch_angle - float(reference_angle_deg)
        )
        notch_radius_values = boundary_radii[candidate_indices]
        boundary_x = wafer_cx + notch_radius_values * np.cos(candidate_angles)
        boundary_y = wafer_cy + notch_radius_values * np.sin(candidate_angles)
        notch_center = (
            float((boundary_x * candidate_depth).sum() / weight_sum),
            float((boundary_y * candidate_depth).sum() / weight_sum),
        )
        notch_angle_rad = math.radians(notch_angle)
        notch_point = (
            float(wafer_cx + wafer_radius * math.cos(notch_angle_rad)),
            float(wafer_cy + wafer_radius * math.sin(notch_angle_rad)),
        )
        notch_left = (float(boundary_x[0]), float(boundary_y[0]))
        notch_right = (float(boundary_x[-1]), float(boundary_y[-1]))
        notch_depth = float(np.max(candidate_depth))
        notch_width = float(len(candidate_indices) * degree_step)
        arc_stride = max(1, len(candidate_indices) // 256)
        candidate_arc = tuple(
            (float(boundary_x[index]), float(boundary_y[index]))
            for index in range(0, len(candidate_indices), arc_stride)
        )
    elif mode == "error":
        raise RuntimeError(
            "V5 notch was not found in the aligned image: "
            f"effective_depth_threshold={effective_threshold:.2f}px, "
            f"search={float(reference_angle_deg):.1f}+/-"
            f"{float(search_half_width_deg):.1f}deg."
        )

    overlay = source.copy()
    line_width = (
        max(1, int(thickness))
        if thickness is not None
        else max(2, int(round(max(height, width) / 3000.0)))
    )
    center_int = (int(round(wafer_cx)), int(round(wafer_cy)))
    radius_int = int(round(wafer_radius))

    # Actual threshold contour (gray) and ideal V5 enclosing ring (cyan).
    cv2.drawContours(
        overlay, [wafer_contour], -1, (150, 150, 150),
        max(1, line_width // 2), cv2.LINE_AA
    )
    cv2.circle(
        overlay, center_int, radius_int, (255, 255, 0), line_width, cv2.LINE_AA
    )
    cv2.drawMarker(
        overlay, center_int, (255, 0, 0), cv2.MARKER_CROSS,
        max(12, line_width * 8), max(1, line_width), cv2.LINE_AA
    )

    def ring_point(angle_deg: float, ratio: float = 1.0) -> Tuple[int, int]:
        angle_rad = math.radians(float(angle_deg))
        return (
            int(round(wafer_cx + wafer_radius * ratio * math.cos(angle_rad))),
            int(round(wafer_cy + wafer_radius * ratio * math.sin(angle_rad))),
        )

    # Search-sector limits are magenta; aligned reference is green.
    for search_angle in (
        float(reference_angle_deg) - float(search_half_width_deg),
        float(reference_angle_deg) + float(search_half_width_deg),
    ):
        cv2.line(
            overlay, center_int, ring_point(search_angle),
            (255, 0, 255), max(1, line_width // 2), cv2.LINE_AA
        )
    reference_endpoint = ring_point(reference_angle_deg)
    cv2.arrowedLine(
        overlay, center_int, reference_endpoint, (0, 220, 0),
        line_width, cv2.LINE_AA, tipLength=0.018
    )

    if found and notch_angle is not None and notch_point is not None:
        if candidate_arc:
            arc_points = np.rint(np.asarray(candidate_arc)).astype(np.int32)
            cv2.polylines(
                overlay, [arc_points], False, (0, 255, 255),
                max(2, line_width * 2), cv2.LINE_AA
            )
        for boundary_point in (notch_left, notch_right):
            if boundary_point is not None:
                point_int = tuple(int(round(value)) for value in boundary_point)
                cv2.line(
                    overlay, center_int, point_int, (0, 165, 255),
                    max(1, line_width // 2), cv2.LINE_AA
                )
                cv2.circle(
                    overlay, point_int, max(4, line_width * 2),
                    (255, 255, 255), -1, cv2.LINE_AA
                )
        notch_endpoint = tuple(int(round(value)) for value in notch_point)
        cv2.arrowedLine(
            overlay, center_int, notch_endpoint, (0, 0, 255),
            max(2, line_width * 2), cv2.LINE_AA, tipLength=0.018
        )
        if notch_center is not None:
            cv2.circle(
                overlay,
                tuple(int(round(value)) for value in notch_center),
                max(5, line_width * 3), (0, 128, 255), -1, cv2.LINE_AA
            )
        cv2.circle(
            overlay, notch_endpoint, max(7, line_width * 4),
            (0, 0, 255), -1, cv2.LINE_AA
        )

        # Draw the signed shortest arc from the green reference to the red
        # detected direction. This is the residual angle in the aligned image.
        arc_count = max(8, int(abs(residual_angle) * 2.0) + 2)
        guide_angles = np.linspace(
            float(reference_angle_deg),
            float(reference_angle_deg) + residual_angle,
            arc_count,
        )
        guide_radius = wafer_radius * 0.24
        guide_arc = np.rint(np.column_stack((
            wafer_cx + guide_radius * np.cos(np.radians(guide_angles)),
            wafer_cy + guide_radius * np.sin(np.radians(guide_angles)),
        ))).astype(np.int32)
        cv2.polylines(
            overlay, [guide_arc], False, (0, 128, 255),
            max(2, line_width), cv2.LINE_AA
        )

    if draw_text:
        if found and notch_angle is not None:
            summary = (
                f"V5 found=True  notch={notch_angle:.4f} deg  "
                f"aligned residual={residual_angle:+.4f} deg"
            )
            detail = (
                f"center=({wafer_cx:.1f},{wafer_cy:.1f})  "
                f"radius={wafer_radius:.1f}px  depth={notch_depth:.2f}px  "
                f"width={notch_width:.3f}deg"
            )
        else:
            summary = "V5 found=False  residual=+0.0000 deg"
            detail = (
                f"center=({wafer_cx:.1f},{wafer_cy:.1f})  "
                f"radius={wafer_radius:.1f}px  "
                f"threshold={effective_threshold:.2f}px"
            )
        font_scale = max(0.55, min(1.2, max(height, width) / 9000.0))
        text_x = max(12, line_width * 6)
        text_y = max(34, line_width * 17)
        for row, text_value in enumerate((summary, detail)):
            y = text_y + row * int(round(34 * font_scale))
            cv2.putText(
                overlay, text_value, (text_x, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 0, 0), max(3, line_width * 3), cv2.LINE_AA
            )
            cv2.putText(
                overlay, text_value, (text_x, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), max(1, line_width), cv2.LINE_AA
            )

    return AlignedNotchGuideResult(
        overlay_image=overlay,
        found=found,
        wafer_center_px=(wafer_cx, wafer_cy),
        wafer_radius_px=wafer_radius,
        notch_center_px=notch_center,
        notch_point_px=notch_point,
        notch_left_px=notch_left,
        notch_right_px=notch_right,
        notch_angle_deg=notch_angle,
        reference_angle_deg=float(reference_angle_deg) % 360.0,
        residual_angle_deg=float(residual_angle),
        notch_depth_px=notch_depth,
        notch_width_deg=notch_width,
        effective_depth_threshold_px=float(effective_threshold),
        candidate_arc_px=candidate_arc,
        wafer_contour_px=wafer_contour.copy(),
        search_center_angle_deg=float(reference_angle_deg) % 360.0,
        search_half_width_deg=float(search_half_width_deg),
        detection_method="v5_silhouette_radial_aligned",
    )


def _affine_point(matrix: np.ndarray, point: Point) -> Point:
    transformed = np.asarray(matrix, dtype=np.float64) @ np.asarray(
        (float(point[0]), float(point[1]), 1.0), dtype=np.float64
    )
    return float(transformed[0]), float(transformed[1])


def _affine_pair(matrix: np.ndarray, pair) -> Tuple[Point, Point]:
    return _affine_point(matrix, pair[0]), _affine_point(matrix, pair[1])


def estimate_grid_from_yolo_notch(
    clip_image: ImageInput,
    detections: Union[str, Path, np.ndarray, Sequence[Any]],
    *,
    notch_correction_angle_deg: float,
    notch_confidence: float,
    reference_point_clip: Optional[Point] = None,
    detection_format: DetectionFormat = "auto",
    normalized: Optional[bool] = None,
    confidence_threshold: float = 0.25,
    refine: bool = True,
    refine_radius: int = 18,
    refine_mode: RefinementMode = "auto",
    refine_max_street_width: Optional[int] = None,
    refine_corner_patch_ratio: float = 0.22,
    refine_corner_reference_weight: float = 0.70,
    refine_noise_kernel: int = 5,
    refine_min_confidence: float = 0.15,
    axis_tolerance: float = 0.18,
    perpendicular_tolerance_px: float = 5.0,
) -> GridEstimate:
    """Estimate centre and pitch while taking angle only from the notch."""

    if not 0.0 <= float(refine_min_confidence) <= 1.0:
        raise ValueError("refine_min_confidence must be between 0.0 and 1.0.")
    image = _load_bgr(clip_image)
    height, width = image.shape[:2]
    points = parse_yolo_points(
        detections,
        (width, height),
        detection_format=detection_format,
        normalized=normalized,
        confidence_threshold=confidence_threshold,
    )
    if len(points) < 3:
        raise ValueError(
            f"At least three YOLO cross-points are required; received {len(points)}."
        )
    raw_points = list(points)
    refinement_confidences = [0.0] * len(points)
    if refine:
        refined_points = []
        refinement_confidences = []
        for point in raw_points:
            candidate, candidate_confidence = refine_cross_point(
                image,
                point,
                search_radius=refine_radius,
                max_street_width=refine_max_street_width,
                mode=refine_mode,
                corner_patch_ratio=refine_corner_patch_ratio,
                corner_reference_weight=refine_corner_reference_weight,
                noise_kernel=refine_noise_kernel,
            )
            value = float(candidate_confidence)
            refinement_confidences.append(value)
            refined_points.append(
                candidate if value >= float(refine_min_confidence) else point
            )
        points = refined_points

    array = np.asarray(points, dtype=np.float64)
    selection_reference = np.asarray(
        reference_point_clip
        if reference_point_clip is not None
        else (width / 2.0, height / 2.0),
        dtype=np.float64,
    ).reshape(-1)
    if selection_reference.size != 2 or not np.all(np.isfinite(selection_reference)):
        raise ValueError("reference_point_clip must contain two finite coordinates.")
    center_index = int(
        np.argmin(np.linalg.norm(array - selection_reference, axis=1))
    )
    center = array[center_index]

    # This is the essential change: axes come only from the notch correction.
    angle = math.radians(float(notch_correction_angle_deg))
    axis_x = np.asarray((math.cos(angle), math.sin(angle)), dtype=np.float64)
    axis_y = np.asarray((-math.sin(angle), math.cos(angle)), dtype=np.float64)
    delta = array - center
    delta[center_index] = 0.0
    side_index, side_vector = _select_axis_neighbour(
        delta,
        axis_x,
        axis_y,
        prefer_positive=True,
        axis_tolerance=axis_tolerance,
        perpendicular_tolerance_px=perpendicular_tolerance_px,
    )
    below_index, below_vector = _select_axis_neighbour(
        delta,
        axis_y,
        axis_x,
        prefer_positive=True,
        axis_tolerance=axis_tolerance,
        perpendicular_tolerance_px=perpendicular_tolerance_px,
    )
    if side_index == below_index:
        raise ValueError("The same YOLO point was selected for both pitch axes.")

    pitch_x = float(np.linalg.norm(side_vector))
    pitch_y = float(np.linalg.norm(below_vector))
    angle_x = _fold_grid_angle(
        math.degrees(math.atan2(float(side_vector[1]), float(side_vector[0])))
    )
    angle_y = _fold_grid_angle(
        math.degrees(math.atan2(float(-below_vector[0]), float(below_vector[1])))
    )
    return GridEstimate(
        points_clip=tuple(_point(point) for point in array),
        center_corner_clip=_point(center),
        side_corner_clip=_point(array[side_index]),
        below_corner_clip=_point(array[below_index]),
        pitch_x=pitch_x,
        pitch_y=pitch_y,
        angle_deg=float(notch_correction_angle_deg),
        angle_x_deg=float(angle_x),
        angle_y_deg=float(angle_y),
        angle_confidence=float(notch_confidence),
        refined=bool(refine),
        raw_points_clip=tuple(_point(point) for point in raw_points),
        refinement_confidences=tuple(refinement_confidences),
        refinement_mode=refine_mode if refine else "none",
        center_corner_raw_clip=_point(raw_points[center_index]),
        side_corner_raw_clip=_point(raw_points[side_index]),
        below_corner_raw_clip=_point(raw_points[below_index]),
        angle_mode="notch",
        robust_angle_deg=float(notch_correction_angle_deg),
        local_angle_deg=float((angle_x + angle_y) / 2.0),
        angle_pairs_clip=(),
        angle_pairs_raw_clip=(),
        angle_pair_axes=(),
        angle_pair_angles_deg=(),
        angle_pair_residuals_deg=(),
        angle_candidate_count=0,
    )


def build_die_map_from_yolo(
    wafer_image: ImageInput,
    clip_image: ImageInput,
    detections: Union[str, Path, np.ndarray, Sequence[Any]],
    *,
    clip_origin: Optional[Point] = None,
    detection_format: DetectionFormat = "auto",
    normalized: Optional[bool] = None,
    confidence_threshold: float = 0.25,
    refine: bool = True,
    refine_radius: int = 18,
    refine_mode: RefinementMode = "auto",
    refine_max_street_width: Optional[int] = None,
    refine_corner_patch_ratio: float = 0.22,
    refine_corner_reference_weight: float = 0.70,
    refine_noise_kernel: int = 5,
    refine_min_confidence: float = 0.15,
    pitch_size: Optional[Tuple[float, float]] = None,
    pixel_per_unit: float = 32.0,
    include_edge: bool = True,
    edge_margin: float = 1.0,
    edge_mode: str = "circle",
    notch_reference_angle_deg: float = 90.0,
    notch_max_dimension: int = 3072,
    notch_angle_samples: int = 3600,
    notch_baseline_window_deg: float = 10.0,
    notch_min_depth_px: Optional[float] = None,
    notch_min_depth_ratio: float = 0.001,
    notch_min_wide_deg: float = 2.0,
    notch_search_center_angle_deg: float = 90.0,
    notch_search_half_width_deg: float = 45.0,
    notch_wafer_center_hint_px: Optional[Point] = None,
    notch_wafer_radius_hint_px: Optional[float] = None,
    notch_failure_mode: Literal["error", "zero"] = "error",
    return_aligned_image: bool = True,
    return_notch_visuals: bool = True,
    notch_visual_max_dimension: int = 2048,
    notch_zoom_size_px: Optional[int] = 256,
    notch_zoom_scale: float = 2.0,
    alignment_interpolation: int = cv2.INTER_CUBIC,
    alignment_border_value: Tuple[int, int, int] = (0, 0, 0),
) -> WaferDieMap:
    """Build a die map with the wafer notch as the sole angle source.

    The notch detector does not require a black background. It fits the wafer
    circle from colour-gradient geometry outside the expected notch sector,
    then searches only ``notch_search_center_angle_deg +/-
    notch_search_half_width_deg`` for a continuous inward depression.

    If the automatic circle is wrong on production data, pass full-image
    ``notch_wafer_center_hint_px=(x, y)`` and
    ``notch_wafer_radius_hint_px=radius``. Use ``notch_failure_mode="error"``
    to stop on a missing notch or ``"zero"`` to keep an unrotated result.
    The returned ``dm`` and ``dm.dies`` use the rotated ``aligned_image``
    coordinate system and therefore have ``dm.grid_angle_deg == 0``. The
    applied image rotation is ``dm.image_rotation_deg``; original coordinates
    are preserved through the affine matrices and ``source_*`` fields.
    Detailed diagnostics and images are returned as ``dm.notch_result``,
    ``dm.notch_overlay_image``, and ``dm.notch_zoom_image``.
    """

    if return_notch_visuals:
        if float(notch_zoom_scale) <= 0.0:
            raise ValueError("notch_zoom_scale must be positive.")
        if notch_zoom_size_px is not None and int(notch_zoom_size_px) <= 0:
            raise ValueError("notch_zoom_size_px must be positive or None.")

    wafer = _load_bgr(wafer_image)
    clip = _load_bgr(clip_image)
    full_height, full_width = wafer.shape[:2]
    clip_height, clip_width = clip.shape[:2]
    notch = detect_wafer_notch(
        wafer,
        reference_angle_deg=notch_reference_angle_deg,
        max_dimension=notch_max_dimension,
        angle_samples=notch_angle_samples,
        baseline_window_deg=notch_baseline_window_deg,
        min_notch_depth_px=notch_min_depth_px,
        min_notch_depth_ratio=notch_min_depth_ratio,
        min_wide_notch_deg=notch_min_wide_deg,
        search_center_angle_deg=notch_search_center_angle_deg,
        search_half_width_deg=notch_search_half_width_deg,
        wafer_center_hint_px=notch_wafer_center_hint_px,
        wafer_radius_hint_px=notch_wafer_radius_hint_px,
        failure_mode=notch_failure_mode,
    )
    if clip_origin is None:
        clip_origin = (
            (full_width - clip_width) / 2.0,
            (full_height - clip_height) / 2.0,
        )
    wafer_center_clip = (
        notch.wafer_center_px[0] - float(clip_origin[0]),
        notch.wafer_center_px[1] - float(clip_origin[1]),
    )
    estimate = estimate_grid_from_yolo_notch(
        clip,
        detections,
        notch_correction_angle_deg=notch.correction_angle_deg,
        notch_confidence=notch.confidence,
        reference_point_clip=wafer_center_clip,
        detection_format=detection_format,
        normalized=normalized,
        confidence_threshold=confidence_threshold,
        refine=refine,
        refine_radius=refine_radius,
        refine_mode=refine_mode,
        refine_max_street_width=refine_max_street_width,
        refine_corner_patch_ratio=refine_corner_patch_ratio,
        refine_corner_reference_weight=refine_corner_reference_weight,
        refine_noise_kernel=refine_noise_kernel,
        refine_min_confidence=refine_min_confidence,
    )
    bx, by, bw, bh = cv2.boundingRect(notch.wafer_contour_px)
    source_boundary = WaferBoundary(
        center_px=notch.wafer_center_px,
        radius_px=notch.wafer_radius_px,
        contour_px=notch.wafer_contour_px,
        area_px=float(cv2.contourArea(notch.wafer_contour_px)),
        bbox_px=(int(bx), int(by), int(bx + bw), int(by + bh)),
        method="notch_geometry_edge_circle",
    )
    origin_full = (
        float(clip_origin[0]) + estimate.center_corner_clip[0],
        float(clip_origin[1]) + estimate.center_corner_clip[1],
    )
    if pitch_size is None:
        map_pitch_x, map_pitch_y = estimate.pitch_x, estimate.pitch_y
        pitch_source = "detected"
    else:
        pitch_values = np.asarray(pitch_size, dtype=np.float64).reshape(-1)
        if (
            pitch_values.size != 2
            or not np.all(np.isfinite(pitch_values))
            or np.any(pitch_values <= 0.0)
        ):
            raise ValueError(
                "pitch_size must be a positive finite (pitch_x, pitch_y) pair."
            )
        map_pitch_x, map_pitch_y = float(pitch_values[0]), float(pitch_values[1])
        pitch_source = "manual"

    # The notch angle rotates the image into its canonical orientation. The
    # returned DM is generated in that aligned-image coordinate system with a
    # zero grid angle; the map itself must not be rotated a second time.
    matrix = cv2.getRotationMatrix2D(
        notch.wafer_center_px, notch.correction_angle_deg, 1.0
    )
    inverse = cv2.invertAffineTransform(matrix)
    aligned_origin = _affine_point(matrix, origin_full)
    aligned_contour = cv2.transform(
        source_boundary.contour_px.astype(np.float32), matrix
    )
    aligned_contour = np.rint(aligned_contour).astype(np.int32)
    aligned_center = _affine_point(matrix, source_boundary.center_px)
    aligned_bx, aligned_by, aligned_bw, aligned_bh = cv2.boundingRect(aligned_contour)
    aligned_boundary = WaferBoundary(
        center_px=aligned_center,
        radius_px=source_boundary.radius_px,
        contour_px=aligned_contour,
        area_px=float(cv2.contourArea(aligned_contour)),
        bbox_px=(
            int(aligned_bx),
            int(aligned_by),
            int(aligned_bx + aligned_bw),
            int(aligned_by + aligned_bh),
        ),
        method="notch_aligned_geometry_edge_circle",
    )

    die_map = generate_die_map(
        aligned_boundary,
        (full_height, full_width),
        aligned_origin,
        map_pitch_x,
        map_pitch_y,
        0.0,
        pixel_per_unit=pixel_per_unit,
        include_edge=include_edge,
        edge_margin=edge_margin,
        edge_mode=edge_mode,
        angle_confidence=notch.confidence,
        grid_estimate=estimate,
    )

    def pair_to_full(pair):
        return (
            (
                float(clip_origin[0]) + float(pair[0][0]),
                float(clip_origin[1]) + float(pair[0][1]),
            ),
            (
                float(clip_origin[0]) + float(pair[1][0]),
                float(clip_origin[1]) + float(pair[1][1]),
            ),
        )

    source_pitch_x_points = pair_to_full(estimate.pitch_x_points_clip)
    source_pitch_y_points = pair_to_full(estimate.pitch_y_points_clip)
    source_pitch_x_points_raw = pair_to_full(estimate.pitch_x_points_raw_clip)
    source_pitch_y_points_raw = pair_to_full(estimate.pitch_y_points_raw_clip)
    die_map.pitch_x_points_full = _affine_pair(matrix, source_pitch_x_points)
    die_map.pitch_y_points_full = _affine_pair(matrix, source_pitch_y_points)
    die_map.pitch_x_points_raw_full = _affine_pair(matrix, source_pitch_x_points_raw)
    die_map.pitch_y_points_raw_full = _affine_pair(matrix, source_pitch_y_points_raw)
    die_map.source_pitch_x_points_full = source_pitch_x_points
    die_map.source_pitch_y_points_full = source_pitch_y_points
    die_map.source_pitch_x_points_raw_full = source_pitch_x_points_raw
    die_map.source_pitch_y_points_raw_full = source_pitch_y_points_raw
    die_map.angle_pairs_full = ()
    die_map.angle_pairs_raw_full = ()
    die_map.detected_pitch_x = float(estimate.pitch_x)
    die_map.detected_pitch_y = float(estimate.pitch_y)
    die_map.pitch_source = pitch_source
    die_map.original_to_aligned_matrix = matrix
    die_map.aligned_to_original_matrix = inverse
    die_map.coordinate_space = "aligned_image"
    die_map.source_grid_angle_deg = float(notch.correction_angle_deg)
    die_map.image_rotation_deg = float(notch.correction_angle_deg)
    die_map.source_x0 = float(origin_full[0])
    die_map.source_y0 = float(origin_full[1])
    die_map.original_wafer_boundary = source_boundary
    if return_aligned_image:
        die_map.aligned_image = cv2.warpAffine(
            wafer,
            matrix,
            (full_width, full_height),
            flags=alignment_interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=alignment_border_value,
        )
    die_map.angle_align_method = "notch" if notch.found else "notch_zero_fallback"
    die_map.notch_result = notch
    die_map.notch_point_px = notch.notch_point_px
    die_map.notch_deepest_point_px = notch.notch_deepest_point_px
    die_map.notch_angle_deg = notch.notch_angle_deg
    die_map.notch_reference_angle_deg = notch.reference_angle_deg
    die_map.notch_depth_px = notch.notch_depth_px
    die_map.notch_width_px = notch.notch_width_px
    die_map.notch_detection_method = notch.detection_method
    die_map.notch_edge_support = notch.edge_support
    die_map.notch_circle_fit_residual_px = notch.circle_fit_residual_px
    die_map.notch_search_center_angle_deg = notch.search_center_angle_deg
    die_map.notch_search_half_width_deg = notch.search_half_width_deg
    die_map.notch_correction_angle_deg = float(notch.correction_angle_deg)
    die_map.notch_point_aligned_px = _affine_point(matrix, notch.notch_point_px)
    die_map.notch_deepest_point_aligned_px = _affine_point(
        matrix, notch.notch_deepest_point_px
    )
    die_map.notch_overlay_coordinate_space = "original_image"
    if return_notch_visuals:
        die_map.notch_overlay_image = make_notch_overlay(
            wafer,
            notch,
            max_dimension=notch_visual_max_dimension,
        )
        die_map.notch_zoom_image = make_notch_zoom(
            wafer,
            notch,
            size_px=notch_zoom_size_px,
            scale=notch_zoom_scale,
        )
    return die_map


build_die_map = build_die_map_from_yolo


@dataclass(frozen=True)
class AdaptiveBackgroundNotchGuideResult:
    overlay_image: np.ndarray = field(repr=False)
    found: bool
    wafer_center_px: Point
    wafer_radius_px: float
    notch_center_px: Optional[Point]
    notch_deepest_point_px: Optional[Point]
    notch_point_px: Optional[Point]
    notch_left_px: Optional[Point]
    notch_right_px: Optional[Point]
    notch_angle_deg: Optional[float]
    reference_angle_deg: float
    residual_angle_deg: float
    notch_depth_px: float
    notch_width_deg: float
    effective_depth_threshold_px: float
    candidate_arc_px: Tuple[Point, ...] = field(repr=False)
    wafer_contour_px: np.ndarray = field(repr=False)
    background_palette_bgr: Tuple[Tuple[int, int, int], ...]
    segmentation_threshold_lab: float
    analysis_scale: float
    search_center_angle_deg: float
    search_half_width_deg: float
    detection_method: str


def _load_bgr(image: ImageInput) -> np.ndarray:
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] == 3:
            return image
        if image.ndim == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        raise ValueError(f"Unsupported image shape: {image.shape}")
    loaded = cv2.imread(str(image), cv2.IMREAD_COLOR)
    if loaded is None:
        raise FileNotFoundError(str(image))
    return loaded


def _normalise_angle(angle_deg: float) -> float:
    return float((float(angle_deg) + 180.0) % 360.0 - 180.0)


def _odd_kernel(value: float, *, minimum: int = 1) -> int:
    size = max(int(minimum), int(round(float(value))))
    return size if size % 2 else size + 1


def _border_pixels(image: np.ndarray, band_px: int) -> np.ndarray:
    height, width = image.shape[:2]
    band = max(2, min(int(band_px), height // 4, width // 4))
    return np.concatenate((
        image[:band, :, :].reshape(-1, 3),
        image[-band:, :, :].reshape(-1, 3),
        image[band:-band, :band, :].reshape(-1, 3),
        image[band:-band, -band:, :].reshape(-1, 3),
    ), axis=0)


def _learn_background_palette(
    image_bgr: np.ndarray,
    *,
    border_band_px: int,
    palette_size: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return LAB palette, BGR palette, and border residual percentile."""

    border_bgr = _border_pixels(image_bgr, border_band_px)
    if len(border_bgr) > 40000:
        step = int(math.ceil(len(border_bgr) / 40000.0))
        border_bgr = border_bgr[::step]
    border_lab = cv2.cvtColor(
        border_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB
    ).reshape(-1, 3).astype(np.float32)

    unique_count = len(np.unique(border_bgr, axis=0))
    cluster_count = max(1, min(int(palette_size), unique_count, len(border_lab)))
    if cluster_count == 1:
        palette_lab = np.median(border_lab, axis=0, keepdims=True).astype(np.float32)
    else:
        cv2.setRNGSeed(20260828)
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
            60,
            0.05,
        )
        _, _, palette_lab = cv2.kmeans(
            border_lab,
            cluster_count,
            None,
            criteria,
            3,
            cv2.KMEANS_PP_CENTERS,
        )
        palette_lab = palette_lab.astype(np.float32)

    border_min_sq = np.full(len(border_lab), np.inf, dtype=np.float32)
    for colour in palette_lab:
        delta = border_lab - colour[None, :]
        border_min_sq = np.minimum(
            border_min_sq, np.sum(delta * delta, axis=1)
        )
    border_residual = float(np.percentile(np.sqrt(border_min_sq), 99.5))

    palette_u8 = np.clip(np.rint(palette_lab), 0, 255).astype(np.uint8)
    palette_bgr = cv2.cvtColor(
        palette_u8.reshape(-1, 1, 3), cv2.COLOR_LAB2BGR
    ).reshape(-1, 3)
    return palette_lab, palette_bgr, border_residual


def _background_difference_mask(
    image_bgr: np.ndarray,
    palette_lab: np.ndarray,
    threshold_lab: float,
) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    minimum_sq = np.full(lab.shape[:2], np.inf, dtype=np.float32)
    for colour in palette_lab:
        delta = lab - colour.reshape(1, 1, 3)
        distance_sq = (
            delta[:, :, 0] * delta[:, :, 0]
            + delta[:, :, 1] * delta[:, :, 1]
            + delta[:, :, 2] * delta[:, :, 2]
        )
        np.minimum(minimum_sq, distance_sq, out=minimum_sq)
    return (minimum_sq > float(threshold_lab) ** 2).astype(np.uint8) * 255


def _candidate_groups(active: np.ndarray):
    if not np.any(active):
        return []
    if np.all(active):
        return [np.arange(len(active), dtype=np.int64)]
    starts = np.flatnonzero(active & ~np.roll(active, 1))
    groups = []
    for start_value in starts:
        start = int(start_value)
        values = [start]
        index = (start + 1) % len(active)
        while active[index] and index != start:
            values.append(index)
            index = (index + 1) % len(active)
        groups.append(np.asarray(values, dtype=np.int64))
    return groups


def _adaptive_draw_aligned_wafer_notch_guide(
    aligned_image: ImageInput,
    *,
    reference_angle_deg: float = 90.0,
    search_center_angle_deg: Optional[float] = None,
    search_half_width_deg: float = 70.0,
    max_analysis_dimension: int = 3072,
    border_band_px: int = 16,
    background_palette_size: int = 3,
    background_distance_threshold_lab: Optional[float] = None,
    background_noise_margin_lab: float = 6.0,
    min_background_distance_lab: float = 8.0,
    wafer_morph_kernel: int = 25,
    silhouette_open_kernel: int = 3,
    angle_samples: int = 14400,
    radial_samples: int = 200,
    min_notch_depth_px: Optional[float] = 4.0,
    min_notch_depth_ratio: float = 0.001,
    noise_margin_px: float = 3.0,
    min_notch_span_deg: float = 0.06,
    smooth_deg: float = 0.25,
    wafer_center_hint_px: Optional[Point] = None,
    wafer_radius_hint_px: Optional[float] = None,
    failure_mode: Literal["error", "zero"] = "zero",
    thickness: Optional[int] = None,
    draw_text: bool = True,
) -> AdaptiveBackgroundNotchGuideResult:
    """Draw V5 ring/notch geometry after learning background from the border.

    The input may be a path or a memory BGR ndarray. Returned coordinates and
    ``overlay_image`` always use the original input resolution. The image is
    not rotated. A large image is downscaled only for analysis, then all points
    are converted back before drawing.

    Use ``failure_mode="zero"`` to keep the ring/search overlay when no notch is
    found, or ``"error"`` to raise ``RuntimeError``.
    """

    mode = str(failure_mode).strip().lower()
    if mode not in ("error", "zero"):
        raise ValueError("failure_mode must be 'error' or 'zero'.")
    if not 1.0 <= float(search_half_width_deg) <= 170.0:
        raise ValueError("search_half_width_deg must be between 1 and 170 degrees.")
    if int(angle_samples) < 720 or int(radial_samples) < 32:
        raise ValueError("angle_samples >= 720 and radial_samples >= 32 are required.")
    if int(background_palette_size) < 1:
        raise ValueError("background_palette_size must be at least 1.")

    source = _load_bgr(aligned_image)
    full_height, full_width = source.shape[:2]
    analysis_limit = max(512, int(max_analysis_dimension))
    scale = min(1.0, analysis_limit / float(max(full_height, full_width)))
    if scale < 1.0:
        work = cv2.resize(
            source, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
    else:
        work = source
    height, width = work.shape[:2]

    scaled_border_band = max(2, int(round(float(border_band_px) * scale)))
    palette_lab, palette_bgr, border_residual = _learn_background_palette(
        work,
        border_band_px=scaled_border_band,
        palette_size=int(background_palette_size),
    )
    if background_distance_threshold_lab is None:
        segmentation_threshold = max(
            float(min_background_distance_lab),
            border_residual + float(background_noise_margin_lab),
        )
    else:
        segmentation_threshold = float(background_distance_threshold_lab)
    if segmentation_threshold <= 0.0:
        raise ValueError("background distance threshold must be positive.")

    raw_mask = _background_difference_mask(
        work, palette_lab, segmentation_threshold
    )
    morph_size = _odd_kernel(float(wafer_morph_kernel) * scale)
    wafer_mask = raw_mask.copy()
    if morph_size >= 3:
        morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_size, morph_size)
        )
        wafer_mask = cv2.morphologyEx(
            wafer_mask, cv2.MORPH_CLOSE, morph_kernel
        )
        wafer_mask = cv2.morphologyEx(
            wafer_mask, cv2.MORPH_OPEN, morph_kernel
        )
    contours, _ = cv2.findContours(
        wafer_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise RuntimeError(
            "Adaptive-background wafer ring was not found. "
            "Adjust background_distance_threshold_lab or border_band_px."
        )
    wafer_contour = max(contours, key=cv2.contourArea)
    (wafer_cx, wafer_cy), wafer_radius = cv2.minEnclosingCircle(wafer_contour)
    if wafer_center_hint_px is not None:
        wafer_cx = float(wafer_center_hint_px[0]) * scale
        wafer_cy = float(wafer_center_hint_px[1]) * scale
    if wafer_radius_hint_px is not None:
        wafer_radius = float(wafer_radius_hint_px) * scale
    if wafer_radius <= min(height, width) * 0.20:
        raise RuntimeError("Adaptive-background wafer radius is implausibly small.")

    open_size = _odd_kernel(float(silhouette_open_kernel) * scale)
    silhouette_mask = raw_mask.copy()
    if open_size >= 3:
        open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_size, open_size)
        )
        silhouette_mask = cv2.morphologyEx(
            silhouette_mask, cv2.MORPH_OPEN, open_kernel
        )
    silhouette_contours, _ = cv2.findContours(
        silhouette_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not silhouette_contours:
        raise RuntimeError("Adaptive-background wafer silhouette was not found.")
    silhouette_contour = max(silhouette_contours, key=cv2.contourArea)
    silhouette = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(silhouette, [silhouette_contour], -1, 1, -1)

    sample_count = int(angle_samples)
    angles = np.linspace(0.0, 2.0 * math.pi, sample_count, endpoint=False)
    radii_axis = np.linspace(
        wafer_radius * 0.93,
        wafer_radius * 1.015,
        int(radial_samples),
    )
    xs = (
        wafer_cx + radii_axis[None, :] * np.cos(angles)[:, None]
    ).astype(np.int32)
    ys = (
        wafer_cy + radii_axis[None, :] * np.sin(angles)[:, None]
    ).astype(np.int32)
    np.clip(xs, 0, width - 1, out=xs)
    np.clip(ys, 0, height - 1, out=ys)
    on_wafer = silhouette[ys, xs] > 0
    last_indices = np.where(
        on_wafer.any(axis=1),
        on_wafer.shape[1] - 1 - np.argmax(on_wafer[:, ::-1], axis=1),
        0,
    )
    boundary_radii = radii_axis[last_indices]
    depth = np.median(boundary_radii) - boundary_radii

    smooth_window = max(
        3, int(round(float(smooth_deg) / 360.0 * sample_count))
    )
    smooth_kernel = np.ones(smooth_window, dtype=np.float64) / smooth_window
    padded = np.concatenate((depth[-smooth_window:], depth, depth[:smooth_window]))
    depth = np.convolve(padded, smooth_kernel, mode="same")[
        smooth_window:smooth_window + sample_count
    ]

    angles_deg = np.degrees(angles)
    search_center = (
        float(reference_angle_deg)
        if search_center_angle_deg is None
        else float(search_center_angle_deg)
    )
    in_sector = np.abs(
        (angles_deg - search_center + 180.0) % 360.0 - 180.0
    ) <= float(search_half_width_deg)
    scaled_min_depth = (
        float(min_notch_depth_px) * scale
        if min_notch_depth_px is not None
        else max(1.25, float(wafer_radius) * float(min_notch_depth_ratio))
    )
    scaled_noise_margin = float(noise_margin_px) * scale
    active = (depth > scaled_min_depth) & in_sector
    degree_step = 360.0 / sample_count
    groups = [
        group for group in _candidate_groups(active)
        if len(group) * degree_step >= float(min_notch_span_deg)
    ]
    candidate_indices = (
        max(groups, key=lambda group: float(depth[group].sum()))
        if groups else np.asarray((), dtype=np.int64)
    )
    outside_depth = depth[~in_sector]
    noise_floor = (
        float(np.percentile(outside_depth, 99.5))
        if outside_depth.size else 0.0
    )
    effective_threshold = max(
        scaled_min_depth, noise_floor + scaled_noise_margin
    )
    found = bool(
        candidate_indices.size
        and float(np.max(depth[candidate_indices])) >= effective_threshold
    )

    notch_center_work: Optional[Point] = None
    notch_deepest_work: Optional[Point] = None
    notch_point_work: Optional[Point] = None
    notch_left_work: Optional[Point] = None
    notch_right_work: Optional[Point] = None
    notch_angle: Optional[float] = None
    residual_angle = 0.0
    notch_depth_work = 0.0
    notch_width = 0.0
    candidate_arc_work: Tuple[Point, ...] = ()

    if found:
        candidate_depth = depth[candidate_indices]
        candidate_angles = angles[candidate_indices]
        weight_sum = float(candidate_depth.sum())
        notch_angle = float(math.degrees(math.atan2(
            float((np.sin(candidate_angles) * candidate_depth).sum()),
            float((np.cos(candidate_angles) * candidate_depth).sum()),
        )) % 360.0)
        residual_angle = _normalise_angle(
            notch_angle - float(reference_angle_deg)
        )
        notch_radii = boundary_radii[candidate_indices]
        boundary_x = wafer_cx + notch_radii * np.cos(candidate_angles)
        boundary_y = wafer_cy + notch_radii * np.sin(candidate_angles)
        notch_center_work = (
            float((boundary_x * candidate_depth).sum() / weight_sum),
            float((boundary_y * candidate_depth).sum() / weight_sum),
        )
        peak_local_index = int(np.argmax(candidate_depth))
        notch_deepest_work = (
            float(boundary_x[peak_local_index]),
            float(boundary_y[peak_local_index]),
        )
        notch_angle_rad = math.radians(notch_angle)
        notch_point_work = (
            float(wafer_cx + wafer_radius * math.cos(notch_angle_rad)),
            float(wafer_cy + wafer_radius * math.sin(notch_angle_rad)),
        )
        notch_left_work = (float(boundary_x[0]), float(boundary_y[0]))
        notch_right_work = (float(boundary_x[-1]), float(boundary_y[-1]))
        notch_depth_work = float(np.max(candidate_depth))
        notch_width = float(len(candidate_indices) * degree_step)
        arc_stride = max(1, len(candidate_indices) // 256)
        candidate_arc_work = tuple(
            (float(boundary_x[index]), float(boundary_y[index]))
            for index in range(0, len(candidate_indices), arc_stride)
        )
    elif mode == "error":
        raise RuntimeError(
            "Adaptive-background V5 notch was not found: "
            f"effective_depth_threshold={effective_threshold / scale:.2f}px, "
            f"search={search_center:.1f}+/-"
            f"{float(search_half_width_deg):.1f}deg."
        )

    inv_scale = 1.0 / scale

    def full_point(point: Optional[Point]) -> Optional[Point]:
        if point is None:
            return None
        return float(point[0] * inv_scale), float(point[1] * inv_scale)

    full_center = (float(wafer_cx * inv_scale), float(wafer_cy * inv_scale))
    full_radius = float(wafer_radius * inv_scale)
    notch_center = full_point(notch_center_work)
    notch_deepest = full_point(notch_deepest_work)
    notch_point = full_point(notch_point_work)
    notch_left = full_point(notch_left_work)
    notch_right = full_point(notch_right_work)
    candidate_arc = tuple(
        (float(point[0] * inv_scale), float(point[1] * inv_scale))
        for point in candidate_arc_work
    )
    contour_full = np.rint(
        wafer_contour.astype(np.float64) * inv_scale
    ).astype(np.int32)

    overlay = source.copy()
    line_width = (
        max(1, int(thickness)) if thickness is not None
        else max(2, int(round(max(full_height, full_width) / 3000.0)))
    )
    center_int = tuple(int(round(value)) for value in full_center)
    radius_int = int(round(full_radius))

    cv2.drawContours(
        overlay, [contour_full], -1, (150, 150, 150),
        max(1, line_width // 2), cv2.LINE_AA
    )
    cv2.circle(
        overlay, center_int, radius_int, (255, 255, 0), line_width, cv2.LINE_AA
    )
    cv2.drawMarker(
        overlay, center_int, (255, 0, 0), cv2.MARKER_CROSS,
        max(12, line_width * 8), max(1, line_width), cv2.LINE_AA
    )

    def ring_point(angle_deg: float, ratio: float = 1.0) -> Tuple[int, int]:
        angle_rad = math.radians(float(angle_deg))
        return (
            int(round(full_center[0] + full_radius * ratio * math.cos(angle_rad))),
            int(round(full_center[1] + full_radius * ratio * math.sin(angle_rad))),
        )

    for search_angle in (
        search_center - float(search_half_width_deg),
        search_center + float(search_half_width_deg),
    ):
        cv2.line(
            overlay, center_int, ring_point(search_angle),
            (255, 0, 255), max(1, line_width // 2), cv2.LINE_AA
        )
    cv2.arrowedLine(
        overlay, center_int, ring_point(reference_angle_deg), (0, 220, 0),
        line_width, cv2.LINE_AA, tipLength=0.018
    )

    if found and notch_angle is not None and notch_point is not None:
        if candidate_arc:
            cv2.polylines(
                overlay,
                [np.rint(np.asarray(candidate_arc)).astype(np.int32)],
                False,
                (0, 255, 255),
                max(2, line_width * 2),
                cv2.LINE_AA,
            )
        for boundary_point in (notch_left, notch_right):
            if boundary_point is None:
                continue
            point_int = tuple(int(round(value)) for value in boundary_point)
            cv2.line(
                overlay, center_int, point_int, (0, 165, 255),
                max(1, line_width // 2), cv2.LINE_AA
            )
            cv2.circle(
                overlay, point_int, max(4, line_width * 2),
                (255, 255, 255), -1, cv2.LINE_AA
            )
        notch_int = tuple(int(round(value)) for value in notch_point)
        cv2.arrowedLine(
            overlay, center_int, notch_int, (0, 0, 255),
            max(2, line_width * 2), cv2.LINE_AA, tipLength=0.018
        )
        if notch_center is not None:
            cv2.circle(
                overlay,
                tuple(int(round(value)) for value in notch_center),
                max(5, line_width * 3),
                (0, 128, 255),
                -1,
                cv2.LINE_AA,
            )
        cv2.circle(
            overlay, notch_int, max(7, line_width * 4),
            (0, 0, 255), -1, cv2.LINE_AA
        )
        guide_angles = np.linspace(
            float(reference_angle_deg),
            float(reference_angle_deg) + residual_angle,
            max(8, int(abs(residual_angle) * 2.0) + 2),
        )
        guide_radius = full_radius * 0.24
        guide_arc = np.rint(np.column_stack((
            full_center[0] + guide_radius * np.cos(np.radians(guide_angles)),
            full_center[1] + guide_radius * np.sin(np.radians(guide_angles)),
        ))).astype(np.int32)
        cv2.polylines(
            overlay, [guide_arc], False, (0, 128, 255),
            max(2, line_width), cv2.LINE_AA
        )

    if draw_text:
        palette_text = "/".join(
            f"{int(colour[0])},{int(colour[1])},{int(colour[2])}"
            for colour in palette_bgr
        )
        if found and notch_angle is not None:
            summary = (
                f"adaptive V5 found=True  notch={notch_angle:.4f} deg  "
                f"aligned residual={residual_angle:+.4f} deg"
            )
            detail = (
                f"center=({full_center[0]:.1f},{full_center[1]:.1f})  "
                f"radius={full_radius:.1f}px  "
                f"depth={notch_depth_work * inv_scale:.2f}px  "
                f"width={notch_width:.3f}deg"
            )
        else:
            summary = "adaptive V5 found=False  residual=+0.0000 deg"
            detail = (
                f"center=({full_center[0]:.1f},{full_center[1]:.1f})  "
                f"radius={full_radius:.1f}px"
            )
        background_text = (
            f"border BGR={palette_text}  LAB threshold={segmentation_threshold:.2f}"
        )
        font_scale = max(
            0.55, min(1.2, max(full_height, full_width) / 9000.0)
        )
        text_x = max(12, line_width * 6)
        text_y = max(34, line_width * 17)
        for row, text_value in enumerate((summary, detail, background_text)):
            y = text_y + row * int(round(34 * font_scale))
            cv2.putText(
                overlay, text_value, (text_x, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 0, 0), max(3, line_width * 3), cv2.LINE_AA
            )
            cv2.putText(
                overlay, text_value, (text_x, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), max(1, line_width), cv2.LINE_AA
            )

    return AdaptiveBackgroundNotchGuideResult(
        overlay_image=overlay,
        found=found,
        wafer_center_px=full_center,
        wafer_radius_px=full_radius,
        notch_center_px=notch_center,
        notch_deepest_point_px=notch_deepest,
        notch_point_px=notch_point,
        notch_left_px=notch_left,
        notch_right_px=notch_right,
        notch_angle_deg=notch_angle,
        reference_angle_deg=float(reference_angle_deg) % 360.0,
        residual_angle_deg=float(residual_angle),
        notch_depth_px=float(notch_depth_work * inv_scale),
        notch_width_deg=float(notch_width),
        effective_depth_threshold_px=float(effective_threshold * inv_scale),
        candidate_arc_px=candidate_arc,
        wafer_contour_px=contour_full,
        background_palette_bgr=tuple(
            tuple(int(value) for value in colour) for colour in palette_bgr
        ),
        segmentation_threshold_lab=float(segmentation_threshold),
        analysis_scale=float(scale),
        search_center_angle_deg=search_center % 360.0,
        search_half_width_deg=float(search_half_width_deg),
        detection_method="v5_border_adaptive_silhouette_radial_aligned",
    )

# [SECTOR: 86_ADAPTIVE_BACKGROUND_ANGLE_OVERRIDE] ---------------------------
# Everything above is the existing full copy-paste notch/DM pipeline. Only the
# global detect_wafer_notch function is replaced below. Python resolves that
# global at call time, so build_die_map_from_yolo, locate_die, pitch, overlays,
# coordinate transforms, and return fields remain unchanged.

draw_aligned_wafer_notch_guide_adaptive = (
    _adaptive_draw_aligned_wafer_notch_guide
)
__all__.extend([
    "AdaptiveBackgroundNotchGuideResult",
    "draw_aligned_wafer_notch_guide_adaptive",
])


def detect_wafer_notch(
    image: ImageInput,
    *,
    reference_angle_deg: float = 90.0,
    max_dimension: int = 3072,
    angle_samples: int = 3600,
    baseline_window_deg: float = 10.0,
    radial_inner_ratio: float = 0.85,
    min_notch_depth_px: Optional[float] = None,
    min_notch_depth_ratio: float = 0.001,
    min_wide_notch_deg: float = 2.0,
    search_center_angle_deg: float = 90.0,
    search_half_width_deg: float = 45.0,
    wafer_center_hint_px: Optional[Point] = None,
    wafer_radius_hint_px: Optional[float] = None,
    failure_mode: Literal["error", "zero"] = "error",
    require_notch: Optional[bool] = None,
    background_palette_size: int = 3,
    background_distance_threshold_lab: Optional[float] = None,
    background_noise_margin_lab: float = 6.0,
    min_background_distance_lab: float = 8.0,
    border_band_px: int = 16,
) -> NotchAngleResult:
    """Drop-in notch detector using border-adaptive V5 geometry only.

    The signature keeps all established arguments. Compatibility-only
    ``baseline_window_deg``, ``radial_inner_ratio``, and
    ``min_wide_notch_deg`` are accepted but do not alter the V5 radial method.
    """

    mode = str(failure_mode).strip().lower()
    if require_notch is not None:
        mode = "error" if bool(require_notch) else "zero"
    if mode not in ("error", "zero"):
        raise ValueError("failure_mode must be 'error' or 'zero'.")
    _ = baseline_window_deg, radial_inner_ratio, min_wide_notch_deg

    guide = _adaptive_draw_aligned_wafer_notch_guide(
        image,
        reference_angle_deg=reference_angle_deg,
        search_center_angle_deg=search_center_angle_deg,
        search_half_width_deg=search_half_width_deg,
        max_analysis_dimension=max_dimension,
        border_band_px=border_band_px,
        background_palette_size=background_palette_size,
        background_distance_threshold_lab=background_distance_threshold_lab,
        background_noise_margin_lab=background_noise_margin_lab,
        min_background_distance_lab=min_background_distance_lab,
        angle_samples=angle_samples,
        min_notch_depth_px=min_notch_depth_px,
        min_notch_depth_ratio=min_notch_depth_ratio,
        wafer_center_hint_px=wafer_center_hint_px,
        wafer_radius_hint_px=wafer_radius_hint_px,
        failure_mode=mode,
        draw_text=False,
    )

    center = guide.wafer_center_px
    radius = float(guide.wafer_radius_px)
    if guide.found:
        notch_angle = float(guide.notch_angle_deg)
        notch_point = guide.notch_point_px
        deepest_point = (
            guide.notch_deepest_point_px
            or guide.notch_center_px
            or notch_point
        )
        correction = _normalise_angle(
            notch_angle - float(reference_angle_deg)
        )
    else:
        notch_angle = float(reference_angle_deg) % 360.0
        notch_angle_rad = math.radians(notch_angle)
        notch_point = (
            float(center[0] + radius * math.cos(notch_angle_rad)),
            float(center[1] + radius * math.sin(notch_angle_rad)),
        )
        deepest_point = notch_point
        correction = 0.0

    notch_width_px = float(
        2.0 * radius * math.sin(math.radians(guide.notch_width_deg) / 2.0)
    )
    threshold = max(0.1, float(guide.effective_depth_threshold_px))
    depth_ratio = float(guide.notch_depth_px) / threshold
    confidence = (
        float(np.clip((depth_ratio - 0.70) / 1.30, 0.0, 1.0))
        if guide.found else 0.0
    )
    contour_points = guide.wafer_contour_px.reshape(-1, 2).astype(np.float64)
    contour_radii = np.hypot(
        contour_points[:, 0] - float(center[0]),
        contour_points[:, 1] - float(center[1]),
    )
    circle_residual = float(np.median(np.abs(contour_radii - radius)))
    auto_min_depth = (
        float(min_notch_depth_px)
        if min_notch_depth_px is not None
        else max(1.25 / max(guide.analysis_scale, 1e-9), radius * float(min_notch_depth_ratio))
    )
    radial_noise = max(
        0.0, float(guide.effective_depth_threshold_px) - auto_min_depth
    )

    return NotchAngleResult(
        found=bool(guide.found),
        wafer_center_px=center,
        wafer_radius_px=radius,
        notch_point_px=notch_point,
        notch_deepest_point_px=deepest_point,
        notch_angle_deg=notch_angle,
        reference_angle_deg=float(reference_angle_deg),
        correction_angle_deg=float(correction),
        notch_depth_px=float(guide.notch_depth_px),
        notch_width_deg=float(guide.notch_width_deg),
        notch_width_px=notch_width_px,
        confidence=confidence,
        radial_noise_px=radial_noise,
        candidate_arc_px=guide.candidate_arc_px,
        wafer_contour_px=guide.wafer_contour_px,
        segmentation_threshold=float(guide.segmentation_threshold_lab),
        scale=float(guide.analysis_scale),
        failure_mode=mode,
        detection_method="v5_border_adaptive_angle_only",
        search_center_angle_deg=float(search_center_angle_deg) % 360.0,
        search_half_width_deg=float(search_half_width_deg),
        edge_support=1.0 if guide.found else 0.0,
        circle_fit_residual_px=circle_residual,
    )

