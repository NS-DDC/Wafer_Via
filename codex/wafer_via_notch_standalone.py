"""Standalone notch-aligned YOLO wafer die-map pipeline.

The detector model owns cross-point detection.  This module only converts its
512 x 512 centre-clip coordinates into a centre corner, X/Y pitch, grid angle,
wafer boundary, die map, overlays, and ``locate_die`` results.  No fixed die or
street colour is used.
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

__all__ = [
    "GridEstimate",
    "WaferBoundary",
    "WaferDieMap",
    "inspect_yolo_results",
    "parse_yolo_points",
    "refine_cross_point",
    "detect_wafer_boundary",
    "generate_die_map",
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
            except Exception as exc:
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
                item_format = str(item.get("bbox_format", "xyxy"))
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


def _fold_grid_angle(angle_deg: float) -> float:
    return (float(angle_deg) + 45.0) % 90.0 - 45.0


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
        qx, qy = float(point[0]), float(point[1])
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


def make_clip_overlay(clip_image: ImageInput, estimate: GridEstimate) -> np.ndarray:
    overlay = _load_bgr(clip_image).copy()
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
        f"A={estimate.angle_deg:.3f}deg(notch) "
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


__all__.extend([
    "AlignedNotchGuideResult",
    "NotchAngleResult",
    "detect_wafer_notch",
    "align_wafer_by_notch",
    "draw_aligned_wafer_notch_guide",
    "make_notch_overlay",
    "make_notch_zoom",
    "estimate_grid_from_yolo_notch",
    "build_die_map_from_yolo",
    "build_die_map",
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
    roi_center_px: Optional[Point] = None
    roi_bounds_px: Optional[Tuple[float, float, float, float]] = None
    semicircle_center_px: Optional[Point] = None
    semicircle_radius_px: Optional[float] = None
    semicircle_radius_x_px: Optional[float] = None
    semicircle_radius_y_px: Optional[float] = None
    semicircle_shape: str = "none"
    semicircle_score: float = 0.0
    semicircle_fit_residual_px: float = 0.0
    background_segmentation_used: bool = False
    background_palette_bgr: Tuple[Tuple[int, int, int], ...] = ()
    background_distance_threshold_lab: float = 0.0


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


def _lab_edge_strength_from_lab(lab_image: np.ndarray):
    """Return colour-transition strength from a reusable raw LAB image."""

    lab = np.asarray(lab_image)
    if lab.ndim != 3 or lab.shape[2] != 3:
        raise ValueError("lab_image must have shape (height, width, 3).")
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


def _lab_edge_strength(image_bgr: np.ndarray):
    """Return colour-transition strength without choosing either side's colour."""

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    return _lab_edge_strength_from_lab(lab)


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


@dataclass(frozen=True)
class _LocalSemicircleCandidate:
    center: Point
    radius: float
    score: float
    edge_support: float
    arc_coverage: float
    arc_points: Tuple[Point, ...]
    roi_bounds: Tuple[int, int, int, int]
    fit_residual: float = 0.0
    radius_x: Optional[float] = None
    radius_y: Optional[float] = None
    shape: str = "semicircle"


@dataclass(frozen=True)
class _RoiBackgroundGeometry:
    palette_lab: np.ndarray = field(repr=False)
    distance_threshold_lab: float
    sample_mask: np.ndarray = field(repr=False)
    background_like_mask: np.ndarray = field(repr=False)
    exterior_background_mask: np.ndarray = field(repr=False)
    wafer_mask: np.ndarray = field(repr=False)
    wafer_contour: np.ndarray = field(repr=False)
    wafer_center: Point
    wafer_radius: float
    wafer_circle_residual: float
    roi_bounds: Tuple[int, int, int, int]
    outward_unit: Point


def _robust_circle_from_points(
    points: np.ndarray,
    *,
    minimum_points: int = 30,
) -> Tuple[Point, float, float, np.ndarray]:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(values) < int(minimum_points):
        raise RuntimeError("Circle fit has insufficient boundary points.")
    center = np.median(values, axis=0)
    radius = float(np.median(np.linalg.norm(values - center, axis=1)))
    keep = np.ones(len(values), dtype=bool)
    residual = np.zeros(len(values), dtype=np.float64)
    noise = float("inf")
    for _ in range(10):
        selected = values[keep]
        design = np.column_stack((
            2.0 * selected[:, 0],
            2.0 * selected[:, 1],
            np.ones(len(selected)),
        ))
        targets = np.sum(selected * selected, axis=1)
        coefficients = np.linalg.lstsq(design, targets, rcond=None)[0]
        center = coefficients[:2]
        radius_squared = float(coefficients[2] + center @ center)
        if radius_squared <= 0.0:
            raise RuntimeError("Circle fit produced a non-positive radius.")
        radius = math.sqrt(radius_squared)
        residual = np.linalg.norm(values - center, axis=1) - radius
        median = float(np.median(residual[keep]))
        noise = float(1.4826 * np.median(np.abs(residual[keep] - median)))
        new_keep = np.abs(residual - median) <= max(1.25, 3.0 * noise)
        if int(new_keep.sum()) < int(minimum_points) or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    fit_residual = float(np.median(np.abs(residual[keep])))
    return (float(center[0]), float(center[1])), float(radius), fit_residual, keep


def _learn_background_from_notch_roi(
    image_bgr: np.ndarray,
    roi_center: Point,
    roi_half_size: Point,
    center_hint: Point,
    *,
    palette_size: int = 3,
    outer_band_fraction: float = 0.28,
    distance_threshold_lab: Optional[float] = None,
    noise_margin_lab: float = 4.0,
    morph_size_px: float = 24.0,
    lab_image: Optional[np.ndarray] = None,
) -> _RoiBackgroundGeometry:
    """Learn exterior colour in the outward ROI band and segment the wafer."""

    height, width = image_bgr.shape[:2]
    x0 = max(0, int(math.floor(roi_center[0] - roi_half_size[0])))
    y0 = max(0, int(math.floor(roi_center[1] - roi_half_size[1])))
    x1 = min(width, int(math.ceil(roi_center[0] + roi_half_size[0])) + 1)
    y1 = min(height, int(math.ceil(roi_center[1] + roi_half_size[1])) + 1)
    if x1 - x0 < 24 or y1 - y0 < 24:
        raise ValueError("notch ROI is too small or lies outside the image.")
    if not 1 <= int(palette_size) <= 8:
        raise ValueError("notch_background_palette_size must be between 1 and 8.")
    if not 0.10 <= float(outer_band_fraction) <= 0.60:
        raise ValueError("notch_background_outer_band_fraction must be in [0.10, 0.60].")

    outward = np.asarray(roi_center, dtype=np.float64) - np.asarray(
        center_hint, dtype=np.float64
    )
    outward_length = float(np.linalg.norm(outward))
    if outward_length <= 1e-6:
        outward = np.asarray((0.0, 1.0), dtype=np.float64)
    else:
        outward /= outward_length

    roi_height, roi_width = y1 - y0, x1 - x0
    local_y, local_x = np.indices((roi_height, roi_width), dtype=np.float32)
    global_x = local_x + float(x0)
    global_y = local_y + float(y0)
    projection = (
        (global_x - float(roi_center[0])) * float(outward[0])
        + (global_y - float(roi_center[1])) * float(outward[1])
    )
    quantile = 1.0 - float(outer_band_fraction)
    projection_threshold = float(np.quantile(projection, quantile))
    sample_local = projection >= projection_threshold
    sample_mask = np.zeros((height, width), dtype=np.uint8)
    sample_mask[y0:y1, x0:x1] = sample_local.astype(np.uint8) * 255

    if lab_image is None:
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    else:
        lab = np.asarray(lab_image)
        if lab.shape != image_bgr.shape:
            raise ValueError("lab_image shape must match image_bgr shape.")
        if lab.ndim != 3 or lab.shape[2] != 3:
            raise ValueError("lab_image must have shape (height, width, 3).")
    lab = lab.astype(np.float32, copy=False)
    samples = lab[y0:y1, x0:x1][sample_local].reshape(-1, 3)
    if len(samples) < 64:
        raise RuntimeError("The outward notch ROI band has too few background pixels.")
    stride = max(1, len(samples) // 30000)
    samples_for_fit = samples[::stride].astype(np.float32)
    distinct = np.unique(samples_for_fit.astype(np.uint8), axis=0)
    cluster_count = min(int(palette_size), len(distinct), len(samples_for_fit))
    if cluster_count <= 1:
        palette = np.median(samples_for_fit, axis=0, keepdims=True).astype(np.float32)
        labels = np.zeros((len(samples_for_fit), 1), dtype=np.int32)
    else:
        cv2.setRNGSeed(1907)
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            60,
            0.20,
        )
        _, labels, palette = cv2.kmeans(
            samples_for_fit,
            cluster_count,
            None,
            criteria,
            5,
            cv2.KMEANS_PP_CENTERS,
        )
    assigned = palette[labels.reshape(-1)]
    sample_residual = np.linalg.norm(samples_for_fit - assigned, axis=1)
    automatic_threshold = max(
        8.0,
        float(np.percentile(sample_residual, 98.0)) + float(noise_margin_lab),
    )
    threshold = (
        automatic_threshold
        if distance_threshold_lab is None
        else float(distance_threshold_lab)
    )
    if threshold <= 0.0:
        raise ValueError("notch_background_distance_threshold_lab must be positive.")

    nearest_distance = np.full((height, width), np.inf, dtype=np.float32)
    for colour in palette:
        delta = lab - colour.reshape(1, 1, 3)
        distance = np.sqrt(np.sum(delta * delta, axis=2)).astype(np.float32)
        np.minimum(nearest_distance, distance, out=nearest_distance)
    background_like = (nearest_distance <= threshold).astype(np.uint8) * 255
    morph_size = max(3, int(round(float(morph_size_px))) | 1)
    morph_size = min(morph_size, max(3, (min(height, width) // 12) | 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_size, morph_size))
    background_like = cv2.morphologyEx(
        background_like, cv2.MORPH_CLOSE, kernel
    )

    component_count, components, stats, _ = cv2.connectedComponentsWithStats(
        (background_like > 0).astype(np.uint8), 8
    )
    border_labels = np.unique(np.concatenate((
        components[0, :],
        components[-1, :],
        components[:, 0],
        components[:, -1],
    )))
    border_labels = border_labels[border_labels > 0]
    if not len(border_labels):
        raise RuntimeError("ROI background colour did not connect to the image border.")
    exterior_label = int(max(
        border_labels, key=lambda label: int(stats[int(label), cv2.CC_STAT_AREA])
    ))
    exterior_background = (components == exterior_label).astype(np.uint8) * 255

    foreground = (exterior_background == 0).astype(np.uint8)
    open_size = max(3, int(round(float(morph_size_px) * 0.45)) | 1)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, open_kernel)
    fg_count, fg_components, fg_stats, fg_centroids = cv2.connectedComponentsWithStats(
        foreground, 8
    )
    center_x = int(np.clip(round(center_hint[0]), 0, width - 1))
    center_y = int(np.clip(round(center_hint[1]), 0, height - 1))
    wafer_label = int(fg_components[center_y, center_x])
    if wafer_label <= 0:
        candidates = []
        for label in range(1, fg_count):
            area = int(fg_stats[label, cv2.CC_STAT_AREA])
            centroid = fg_centroids[label]
            distance_to_hint = float(np.linalg.norm(centroid - np.asarray(center_hint)))
            candidates.append((area / max(1.0, 1.0 + distance_to_hint), label))
        if not candidates:
            raise RuntimeError("Wafer component was not found after ROI background segmentation.")
        wafer_label = int(max(candidates)[1])
    wafer_mask = (fg_components == wafer_label).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        wafer_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        raise RuntimeError("Wafer contour was not found after ROI background segmentation.")
    wafer_contour = max(contours, key=cv2.contourArea)
    contour_values = wafer_contour.reshape(-1, 2)
    border_clear = (
        (contour_values[:, 0] > 1)
        & (contour_values[:, 0] < width - 2)
        & (contour_values[:, 1] > 1)
        & (contour_values[:, 1] < height - 2)
    )
    fit_points = contour_values[border_clear]
    if len(fit_points) > 6000:
        fit_points = fit_points[::max(1, len(fit_points) // 6000)]
    wafer_center, wafer_radius, circle_residual, _ = _robust_circle_from_points(
        fit_points, minimum_points=100
    )
    if wafer_radius <= min(height, width) * 0.20:
        raise RuntimeError("Background-segmented wafer radius is implausibly small.")

    return _RoiBackgroundGeometry(
        palette_lab=palette.astype(np.float32),
        distance_threshold_lab=float(threshold),
        sample_mask=sample_mask,
        background_like_mask=background_like,
        exterior_background_mask=exterior_background,
        wafer_mask=wafer_mask,
        wafer_contour=wafer_contour,
        wafer_center=wafer_center,
        wafer_radius=float(wafer_radius),
        wafer_circle_residual=float(circle_residual),
        roi_bounds=(x0, y0, x1, y1),
        outward_unit=(float(outward[0]), float(outward[1])),
    )


def _sample_fitted_arc_support(
    edge: np.ndarray,
    arc_points: np.ndarray,
) -> Tuple[float, float, float]:
    """Measure local edge support and left/right balance along a fitted arc."""

    height, width = edge.shape
    values = []
    for point in np.asarray(arc_points, dtype=np.float64).reshape(-1, 2):
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        x0, x1 = max(0, x - 2), min(width, x + 3)
        y0, y1 = max(0, y - 2), min(height, y + 3)
        values.append(
            0.0 if x0 >= x1 or y0 >= y1 else float(np.max(edge[y0:y1, x0:x1]))
        )
    support = np.asarray(values, dtype=np.float64)
    if not len(support):
        return 0.0, 0.0, 0.0
    edge_support = float(np.mean(support))
    support_floor = max(0.08, float(np.percentile(support, 35.0)) * 0.70)
    coverage = float(np.mean(support >= support_floor))
    midpoint = len(support) // 2
    left = float(np.mean(support[:midpoint])) if midpoint else edge_support
    right = (
        float(np.mean(support[midpoint + 1 :]))
        if midpoint + 1 < len(support) else edge_support
    )
    symmetry = min(left, right) / max(1e-6, max(left, right))
    return edge_support, coverage, float(symmetry)


def _fit_semiellipse_from_background_boundary(
    geometry: _RoiBackgroundGeometry,
    roi_center: Point,
    roi_half_size: Point,
    radius_range: Optional[Tuple[float, float]],
) -> Optional[_LocalSemicircleCandidate]:
    """Fit a shallow/wide semi-ellipse to the exterior intrusion boundary."""

    contours, _ = cv2.findContours(
        geometry.exterior_background_mask,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        return None
    points = np.concatenate([contour.reshape(-1, 2) for contour in contours], axis=0)
    x0, y0, x1, y1 = geometry.roi_bounds
    in_roi = (
        (points[:, 0] >= x0)
        & (points[:, 0] < x1)
        & (points[:, 1] >= y0)
        & (points[:, 1] < y1)
    )
    points = points[in_roi].astype(np.float64)
    if len(points) < 24:
        return None

    wafer_center = np.asarray(geometry.wafer_center, dtype=np.float64)
    outward = np.asarray(geometry.outward_unit, dtype=np.float64)
    outward /= max(1e-9, float(np.linalg.norm(outward)))
    tangent = np.asarray((-outward[1], outward[0]), dtype=np.float64)
    vectors = points - wafer_center
    radial_distance = np.linalg.norm(vectors, axis=1)
    depth = float(geometry.wafer_radius) - radial_distance
    tangential = vectors @ tangent
    outward_projection = vectors @ outward
    minimum_half_size = min(float(roi_half_size[0]), float(roi_half_size[1]))
    if radius_range is None:
        min_half_width = max(3.0, minimum_half_size * 0.035)
        max_half_width = max(min_half_width + 2.0, minimum_half_size * 0.55)
    else:
        min_half_width, max_half_width = (
            float(radius_range[0]), float(radius_range[1])
        )
        if min_half_width <= 0.0 or max_half_width <= min_half_width:
            raise ValueError(
                "notch_semicircle_radius_range_px must be (positive_min, larger_max)."
            )

    noise_floor = max(0.75, float(geometry.wafer_circle_residual) * 1.35)
    usable = (
        (depth >= noise_floor)
        & (depth <= max(4.0, minimum_half_size * 0.75))
        & (outward_projection >= float(geometry.wafer_radius) - 2.2 * minimum_half_size)
        & (np.abs(tangential - float(np.dot(np.asarray(roi_center) - wafer_center, tangent)))
           <= float(roi_half_size[0]) * 1.10)
    )
    tangential = tangential[usable]
    depth = depth[usable]
    if len(depth) < 24:
        return None


    bins = np.rint(tangential).astype(np.int32)
    unique_bins = np.unique(bins)
    fitted_t = []
    fitted_d = []
    for value in unique_bins:
        selected_depth = depth[bins == value]
        fitted_t.append(float(value))
        fitted_d.append(float(np.max(selected_depth)))
    fitted_t = np.asarray(fitted_t, dtype=np.float64)
    fitted_d = np.asarray(fitted_d, dtype=np.float64)
    if len(fitted_t) < 18:
        return None

    order = np.argsort(fitted_t)
    fitted_t, fitted_d = fitted_t[order], fitted_d[order]
    expected_t = float(np.dot(np.asarray(roi_center) - wafer_center, tangent))
    central = np.abs(fitted_t - expected_t) <= float(roi_half_size[0]) * 0.75
    if not np.any(central):
        return None
    local_peak = float(np.max(fitted_d[central]))
    strong_threshold = max(noise_floor * 1.55, local_peak * 0.12)
    strong_indices = np.flatnonzero(central & (fitted_d >= strong_threshold))
    if len(strong_indices) < 8:
        return None
    split_at = np.flatnonzero(np.diff(fitted_t[strong_indices]) > 3.5) + 1
    groups = np.split(strong_indices, split_at)
    groups = [group for group in groups if len(group) >= 8]
    if not groups:
        return None
    group = max(
        groups,
        key=lambda values: float(np.sum(fitted_d[values]))
        * math.exp(
            -0.5
            * (
                (float(np.mean(fitted_t[values])) - expected_t)
                / max(3.0, float(roi_half_size[0]) * 0.30)
            )
            ** 2
        ),
    )
    lower = float(fitted_t[group[0]] - 4.0)
    upper = float(fitted_t[group[-1]] + 4.0)
    selected_group = (fitted_t >= lower) & (fitted_t <= upper)
    fitted_t, fitted_d = fitted_t[selected_group], fitted_d[selected_group]
    if len(fitted_t) < 18:
        return None


    origin_t = float(np.median(fitted_t))
    x = fitted_t - origin_t
    keep = np.ones(len(x), dtype=bool)
    half_width = depth_axis = center_offset = fit_residual = 0.0
    for _ in range(10):
        if int(keep.sum()) < 18:
            return None
        design = np.column_stack((x[keep] * x[keep], x[keep], np.ones(int(keep.sum()))))
        coefficients = np.linalg.lstsq(
            design, fitted_d[keep] * fitted_d[keep], rcond=None
        )[0]
        quadratic, linear, constant = (float(value) for value in coefficients)
        if quadratic >= -1e-8:
            return None
        center_local = -linear / (2.0 * quadratic)
        depth_squared = constant - quadratic * center_local * center_local
        if depth_squared <= 0.0:
            return None
        depth_axis = math.sqrt(depth_squared)
        half_width_squared = -depth_squared / quadratic
        if half_width_squared <= 0.0:
            return None
        half_width = math.sqrt(half_width_squared)
        center_offset = origin_t + center_local
        normalized = (fitted_t - center_offset) / max(half_width, 1e-6)
        predicted = depth_axis * np.sqrt(
            np.maximum(0.0, 1.0 - normalized * normalized)
        )
        residual = fitted_d - predicted
        valid_span = np.abs(normalized) <= 1.08
        current = keep & valid_span
        if int(current.sum()) < 18:
            return None
        median = float(np.median(residual[current]))
        noise = float(
            1.4826 * np.median(np.abs(residual[current] - median))
        )
        new_keep = valid_span & (
            np.abs(residual - median) <= max(1.15, 2.8 * noise)
        )
        if int(new_keep.sum()) < 18:
            return None
        fit_residual = float(np.median(np.abs(residual[new_keep])))
        if np.array_equal(new_keep, keep):
            keep = new_keep
            break
        keep = new_keep

    if not (min_half_width * 0.70 <= half_width <= max_half_width * 1.25):
        return None
    aspect = depth_axis / max(half_width, 1e-6)
    if not 0.12 <= aspect <= 1.60:
        return None

    direction_vector = outward * float(geometry.wafer_radius) + tangent * center_offset
    direction_length = float(np.linalg.norm(direction_vector))
    if direction_length <= 1e-6:
        return None
    direction = direction_vector / direction_length
    arc_tangent = np.asarray((-direction[1], direction[0]), dtype=np.float64)
    inward = -direction
    baseline_center = wafer_center + direction * float(geometry.wafer_radius)
    unit = np.linspace(-1.0, 1.0, 97, dtype=np.float64)
    arc_values = (
        baseline_center[None, :]
        + arc_tangent[None, :] * (unit * half_width)[:, None]
        + inward[None, :] * (
            depth_axis * np.sqrt(np.maximum(0.0, 1.0 - unit * unit))
        )[:, None]
    )
    boundary_edge = cv2.morphologyEx(
        geometry.exterior_background_mask,
        cv2.MORPH_GRADIENT,
        np.ones((3, 3), np.uint8),
    ).astype(np.float32) / 255.0
    edge_support, arc_coverage, symmetry = _sample_fitted_arc_support(
        boundary_edge, arc_values
    )
    apex = baseline_center + inward * depth_axis
    center_distance = float(np.linalg.norm(apex - np.asarray(roi_center)))
    center_prior = math.exp(
        -0.5 * (center_distance / max(4.0, minimum_half_size * 0.28)) ** 2
    )
    fit_quality = math.exp(
        -fit_residual / max(1.0, depth_axis * 0.10)
    )
    score = float(np.clip(
        0.26 * edge_support
        + 0.18 * arc_coverage
        + 0.10 * symmetry
        + 0.18 * center_prior
        + 0.28 * fit_quality,
        0.0,
        1.0,
    ))
    stride = max(1, len(arc_values) // 48)
    return _LocalSemicircleCandidate(
        center=(float(baseline_center[0]), float(baseline_center[1])),
        radius=float(half_width),
        score=score,
        edge_support=float(edge_support),
        arc_coverage=float(arc_coverage),
        arc_points=tuple(
            (float(point[0]), float(point[1])) for point in arc_values[::stride]
        ),
        roi_bounds=geometry.roi_bounds,
        fit_residual=float(fit_residual),
        radius_x=float(half_width),
        radius_y=float(depth_axis),
        shape="semiellipse",
    )


def _fit_circle_from_background_boundary(
    geometry: _RoiBackgroundGeometry,
    roi_center: Point,
    roi_half_size: Point,
    radius_range: Optional[Tuple[float, float]],
) -> Optional[_LocalSemicircleCandidate]:
    """Fit the exterior-background intrusion contour around the expected notch."""

    contours, _ = cv2.findContours(
        geometry.exterior_background_mask,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        return None
    points = np.concatenate([contour.reshape(-1, 2) for contour in contours], axis=0)
    x0, y0, x1, y1 = geometry.roi_bounds
    in_roi = (
        (points[:, 0] >= x0)
        & (points[:, 0] < x1)
        & (points[:, 1] >= y0)
        & (points[:, 1] < y1)
    )
    points = points[in_roi]
    if len(points) < 20:
        return None

    minimum_half_size = min(float(roi_half_size[0]), float(roi_half_size[1]))
    if radius_range is None:
        min_radius = max(3.0, minimum_half_size * 0.035)
        max_radius = max(min_radius + 2.0, minimum_half_size * 0.55)
    else:
        min_radius, max_radius = float(radius_range[0]), float(radius_range[1])
    relative = points.astype(np.float64) - np.asarray(roi_center, dtype=np.float64)
    distances = np.linalg.norm(relative, axis=1)
    inward = -np.asarray(geometry.outward_unit, dtype=np.float64)
    inward_projection = relative @ inward
    usable = (
        (distances >= min_radius)
        & (distances <= max_radius)
        & (inward_projection >= -max(2.0, min_radius * 0.25))
    )
    points = points[usable]
    distances = distances[usable]
    if len(points) < 20:
        return None

    bin_width = max(1.0, minimum_half_size / 240.0)
    bins = np.arange(min_radius, max_radius + 2.0 * bin_width, bin_width)
    histogram, edges = np.histogram(distances, bins=bins)
    if not np.any(histogram):
        return None
    smooth_histogram = cv2.GaussianBlur(
        histogram.astype(np.float32).reshape(1, -1), (5, 1), 0
    ).reshape(-1)
    peak_indices = np.argsort(smooth_histogram)[::-1][:12]
    boundary_edge = cv2.morphologyEx(
        geometry.exterior_background_mask,
        cv2.MORPH_GRADIENT,
        np.ones((3, 3), np.uint8),
    ).astype(np.float32) / 255.0
    best: Optional[_LocalSemicircleCandidate] = None
    for peak_index in peak_indices:
        peak_radius = float((edges[peak_index] + edges[peak_index + 1]) * 0.5)
        band = max(2.0, peak_radius * 0.08)
        selected = points[np.abs(distances - peak_radius) <= band]
        if len(selected) < 20:
            continue
        try:
            center, radius, fit_residual, _ = _robust_circle_from_points(
                selected, minimum_points=18
            )
        except RuntimeError:
            continue
        center_distance = float(np.linalg.norm(
            np.asarray(center) - np.asarray(roi_center)
        ))
        if center_distance > minimum_half_size * 0.45:
            continue
        if not min_radius * 0.70 <= radius <= max_radius * 1.20:
            continue
        inward_angle = math.atan2(
            geometry.wafer_center[1] - center[1],
            geometry.wafer_center[0] - center[0],
        )
        edge_support, arc_coverage, symmetry, arc_points = _sample_semicircle_support(
            boundary_edge, center, radius, inward_angle
        )
        center_prior = math.exp(
            -0.5 * (center_distance / max(4.0, minimum_half_size * 0.20)) ** 2
        )
        fit_quality = math.exp(
            -fit_residual / max(1.25, radius * 0.055)
        )
        score = float(np.clip(
            0.24 * edge_support
            + 0.16 * arc_coverage
            + 0.10 * symmetry
            + 0.25 * center_prior
            + 0.25 * fit_quality,
            0.0,
            1.0,
        ))
        candidate = _LocalSemicircleCandidate(
            center=center,
            radius=radius,
            score=score,
            edge_support=edge_support,
            arc_coverage=arc_coverage,
            arc_points=arc_points,
            roi_bounds=geometry.roi_bounds,
            fit_residual=fit_residual,
        )
        if best is None or candidate.score > best.score:
            best = candidate
    return best


def _fit_semicircle_from_background_boundary(
    geometry: _RoiBackgroundGeometry,
    roi_center: Point,
    roi_half_size: Point,
    radius_range: Optional[Tuple[float, float]],
) -> Optional[_LocalSemicircleCandidate]:
    """Fit a semi-ellipse first, then retain the historical circle fallback."""

    candidate = _fit_semiellipse_from_background_boundary(
        geometry, roi_center, roi_half_size, radius_range
    )
    if candidate is not None:
        return candidate
    return _fit_circle_from_background_boundary(
        geometry, roi_center, roi_half_size, radius_range
    )


def _normalise_roi_half_size(
    value: Union[float, Tuple[float, float]],
    *,
    scale: float,
) -> Point:
    if isinstance(value, (int, float, np.integer, np.floating)):
        half_width = half_height = float(value)
    else:
        if len(value) != 2:
            raise ValueError("notch_roi_half_size_px must be a number or (half_width, half_height).")
        half_width, half_height = float(value[0]), float(value[1])
    if half_width <= 0.0 or half_height <= 0.0:
        raise ValueError("notch_roi_half_size_px values must be positive.")
    return half_width * float(scale), half_height * float(scale)


def _sample_semicircle_support(
    edge: np.ndarray,
    center: Point,
    radius: float,
    inward_angle_rad: float,
) -> Tuple[float, float, float, Tuple[Point, ...]]:
    """Measure the inward-facing half of a local U-shaped notch circle."""

    arc_angles = np.linspace(
        float(inward_angle_rad) - math.radians(100.0),
        float(inward_angle_rad) + math.radians(100.0),
        161,
        dtype=np.float64,
    )
    band = max(1.0, min(5.0, float(radius) * 0.10))
    radial_offsets = np.linspace(-band, band, 7, dtype=np.float64)
    radii = np.maximum(1.0, float(radius) + radial_offsets)
    sampled = _polar_sample(edge, center, radii, arc_angles)
    supported = sampled.max(axis=1)
    edge_support = float(np.mean(supported))
    support_floor = max(0.08, float(np.percentile(supported, 35.0)) * 0.70)
    arc_coverage = float(np.mean(supported >= support_floor))
    midpoint = len(supported) // 2
    left_support = float(np.mean(supported[:midpoint]))
    right_support = float(np.mean(supported[midpoint + 1:]))
    symmetry = min(left_support, right_support) / max(
        1e-6, max(left_support, right_support)
    )
    stride = max(1, len(arc_angles) // 48)
    arc_points = tuple(
        (
            float(center[0] + math.cos(float(angle)) * float(radius)),
            float(center[1] + math.sin(float(angle)) * float(radius)),
        )
        for angle in arc_angles[::stride]
    )
    return edge_support, arc_coverage, float(symmetry), arc_points


def _refine_semicircle_candidate(
    edge: np.ndarray,
    wafer_center: Point,
    candidate: _LocalSemicircleCandidate,
) -> _LocalSemicircleCandidate:
    """Robustly re-fit the actual inward arc after coarse Hough detection."""

    initial_center = np.asarray(candidate.center, dtype=np.float64)
    initial_radius = float(candidate.radius)
    inward_angle = math.atan2(
        float(wafer_center[1]) - float(initial_center[1]),
        float(wafer_center[0]) - float(initial_center[0]),
    )
    arc_angles = np.linspace(
        inward_angle - math.radians(100.0),
        inward_angle + math.radians(100.0),
        241,
        dtype=np.float64,
    )
    radii = np.linspace(
        max(2.0, initial_radius * 0.65), initial_radius * 1.35, 101
    )
    polar = _polar_sample(edge, candidate.center, radii, arc_angles)
    indices = np.argmax(polar, axis=1)
    support = polar[np.arange(len(arc_angles)), indices]
    selected_radii = radii[indices]
    support_floor = max(0.08, float(np.percentile(support, 35.0)))
    keep = (
        (support >= support_floor)
        & (np.abs(selected_radii - initial_radius) <= initial_radius * 0.30)
    )
    points = np.column_stack((
        initial_center[0] + np.cos(arc_angles) * selected_radii,
        initial_center[1] + np.sin(arc_angles) * selected_radii,
    ))
    if int(keep.sum()) < 30:
        return candidate

    fitted_center = initial_center.copy()
    fitted_radius = initial_radius
    residual = np.zeros(len(points), dtype=np.float64)
    for _ in range(8):
        selected = points[keep]
        weights = np.clip(support[keep], 0.03, 1.0)
        design = np.column_stack((
            2.0 * selected[:, 0],
            2.0 * selected[:, 1],
            np.ones(len(selected)),
        ))
        values = np.sum(selected * selected, axis=1)
        root_weights = np.sqrt(weights)
        coefficients = np.linalg.lstsq(
            design * root_weights[:, None], values * root_weights, rcond=None
        )[0]
        fitted_center = coefficients[:2]
        radius_squared = float(coefficients[2] + fitted_center @ fitted_center)
        if radius_squared <= 0.0:
            return candidate
        fitted_radius = math.sqrt(radius_squared)
        residual = np.linalg.norm(points - fitted_center, axis=1) - fitted_radius
        median = float(np.median(residual[keep]))
        noise = float(1.4826 * np.median(np.abs(residual[keep] - median)))
        refined_keep = keep & (np.abs(residual - median) <= max(1.25, 2.5 * noise))
        if int(refined_keep.sum()) < 30 or np.array_equal(refined_keep, keep):
            break
        keep = refined_keep

    center_shift = float(np.linalg.norm(fitted_center - initial_center))
    fit_residual = float(np.median(np.abs(residual[keep])))
    reliable = bool(
        center_shift <= initial_radius * 0.45
        and initial_radius * 0.55 <= fitted_radius <= initial_radius * 1.55
        and fit_residual <= max(1.5, fitted_radius * 0.06)
    )
    if not reliable:
        return replace(candidate, fit_residual=fit_residual)

    refined_inward_angle = math.atan2(
        float(wafer_center[1]) - float(fitted_center[1]),
        float(wafer_center[0]) - float(fitted_center[0]),
    )
    edge_support, arc_coverage, _, arc_points = _sample_semicircle_support(
        edge,
        (float(fitted_center[0]), float(fitted_center[1])),
        fitted_radius,
        refined_inward_angle,
    )
    return replace(
        candidate,
        center=(float(fitted_center[0]), float(fitted_center[1])),
        radius=float(fitted_radius),
        edge_support=float(edge_support),
        arc_coverage=float(arc_coverage),
        arc_points=arc_points,
        fit_residual=fit_residual,
    )


def _detect_semicircle_in_roi(
    edge: np.ndarray,
    wafer_center: Point,
    wafer_radius: float,
    roi_center: Point,
    roi_half_size: Point,
    radius_range: Optional[Tuple[float, float]],
) -> Optional[_LocalSemicircleCandidate]:
    """Find a small inward-facing semicircle only inside a user ROI."""

    height, width = edge.shape
    x0 = max(0, int(math.floor(float(roi_center[0]) - float(roi_half_size[0]))))
    y0 = max(0, int(math.floor(float(roi_center[1]) - float(roi_half_size[1]))))
    x1 = min(width, int(math.ceil(float(roi_center[0]) + float(roi_half_size[0]))) + 1)
    y1 = min(height, int(math.ceil(float(roi_center[1]) + float(roi_half_size[1]))) + 1)
    if x1 - x0 < 24 or y1 - y0 < 24:
        raise ValueError("notch ROI is too small or lies outside the image.")

    minimum_half_size = min(float(roi_half_size[0]), float(roi_half_size[1]))
    if radius_range is None:
        min_radius = max(3.0, minimum_half_size * 0.035)
        max_radius = max(min_radius + 2.0, minimum_half_size * 0.55)
    else:
        min_radius, max_radius = float(radius_range[0]), float(radius_range[1])
        if min_radius <= 0.0 or max_radius <= min_radius:
            raise ValueError(
                "notch_semicircle_radius_range_px must be (positive_min, larger_max)."
            )

    crop = np.clip(edge[y0:y1, x0:x1] * 255.0, 0.0, 255.0).astype(np.uint8)
    crop = cv2.GaussianBlur(crop, (5, 5), 0)
    circles = cv2.HoughCircles(
        crop,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=max(4.0, min_radius * 0.70),
        param1=40.0,
        param2=max(6.0, min(crop.shape) * 0.018),
        minRadius=max(2, int(math.floor(min_radius))),
        maxRadius=max(3, int(math.ceil(max_radius))),
    )
    if circles is None:
        return None

    wafer_center_array = np.asarray(wafer_center, dtype=np.float64)
    roi_center_array = np.asarray(roi_center, dtype=np.float64)


    roi_scale = max(4.0, 0.20 * minimum_half_size)
    candidate_pool: List[_LocalSemicircleCandidate] = []
    for local_x, local_y, candidate_radius in circles[0]:
        center = np.asarray(
            (float(local_x) + float(x0), float(local_y) + float(y0)),
            dtype=np.float64,
        )
        radius = float(candidate_radius)
        center_radius = float(np.linalg.norm(center - wafer_center_array))
        ring_error = abs(center_radius - float(wafer_radius))
        if ring_error > max(radius * 2.0, minimum_half_size * 0.20):
            continue

        inward_angle = math.atan2(
            float(wafer_center[1]) - float(center[1]),
            float(wafer_center[0]) - float(center[0]),
        )
        edge_support, arc_coverage, symmetry, arc_points = _sample_semicircle_support(
            edge, (float(center[0]), float(center[1])), radius, inward_angle
        )
        hint_distance = float(np.linalg.norm(center - roi_center_array))
        center_prior = math.exp(-0.5 * (hint_distance / roi_scale) ** 2)
        ring_prior = math.exp(
            -0.5 * (ring_error / max(2.0, radius * 0.90)) ** 2
        )
        score = float(np.clip(
            0.32 * edge_support
            + 0.16 * arc_coverage
            + 0.12 * symmetry
            + 0.32 * center_prior
            + 0.08 * ring_prior,
            0.0,
            1.0,
        ))
        candidate = _LocalSemicircleCandidate(
            center=(float(center[0]), float(center[1])),
            radius=radius,
            score=score,
            edge_support=edge_support,
            arc_coverage=arc_coverage,
            arc_points=arc_points,
            roi_bounds=(x0, y0, x1, y1),
        )
        candidate_pool.append(candidate)
    if not candidate_pool:
        return None


    best: Optional[_LocalSemicircleCandidate] = None
    for coarse in sorted(
        candidate_pool, key=lambda item: item.score, reverse=True
    )[:20]:
        refined = _refine_semicircle_candidate(edge, wafer_center, coarse)
        fit_quality = (
            math.exp(
                -float(refined.fit_residual)
                / max(1.5, float(refined.radius) * 0.06)
            )
            if refined.fit_residual > 0.0
            else 0.0
        )
        refined = replace(
            refined,
            score=float(np.clip(0.80 * coarse.score + 0.20 * fit_quality, 0.0, 1.0)),
        )
        if best is None or refined.score > best.score:
            best = refined
    return best


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
    notch_roi_center_px: Optional[Point] = None,
    notch_roi_half_size_px: Union[float, Tuple[float, float]] = 600.0,
    notch_semicircle_radius_range_px: Optional[Tuple[float, float]] = None,
    notch_semicircle_min_score: float = 0.55,
    notch_use_roi_background: bool = True,
    notch_background_palette_size: int = 3,
    notch_background_outer_band_fraction: float = 0.28,
    notch_background_distance_threshold_lab: Optional[float] = None,
    notch_background_noise_margin_lab: float = 4.0,
    notch_background_morph_px: float = 24.0,
    failure_mode: Literal["error", "zero"] = "error",
    require_notch: Optional[bool] = None,
) -> NotchAngleResult:
    """Find a local inward deviation of the wafer's geometric outer edge.

    ``notch_angle_deg`` uses image coordinates (right=0, down=90). The returned
    ``correction_angle_deg`` is suitable for ``cv2.getRotationMatrix2D`` and
    moves the detected notch to ``reference_angle_deg``.

    Without an ROI, LAB colour-gradient magnitude supplies edge evidence and
    no foreground/background colour is assumed. With
    ``notch_roi_center_px=(x, y)`` the default ROI mode learns the wafer-exterior
    background palette from the outward part of that crop. Only the
    border-connected background is retained; its boundary supplies both a
    noise-resistant wafer silhouette and the local inward semicircle or
    wide/shallow semi-ellipse.
    ``notch_roi_half_size_px`` is a scalar or ``(half_width, half_height)`` in
    full-resolution pixels. ``notch_semicircle_radius_range_px`` and
    ``notch_background_morph_px`` are also full-resolution pixel values.

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
    if not 0.0 <= float(notch_semicircle_min_score) <= 1.0:
        raise ValueError("notch_semicircle_min_score must be between 0 and 1.")
    if float(notch_background_noise_margin_lab) < 0.0:
        raise ValueError("notch_background_noise_margin_lab must be non-negative.")
    if float(notch_background_morph_px) <= 0.0:
        raise ValueError("notch_background_morph_px must be positive.")

    source = _load_bgr(image)
    full_height, full_width = source.shape[:2]
    scale = min(1.0, float(max_dimension) / max(full_height, full_width))
    if scale < 1.0:
        work = cv2.resize(source, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        work = source


    work_lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB)
    edge, edge_normaliser = _lab_edge_strength_from_lab(work_lab)
    work_height, work_width = work.shape[:2]

    angle_samples = max(720, int(angle_samples))
    angles = np.arange(angle_samples, dtype=np.float64) * (
        2.0 * math.pi / angle_samples
    )
    angles_deg = np.degrees(angles)
    if wafer_center_hint_px is None:
        center = (work_width / 2.0, work_height / 2.0)
    else:
        center = (
            float(wafer_center_hint_px[0]) * scale,
            float(wafer_center_hint_px[1]) * scale,
        )
    roi_center: Optional[Point] = None
    roi_half_size: Optional[Point] = None
    semicircle_radius_range: Optional[Tuple[float, float]] = None
    background_geometry: Optional[_RoiBackgroundGeometry] = None
    effective_search_center_angle_deg = float(search_center_angle_deg) % 360.0
    if notch_roi_center_px is not None:
        roi_center = (
            float(notch_roi_center_px[0]) * scale,
            float(notch_roi_center_px[1]) * scale,
        )
        if not (
            -0.05 * work_width <= roi_center[0] <= 1.05 * work_width
            and -0.05 * work_height <= roi_center[1] <= 1.05 * work_height
        ):
            raise ValueError("notch_roi_center_px lies outside the input image.")
        roi_half_size = _normalise_roi_half_size(
            notch_roi_half_size_px, scale=scale
        )
        effective_search_center_angle_deg = float(
            math.degrees(
                math.atan2(roi_center[1] - center[1], roi_center[0] - center[0])
            )
            % 360.0
        )
        if notch_semicircle_radius_range_px is not None:
            semicircle_radius_range = (
                float(notch_semicircle_radius_range_px[0]) * scale,
                float(notch_semicircle_radius_range_px[1]) * scale,
            )
        if bool(notch_use_roi_background):
            background_geometry = _learn_background_from_notch_roi(
                work,
                roi_center,
                roi_half_size,
                center,
                palette_size=notch_background_palette_size,
                outer_band_fraction=notch_background_outer_band_fraction,
                distance_threshold_lab=notch_background_distance_threshold_lab,
                noise_margin_lab=notch_background_noise_margin_lab,
                morph_size_px=float(notch_background_morph_px) * scale,
                lab_image=work_lab,
            )
            if wafer_center_hint_px is None:
                center = background_geometry.wafer_center
    del work_lab
    search_distance = _angle_distance_deg(
        angles_deg, effective_search_center_angle_deg
    )
    search_mask = search_distance <= float(search_half_width_deg)
    fit_mask = search_distance >= float(search_half_width_deg) + 5.0

    if wafer_radius_hint_px is None:
        if background_geometry is not None:
            radius = float(background_geometry.wafer_radius)
        else:
            radius, _ = _initial_outer_radius(edge, center, angles)
    else:
        radius = float(wafer_radius_hint_px) * scale
    if radius <= min(work_height, work_width) * 0.20:
        raise RuntimeError("Estimated wafer radius is implausibly small.")

    circle_fit_noise = (
        float(background_geometry.wafer_circle_residual)
        if background_geometry is not None else float("inf")
    )


    for _ in range(0 if background_geometry is not None else 4):
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

    if roi_center is not None:
        effective_search_center_angle_deg = float(
            math.degrees(math.atan2(roi_center[1] - cy, roi_center[0] - cx))
            % 360.0
        )
        search_distance = _angle_distance_deg(
            angles_deg, effective_search_center_angle_deg
        )
        search_mask = search_distance <= float(search_half_width_deg)

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


    _ = baseline_window_deg
    depth_limit = (
        float(min_notch_depth_px) * scale
        if min_notch_depth_px is not None
        else max(1.25, float(radius) * float(min_notch_depth_ratio))
    )
    candidate_threshold = max(
        0.50, depth_limit * 0.40, 2.5 * radial_noise
    )


    active = search_mask & (deficit >= candidate_threshold)


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
    detection_method = "geometry_edge_bottom_sector"
    local_arc: Optional[Tuple[Point, ...]] = None
    semicircle_candidate: Optional[_LocalSemicircleCandidate] = None
    if roi_center is not None and roi_half_size is not None:
        if background_geometry is not None:
            semicircle_candidate = _fit_semicircle_from_background_boundary(
                background_geometry,
                roi_center,
                roi_half_size,
                semicircle_radius_range,
            )
            detection_method = "roi_background_connected_notch_arc"
        else:
            semicircle_candidate = _detect_semicircle_in_roi(
                edge,
                center,
                radius,
                roi_center,
                roi_half_size,
                semicircle_radius_range,
            )
            detection_method = "geometry_edge_manual_roi_semicircle"
        if semicircle_candidate is None:
            found = False
            confidence = 0.0
            candidate_support = 0.0
            peak_depth = 0.0
            notch_width_deg = 0.0
            notch_width_px = 0.0
            candidate_indices = np.asarray((), dtype=np.int64)
        else:
            local_arc = semicircle_candidate.arc_points
            semicircle_center = np.asarray(
                semicircle_candidate.center, dtype=np.float64
            )
            wafer_center_array = np.asarray(center, dtype=np.float64)
            outward_vector = semicircle_center - wafer_center_array
            outward_length = float(np.linalg.norm(outward_vector))
            if outward_length <= 1e-6:
                found = False
                outward_unit = np.asarray((0.0, 1.0), dtype=np.float64)
            else:
                outward_unit = outward_vector / outward_length
            inward_unit = -outward_unit
            notch_half_width = float(
                semicircle_candidate.radius_x
                if semicircle_candidate.radius_x is not None
                else semicircle_candidate.radius
            )
            notch_height = float(
                semicircle_candidate.radius_y
                if semicircle_candidate.radius_y is not None
                else semicircle_candidate.radius
            )
            deepest = semicircle_center + inward_unit * float(
                notch_height
            )
            notch_angle_rad = float(
                math.atan2(float(outward_unit[1]), float(outward_unit[0]))
                % (2.0 * math.pi)
            )
            notch_angle_deg = float(math.degrees(notch_angle_rad) % 360.0)
            notch_deepest_point = (float(deepest[0]), float(deepest[1]))
            notch_point = (
                float(cx + math.cos(notch_angle_rad) * radius),
                float(cy + math.sin(notch_angle_rad) * radius),
            )
            peak_depth = max(
                0.0,
                float(radius)
                - float(np.linalg.norm(deepest - wafer_center_array)),
            )
            notch_width_px = float(2.0 * notch_half_width)
            notch_width_deg = float(math.degrees(
                2.0 * math.asin(
                    min(1.0, notch_half_width / max(radius, 1e-6))
                )
            ))
            candidate_support = float(semicircle_candidate.edge_support)
            found = bool(
                outward_length > 1e-6
                and semicircle_candidate.score >= float(notch_semicircle_min_score)
                and semicircle_candidate.arc_coverage >= 0.55
                and peak_depth >= max(0.60, depth_limit * 0.40)
            )
            confidence = (
                float(semicircle_candidate.score) if found else 0.0
            )
            candidate_indices = np.asarray((), dtype=np.int64)
    if not found:
        confidence = 0.0
        if mode == "error":
            roi_message = ""
            if roi_center is not None:
                roi_message = (
                    f" roi_center=({roi_center[0] / scale:.1f},"
                    f"{roi_center[1] / scale:.1f}),"
                    f" semicircle_score={0.0 if semicircle_candidate is None else semicircle_candidate.score:.3f}."
                )
            raise RuntimeError(
                f"Wafer notch was not found: peak_depth={peak_depth / scale:.2f}px, "
                f"width={notch_width_deg:.2f}deg, required_depth={depth_limit / scale:.2f}px. "
                f"search={effective_search_center_angle_deg:.1f}+/-{float(search_half_width_deg):.1f}deg."
                f"{roi_message} Use failure_mode='zero' to return angle 0, or correct the ROI/wafer hints."
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
    if local_arc is not None:
        arc = tuple(
            (float(point[0] * inv_scale), float(point[1] * inv_scale))
            for point in local_arc
        )
    else:
        arc = tuple(
            (
                float((cx + math.cos(angles[index]) * boundary[index]) * inv_scale),
                float((cy + math.sin(angles[index]) * boundary[index]) * inv_scale),
            )
            for index in candidate_indices[::max(1, len(candidate_indices) // 48)]
        )
    if background_geometry is not None:
        contour_full = np.rint(
            background_geometry.wafer_contour.astype(np.float64) * inv_scale
        ).astype(np.int32)
    else:
        contour_stride = max(1, angle_samples // 1440)
        contour_indices = np.arange(0, angle_samples, contour_stride, dtype=np.int64)
        contour_points = np.column_stack((
            cx + np.cos(angles[contour_indices]) * boundary[contour_indices],
            cy + np.sin(angles[contour_indices]) * boundary[contour_indices],
        ))
        contour_full = np.rint(contour_points * inv_scale).astype(np.int32).reshape(-1, 1, 2)
    if background_geometry is None:
        palette_bgr: Tuple[Tuple[int, int, int], ...] = ()
        background_threshold = 0.0
    else:
        palette_lab_u8 = np.clip(
            np.rint(background_geometry.palette_lab), 0, 255
        ).astype(np.uint8).reshape(1, -1, 3)
        converted_palette = cv2.cvtColor(palette_lab_u8, cv2.COLOR_LAB2BGR).reshape(-1, 3)
        palette_bgr = tuple(
            tuple(int(value) for value in colour) for colour in converted_palette
        )
        background_threshold = float(background_geometry.distance_threshold_lab)
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
        detection_method=detection_method,
        search_center_angle_deg=effective_search_center_angle_deg,
        search_half_width_deg=float(search_half_width_deg),
        edge_support=float(candidate_support),
        circle_fit_residual_px=float(circle_fit_noise * inv_scale),
        roi_center_px=(
            None
            if roi_center is None
            else (float(roi_center[0] * inv_scale), float(roi_center[1] * inv_scale))
        ),
        roi_bounds_px=(
            None
            if semicircle_candidate is None
            else tuple(float(value * inv_scale) for value in semicircle_candidate.roi_bounds)
        ),
        semicircle_center_px=(
            None
            if semicircle_candidate is None
            else (
                float(semicircle_candidate.center[0] * inv_scale),
                float(semicircle_candidate.center[1] * inv_scale),
            )
        ),
        semicircle_radius_px=(
            None
            if semicircle_candidate is None
            else float(semicircle_candidate.radius * inv_scale)
        ),
        semicircle_radius_x_px=(
            None
            if semicircle_candidate is None
            else float(
                (
                    semicircle_candidate.radius_x
                    if semicircle_candidate.radius_x is not None
                    else semicircle_candidate.radius
                )
                * inv_scale
            )
        ),
        semicircle_radius_y_px=(
            None
            if semicircle_candidate is None
            else float(
                (
                    semicircle_candidate.radius_y
                    if semicircle_candidate.radius_y is not None
                    else semicircle_candidate.radius
                )
                * inv_scale
            )
        ),
        semicircle_shape=(
            "none" if semicircle_candidate is None else semicircle_candidate.shape
        ),
        semicircle_score=(
            0.0 if semicircle_candidate is None else float(semicircle_candidate.score)
        ),
        semicircle_fit_residual_px=(
            0.0
            if semicircle_candidate is None
            else float(semicircle_candidate.fit_residual * inv_scale)
        ),
        background_segmentation_used=background_geometry is not None,
        background_palette_bgr=palette_bgr,
        background_distance_threshold_lab=background_threshold,
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

    def transform_optional_point(point: Optional[Point]) -> Optional[Point]:
        return None if point is None else transform_point(point)

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
        roi_center_px=transform_optional_point(result.roi_center_px),
        roi_bounds_px=(
            None
            if result.roi_bounds_px is None
            else (
                float(result.roi_bounds_px[0]) * float(scale) + float(offset[0]),
                float(result.roi_bounds_px[1]) * float(scale) + float(offset[1]),
                float(result.roi_bounds_px[2]) * float(scale) + float(offset[0]),
                float(result.roi_bounds_px[3]) * float(scale) + float(offset[1]),
            )
        ),
        semicircle_center_px=transform_optional_point(result.semicircle_center_px),
        semicircle_radius_px=(
            None
            if result.semicircle_radius_px is None
            else float(result.semicircle_radius_px) * float(scale)
        ),
        semicircle_radius_x_px=(
            None
            if result.semicircle_radius_x_px is None
            else float(result.semicircle_radius_x_px) * float(scale)
        ),
        semicircle_radius_y_px=(
            None
            if result.semicircle_radius_y_px is None
            else float(result.semicircle_radius_y_px) * float(scale)
        ),
        semicircle_fit_residual_px=(
            float(result.semicircle_fit_residual_px) * float(scale)
        ),
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
    if result.roi_bounds_px is not None:
        x0, y0, x1, y1 = (int(round(value)) for value in result.roi_bounds_px)
        cv2.rectangle(
            overlay, (x0, y0), (x1, y1), (255, 0, 255),
            max(2, thickness), cv2.LINE_AA
        )
    if result.roi_center_px is not None:
        roi_center = tuple(int(round(value)) for value in result.roi_center_px)
        cv2.drawMarker(
            overlay, roi_center, (255, 0, 255), cv2.MARKER_CROSS,
            max(12, thickness * 8), max(2, thickness), cv2.LINE_AA
        )
    if result.semicircle_center_px is not None and result.semicircle_radius_px is not None:
        local_center = tuple(
            int(round(value)) for value in result.semicircle_center_px
        )
        if result.candidate_arc_px:
            fitted_arc = np.rint(
                np.asarray(result.candidate_arc_px)
            ).astype(np.int32)
            cv2.polylines(
                overlay, [fitted_arc], False, (255, 180, 0),
                max(2, thickness), cv2.LINE_AA
            )
        cv2.circle(
            overlay, local_center, max(4, thickness * 2),
            (255, 180, 0), -1, cv2.LINE_AA
        )
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
        f"circle_residual={result.circle_fit_residual_px:.2f}px  "
        f"arc={result.semicircle_shape}:{result.semicircle_score:.3f}  "
        f"arc_fit={result.semicircle_fit_residual_px:.2f}px"
    )
    cv2.putText(
        overlay, diagnostic_text, (24, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
        (0, 0, 0), 4, cv2.LINE_AA
    )
    cv2.putText(
        overlay, diagnostic_text, (24, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
        (255, 255, 255), 1, cv2.LINE_AA
    )
    if result.background_segmentation_used:
        palette_text = "/".join(
            f"{colour[0]},{colour[1]},{colour[2]}"
            for colour in result.background_palette_bgr
        )
        background_text = (
            f"ROI exterior BGR={palette_text}  "
            f"LAB distance<={result.background_distance_threshold_lab:.1f}"
        )
        cv2.putText(
            overlay, background_text, (24, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
            (0, 0, 0), 4, cv2.LINE_AA
        )
        cv2.putText(
            overlay, background_text, (24, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
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


def make_notch_background_debug_contact_sheet(
    image: ImageInput,
    *,
    notch_roi_center_px: Point,
    notch_roi_half_size_px: Union[float, Tuple[float, float]] = 600.0,
    notch_semicircle_radius_range_px: Optional[Tuple[float, float]] = None,
    wafer_center_hint_px: Optional[Point] = None,
    wafer_radius_hint_px: Optional[float] = None,
    max_dimension: int = 1536,
    background_palette_size: int = 3,
    background_outer_band_fraction: float = 0.28,
    background_distance_threshold_lab: Optional[float] = None,
    background_noise_margin_lab: float = 4.0,
    background_morph_px: float = 24.0,
) -> np.ndarray:
    """Return six labelled stages of the ROI-background notch pipeline."""

    source = _load_bgr(image)
    full_height, full_width = source.shape[:2]
    analysis_scale = min(1.0, float(max_dimension) / max(full_height, full_width))
    if analysis_scale < 1.0:
        work = cv2.resize(
            source, None, fx=analysis_scale, fy=analysis_scale,
            interpolation=cv2.INTER_AREA
        )
    else:
        work = source
    work_height, work_width = work.shape[:2]
    roi_center = (
        float(notch_roi_center_px[0]) * analysis_scale,
        float(notch_roi_center_px[1]) * analysis_scale,
    )
    roi_half_size = _normalise_roi_half_size(
        notch_roi_half_size_px, scale=analysis_scale
    )
    if wafer_center_hint_px is None:
        center_hint = (work_width / 2.0, work_height / 2.0)
    else:
        center_hint = (
            float(wafer_center_hint_px[0]) * analysis_scale,
            float(wafer_center_hint_px[1]) * analysis_scale,
        )
    radius_range = (
        None
        if notch_semicircle_radius_range_px is None
        else (
            float(notch_semicircle_radius_range_px[0]) * analysis_scale,
            float(notch_semicircle_radius_range_px[1]) * analysis_scale,
        )
    )
    geometry = _learn_background_from_notch_roi(
        work,
        roi_center,
        roi_half_size,
        center_hint,
        palette_size=background_palette_size,
        outer_band_fraction=background_outer_band_fraction,
        distance_threshold_lab=background_distance_threshold_lab,
        noise_margin_lab=background_noise_margin_lab,
        morph_size_px=float(background_morph_px) * analysis_scale,
    )
    candidate = _fit_semicircle_from_background_boundary(
        geometry, roi_center, roi_half_size, radius_range
    )
    result = detect_wafer_notch(
        work,
        max_dimension=max_dimension,
        wafer_center_hint_px=(
            None if wafer_center_hint_px is None else center_hint
        ),
        wafer_radius_hint_px=(
            None
            if wafer_radius_hint_px is None
            else float(wafer_radius_hint_px) * analysis_scale
        ),
        notch_roi_center_px=roi_center,
        notch_roi_half_size_px=roi_half_size,
        notch_semicircle_radius_range_px=radius_range,
        notch_background_palette_size=background_palette_size,
        notch_background_outer_band_fraction=background_outer_band_fraction,
        notch_background_distance_threshold_lab=background_distance_threshold_lab,
        notch_background_noise_margin_lab=background_noise_margin_lab,
        notch_background_morph_px=float(background_morph_px) * analysis_scale,
        failure_mode="zero",
    )
    final_overlay = make_notch_overlay(work, result)

    x0, y0, x1, y1 = geometry.roi_bounds
    roi_source = work[y0:y1, x0:x1].copy()
    sample_local = geometry.sample_mask[y0:y1, x0:x1] > 0
    tint = np.full_like(roi_source, (255, 0, 255))
    roi_source[sample_local] = cv2.addWeighted(
        roi_source[sample_local], 0.45, tint[sample_local], 0.55, 0.0
    )

    lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB).astype(np.float32)
    nearest_distance = np.full((work_height, work_width), np.inf, dtype=np.float32)
    for colour in geometry.palette_lab:
        delta = lab - colour.reshape(1, 1, 3)
        np.minimum(
            nearest_distance,
            np.sqrt(np.sum(delta * delta, axis=2)).astype(np.float32),
            out=nearest_distance,
        )
    distance_roi = nearest_distance[y0:y1, x0:x1]
    distance_u8 = np.clip(
        distance_roi / max(1.0, geometry.distance_threshold_lab * 2.0) * 255.0,
        0.0,
        255.0,
    ).astype(np.uint8)
    distance_colour = cv2.applyColorMap(distance_u8, cv2.COLORMAP_TURBO)
    background_like = cv2.cvtColor(
        geometry.background_like_mask[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR
    )
    exterior = cv2.cvtColor(
        geometry.exterior_background_mask[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR
    )

    silhouette = cv2.cvtColor(geometry.wafer_mask, cv2.COLOR_GRAY2BGR)
    circle_center = tuple(int(round(value)) for value in geometry.wafer_center)
    cv2.circle(
        silhouette, circle_center, int(round(geometry.wafer_radius)),
        (255, 255, 0), 3, cv2.LINE_AA
    )
    cv2.rectangle(silhouette, (x0, y0), (x1, y1), (255, 0, 255), 3, cv2.LINE_AA)
    final_roi = final_overlay[y0:y1, x0:x1]
    if candidate is not None:
        local_center = (
            int(round(candidate.center[0] - x0)),
            int(round(candidate.center[1] - y0)),
        )
        cv2.drawMarker(
            final_roi, local_center, (255, 180, 0), cv2.MARKER_CROSS,
            18, 2, cv2.LINE_AA
        )

    panel_width, panel_height = 440, 320

    def panel(image_value: np.ndarray, label: str) -> np.ndarray:
        canvas = np.full((panel_height, panel_width, 3), 24, dtype=np.uint8)
        available_height = panel_height - 40
        resize_scale = min(
            panel_width / max(1, image_value.shape[1]),
            available_height / max(1, image_value.shape[0]),
        )
        resized = cv2.resize(
            image_value,
            None,
            fx=resize_scale,
            fy=resize_scale,
            interpolation=cv2.INTER_AREA if resize_scale < 1.0 else cv2.INTER_NEAREST,
        )
        left = (panel_width - resized.shape[1]) // 2
        top = 34 + (available_height - resized.shape[0]) // 2
        canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
        cv2.putText(
            canvas, label, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
            (245, 245, 245), 1, cv2.LINE_AA
        )
        return canvas

    panels = [
        panel(roi_source, "1  ROI: outward background sample"),
        panel(distance_colour, "2  LAB distance to background"),
        panel(background_like, "3  Background-like mask"),
        panel(exterior, "4  Border-connected background"),
        panel(silhouette, "5  Wafer mask + robust circle"),
        panel(final_roi, "6  Notch arc + final angle"),
    ]
    return np.vstack((np.hstack(panels[:3]), np.hstack(panels[3:])))


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
    notch_roi_center_px: Optional[Point] = None,
    notch_roi_half_size_px: Union[float, Tuple[float, float]] = 600.0,
    notch_semicircle_radius_range_px: Optional[Tuple[float, float]] = None,
    notch_semicircle_min_score: float = 0.55,
    notch_use_roi_background: bool = True,
    notch_background_palette_size: int = 3,
    notch_background_outer_band_fraction: float = 0.28,
    notch_background_distance_threshold_lab: Optional[float] = None,
    notch_background_noise_margin_lab: float = 4.0,
    notch_background_morph_px: float = 24.0,
    notch_failure_mode: Literal["error", "zero"] = "error",
    return_aligned_image: bool = True,
    return_notch_visuals: bool = False,
    notch_visual_max_dimension: int = 5000,
    notch_zoom_size_px: Optional[int] = 256,
    notch_zoom_scale: float = 2.0,
    alignment_interpolation: int = cv2.INTER_CUBIC,
    alignment_border_value: Tuple[int, int, int] = (0, 0, 0),
) -> WaferDieMap:
    """Build a die map with the wafer notch as the sole angle source.

    The notch detector does not require a black background. It fits the wafer
    circle from colour-gradient geometry outside the expected notch sector.
    Set ``notch_roi_center_px=(x, y)`` to constrain the final angle source to
    the local inward semicircle or semi-ellipse inside that full-image ROI.

    If the automatic circle is wrong on production data, pass full-image
    ``notch_wafer_center_hint_px=(x, y)`` and
    ``notch_wafer_radius_hint_px=radius``. Use ``notch_failure_mode="error"``
    to stop on a missing notch or ``"zero"`` to keep an unrotated result.
    The returned ``dm`` and ``dm.dies`` use the rotated ``aligned_image``
    coordinate system and therefore have ``dm.grid_angle_deg == 0``. The
    applied image rotation is ``dm.image_rotation_deg``; original coordinates
    are preserved through the affine matrices and ``source_*`` fields.
    By default the only generated result image is ``dm.aligned_image``.
    Numeric notch diagnostics remain in ``dm.notch_result`` without drawing
    any image. Set ``return_notch_visuals=True`` only while debugging to also
    create ``dm.notch_overlay_image`` and ``dm.notch_zoom_image``.
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
        notch_roi_center_px=notch_roi_center_px,
        notch_roi_half_size_px=notch_roi_half_size_px,
        notch_semicircle_radius_range_px=notch_semicircle_radius_range_px,
        notch_semicircle_min_score=notch_semicircle_min_score,
        notch_use_roi_background=notch_use_roi_background,
        notch_background_palette_size=notch_background_palette_size,
        notch_background_outer_band_fraction=notch_background_outer_band_fraction,
        notch_background_distance_threshold_lab=notch_background_distance_threshold_lab,
        notch_background_noise_margin_lab=notch_background_noise_margin_lab,
        notch_background_morph_px=notch_background_morph_px,
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
    die_map.notch_roi_center_px = notch.roi_center_px
    die_map.notch_roi_bounds_px = notch.roi_bounds_px
    die_map.notch_semicircle_center_px = notch.semicircle_center_px
    die_map.notch_semicircle_radius_px = notch.semicircle_radius_px
    die_map.notch_semicircle_radius_x_px = notch.semicircle_radius_x_px
    die_map.notch_semicircle_radius_y_px = notch.semicircle_radius_y_px
    die_map.notch_semicircle_shape = notch.semicircle_shape
    die_map.notch_semicircle_score = notch.semicircle_score
    die_map.notch_semicircle_fit_residual_px = notch.semicircle_fit_residual_px
    die_map.notch_background_segmentation_used = notch.background_segmentation_used
    die_map.notch_background_palette_bgr = notch.background_palette_bgr
    die_map.notch_background_distance_threshold_lab = (
        notch.background_distance_threshold_lab
    )
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
