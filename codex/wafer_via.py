"""YOLO cross-points to a wafer die map.

The detector model owns cross-point detection.  This module only converts its
512 x 512 centre-clip coordinates into a centre corner, X/Y pitch, grid angle,
wafer boundary, die map, overlays, and ``locate_die`` results.  No fixed die or
street colour is used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import cv2
import numpy as np


ImageInput = Union[str, Path, np.ndarray]
Point = Tuple[float, float]
DetectionFormat = Literal[
    "auto", "point", "point_conf", "xyxy", "xywh", "yolo_txt", "xyxy_conf_class"
]

__all__ = [
    "GridEstimate",
    "WaferBoundary",
    "WaferDieMap",
    "parse_yolo_points",
    "refine_cross_point",
    "estimate_grid_from_yolo",
    "detect_wafer_boundary",
    "generate_die_map",
    "build_die_map_from_yolo",
    "build_die_map",
    "locate_die",
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


def refine_cross_point(
    clip_image: ImageInput,
    approximate_point: Point,
    *,
    search_radius: int = 18,
    max_street_width: Optional[int] = None,
) -> Tuple[Point, float]:
    """Optionally refine a model point to the centre of the crossing streets.

    The two street centres are midpoints of paired Lab-gradient boundaries.
    Median projection suppresses sparse marks/noise and no hue is hard-coded.
    """

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
    confidence = math.sqrt(confidence_x * confidence_y)
    if confidence <= 0.0:
        return (ax, ay), 0.0
    return (float(x1 + local_x), float(y1 + local_y)), float(confidence)


# [SECTOR: 30_GRID_ESTIMATION] ------------------------------------------------
def _fold_grid_angle(angle_deg: float) -> float:
    return (float(angle_deg) + 45.0) % 90.0 - 45.0


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    sorted_values, sorted_weights = values[order], weights[order]
    index = int(np.searchsorted(np.cumsum(sorted_weights), sorted_weights.sum() * 0.5))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def _estimate_grid_orientation(points: np.ndarray, max_rotation_deg: float) -> float:
    angles: List[float] = []
    weights: List[float] = []
    neighbour_count = min(6, len(points) - 1)
    for index, point in enumerate(points):
        delta = points - point
        distances = np.linalg.norm(delta, axis=1)
        neighbours = np.argsort(distances)[1:neighbour_count + 1]
        for neighbour in neighbours:
            if distances[neighbour] < 3.0:
                continue
            angle = _fold_grid_angle(math.degrees(math.atan2(delta[neighbour, 1], delta[neighbour, 0])))
            if abs(angle) <= max_rotation_deg:
                angles.append(angle)
                weights.append(1.0 / distances[neighbour])
    if not angles:
        raise ValueError("No horizontal/vertical YOLO point pairs were found within max_rotation_deg.")
    values = np.asarray(angles)
    weight_array = np.asarray(weights)
    first = _weighted_median(values, weight_array)
    keep = np.abs(values - first) <= 4.0
    if not np.any(keep):
        return first
    return _weighted_median(values[keep], weight_array[keep])


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
    detection_format: DetectionFormat = "auto",
    normalized: Optional[bool] = None,
    confidence_threshold: float = 0.25,
    refine: bool = False,
    refine_radius: int = 18,
    max_rotation_deg: float = 20.0,
    axis_tolerance: float = 0.18,
    perpendicular_tolerance_px: float = 5.0,
    max_axis_disagreement_deg: float = 3.0,
    strict: bool = True,
) -> GridEstimate:
    """Select the centre, adjacent side, and adjacent below cross-points."""

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
    if refine:
        points = [refine_cross_point(image, point, search_radius=refine_radius)[0] for point in points]
    array = np.asarray(points, dtype=np.float64)
    clip_center = np.array((width / 2.0, height / 2.0), dtype=np.float64)
    center_index = int(np.argmin(np.linalg.norm(array - clip_center, axis=1)))
    center = array[center_index]
    orientation = _estimate_grid_orientation(array, max_rotation_deg)
    angle = math.radians(orientation)
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
    if strict and disagreement > max_axis_disagreement_deg:
        raise ValueError(
            f"X/Y angle disagreement is {disagreement:.3f} deg, above "
            f"{max_axis_disagreement_deg:.3f} deg. Check YOLO false positives."
        )
    combined_angle = float((angle_x + angle_y) / 2.0)
    confidence = float(np.clip(1.0 - disagreement / max(max_axis_disagreement_deg * 2.0, 1e-6), 0.0, 1.0))
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
            if signed_distance < margin_distance:
                continue
            corner_inside = [cv2.pointPolygonTest(contour, (float(p[0]), float(p[1])), False) >= 0 for p in polygon]
            partial = not all(corner_inside)
            if not include_edge and partial:
                continue
            x1, y1 = np.floor(polygon.min(axis=0)).astype(int)
            x2, y2 = np.ceil(polygon.max(axis=0)).astype(int)
            cx, cy = int(round(float(center[0]))), int(round(float(center[1])))
            entry: Dict[str, Any] = {
                "index": (ix, iy),
                "center_px": (cx, cy),
                "rect_px": (int(x1), int(y1), int(x2), int(y2)),
                "crop_rect_px": (int(x1), int(y1), int(x2), int(y2)),
                "polygon_px": tuple((float(p[0]), float(p[1])) for p in polygon),
                "real_coord": ((cx - wafer_cx) / pixel_per_unit, (wafer_cy - cy) / pixel_per_unit),
                "is_edge_partial": bool(partial),
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
    elif contour is not None:
        partial = not all(cv2.pointPolygonTest(contour, (float(p[0]), float(p[1])), False) >= 0 for p in polygon)
        ring = True
    else:
        partial, ring = True, True
    rx = (qx - die_map.wafer_cx) / die_map.pixel_per_unit
    ry = (die_map.wafer_cy - qy) / die_map.pixel_per_unit
    cx, cy = int(round(float(center[0]))), int(round(float(center[1])))
    return {
        "input_type": input_type,
        "query_px": (qx, qy),
        "die_index": (ix, iy),
        "die_center_px": (cx, cy),
        "die_rect_px": (int(x1), int(y1), int(x2), int(y2)),
        "crop_rect_px": (int(x1), int(y1), int(x2), int(y2)),
        "die_polygon_px": tuple((float(p[0]), float(p[1])) for p in polygon),
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
        "is_edge_ring": ring,
        "edge_mode": die_map.edge_mode,
        "in_wafer": in_wafer,
    }


# [SECTOR: 70_OVERLAY] --------------------------------------------------------
def make_clip_overlay(clip_image: ImageInput, estimate: GridEstimate) -> np.ndarray:
    overlay = _load_bgr(clip_image).copy()
    for point in estimate.points_clip:
        cv2.circle(overlay, tuple(np.rint(point).astype(int)), 4, (0, 215, 255), -1, cv2.LINE_AA)
    center = tuple(np.rint(estimate.center_corner_clip).astype(int))
    side = tuple(np.rint(estimate.side_corner_clip).astype(int))
    below = tuple(np.rint(estimate.below_corner_clip).astype(int))
    cv2.circle(overlay, center, 7, (0, 255, 0), -1, cv2.LINE_AA)
    cv2.arrowedLine(overlay, center, side, (255, 120, 0), 2, cv2.LINE_AA, tipLength=0.12)
    cv2.arrowedLine(overlay, center, below, (255, 0, 255), 2, cv2.LINE_AA, tipLength=0.12)
    label = f"Px={estimate.pitch_x:.2f} Py={estimate.pitch_y:.2f} A={estimate.angle_deg:.3f}deg"
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
            polygon = np.rint(np.asarray(die["polygon_px"])).astype(np.int32)
            colour = (0, 80, 255) if die["is_edge"] else (0, 220, 0)
            cv2.polylines(overlay, [polygon], True, colour, thickness, cv2.LINE_AA)
    cv2.circle(overlay, (die_map.wafer_cx, die_map.wafer_cy), max(4, thickness * 3), (0, 0, 255), -1)
    cv2.circle(overlay, (int(round(die_map.x0)), int(round(die_map.y0))),
               max(5, thickness * 4), (0, 255, 0), -1)
    return overlay


# [SECTOR: 80_PIPELINE] -------------------------------------------------------
def build_die_map_from_yolo(
    wafer_image: ImageInput,
    clip_image: ImageInput,
    detections: Union[str, Path, np.ndarray, Sequence[Any]],
    *,
    clip_origin: Optional[Point] = None,
    detection_format: DetectionFormat = "auto",
    normalized: Optional[bool] = None,
    confidence_threshold: float = 0.25,
    refine: bool = False,
    pixel_per_unit: float = 32.0,
    include_edge: bool = True,
    edge_margin: float = 1.0,
    edge_mode: str = "circle",
    boundary_max_dimension: int = 2048,
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
    estimate = estimate_grid_from_yolo(
        clip, detections,
        detection_format=detection_format,
        normalized=normalized,
        confidence_threshold=confidence_threshold,
        refine=refine,
    )
    origin_full = (clip_origin[0] + estimate.center_corner_clip[0],
                   clip_origin[1] + estimate.center_corner_clip[1])
    boundary = detect_wafer_boundary(wafer, max_dimension=boundary_max_dimension)
    return generate_die_map(
        boundary, (full_height, full_width), origin_full,
        estimate.pitch_x, estimate.pitch_y, estimate.angle_deg,
        pixel_per_unit=pixel_per_unit,
        include_edge=include_edge,
        edge_margin=edge_margin,
        edge_mode=edge_mode,
        angle_confidence=estimate.angle_confidence,
        grid_estimate=estimate,
    )


build_die_map = build_die_map_from_yolo


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
#     [251.4, 260.9],  # clip 중앙에 가장 가까운 점 -> center corner 후보
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
#     refine=False,                   # 기본 권장: YOLO box 중심을 그대로 사용
#     pixel_per_unit=32.0,            # 실좌표 환산용 px/unit
#     include_edge=True,              # wafer 외곽의 partial die도 map에 포함
#     edge_margin=1.0,
#     edge_mode="circle",            # circle | ring | both
#     boundary_max_dimension=2048,    # 외곽선 검출용 downscale 상한
# )
#
# exact center clip이면 clip_origin은 생략할 수 있습니다.
# 하지만 생산 코드에서는 clip 위치 실수를 방지하기 위해 명시하는 것을 권장합니다.
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
# print("grid angle(deg):", dm.grid_angle_deg)
# print("angle confidence:", dm.angle_confidence)
# print("number of dies:", dm.num_dies)
# print("full image shape:", dm.image_shape)
#
# # 512 clip 안에서 실제 선택된 세 점도 확인할 수 있습니다.
# estimate = dm.grid_estimate
# if estimate is not None:
#     print("center corner in clip:", estimate.center_corner_clip)
#     print("side corner in clip:", estimate.side_corner_clip)
#     print("below corner in clip:", estimate.below_corner_clip)
#     print("angle from X vector:", estimate.angle_x_deg)
#     print("angle from Y vector:", estimate.angle_y_deg)
#
# angle 규칙:
#   - angle > 0이면 오른쪽 이웃으로 갈수록 영상의 Y가 증가하는 기울기입니다.
#   - 같은 +angle 값을 cv2.getRotationMatrix2D에 사용하면 수평 보정할 수 있습니다.
#   - 이 모듈은 full image를 실제 회전시키지 않고 회전 lattice를 원본 좌표에 만듭니다.
#   - 따라서 locate_die() 입력은 항상 원본 10000x10000 좌표를 그대로 사용합니다.
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
#   - 노란점: YOLO가 전달한 전체 십자점
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
# make_wafer_overlay()는 출력 이미지를 별도로 복사하므로 약 300MB가 추가로 필요합니다.
# 메모리가 부족하면 draw_dies=False로 외곽선/기준점만 확인하거나 작은 preview를 씁니다.
#
# -----------------------------------------------------------------------------
# 예제 10. YOLO 중심좌표가 street 중앙에서 약간 벗어날 때만 refine 사용
# -----------------------------------------------------------------------------
#
# dm = build_die_map_from_yolo(
#     wafer_bgr,
#     center_clip_bgr,
#     yolo_data,
#     clip_origin=clip_origin,
#     detection_format="xyxy_conf_class",
#     refine=True,
# )
#
# refine=False:
#   - 기본 권장값입니다.
#   - 학습 모델의 bbox 중심을 신뢰하고 색상 영향을 전혀 받지 않습니다.
#
# refine=True:
#   - YOLO 좌표 주변의 Lab 양방향 경계를 이용해 street 중심을 미세 보정합니다.
#   - 특정 빨강/초록/파랑 hue를 임계값으로 사용하지 않습니다.
#   - YOLO 점이 실제 십자점에서 너무 멀면 잘못된 경계를 선택할 수 있으므로 overlay를 확인합니다.
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
#   5) angle_x_deg와 angle_y_deg 차이가 작은지
#   6) clip overlay와 wafer overlay가 실제 street/grid에 맞는지
