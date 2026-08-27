# Natural-street AI colour series

These three AI-generated wafer photographs use the same natural thin, dark
saw-street style as `../generated_multicolor_natural_streets_v2.png`, while
changing only plausible low-saturation wafer reflection colours.

| Image | Reflection palette | Validation |
| --- | --- | --- |
| `natural_teal_bluegray.png` | teal, blue-gray, cool silver | 38 x 28 px, 1,129 dies |
| `natural_amber_olive_bronze.png` | amber, olive, bronze | 38 x 28 px, 1,129 dies |
| `natural_rose_violet_iceblue.png` | rose-gray, violet-gray, icy blue | 38 x 28 px, 1,129 dies |
| `white_brown_natural_streets_ai.png` | pearl-white streets with subtle amber/brown residue | 35 x 36 px, 867 dies |

Each `_overlay.png` is the corresponding detected grid. The grid and street
geometry was visually inspected, and all three images were tested by
`test_natural_color_series.py`.

`white_brown_natural_streets_ai.png` is a separate, natural-looking version
of the white/brown street condition. It deliberately has no artificial grid
outside the wafer and no square-tile intersections. Its regression test is
`test_white_brown_natural_streets.py`.

## Notch angle samples

> **현재 상태:** 이 폴더의 notch 이미지와 contact sheet는 과거 시각 실험 기록입니다.
> 현재 `geometry_edge_bottom_sector` 검출기를 이 이미지에 맞춰 튜닝하거나 실제 장비
> 성능 검증에 사용하지 않습니다. 실제 데이터 평가는 현재 코드에서 새로 생성한
> overlay의 기준 원, 추적 contour, 검색 구간, fitting 잔차를 보고 수행해야 합니다.

The four `*_notch.png` files preserve their corresponding source image and add
one small wafer-orientation notch at 6 o'clock. The source files are unchanged.
Only the local notch ROI comes from the image edit; pixels outside that ROI are
restored from the exact source image.

Historical files for each sample:

- `*_notch_overlay.png` marks the fitted wafer circle, centre-to-notch
  vector, candidate notch arc, and selected deepest point.
- `*_notch_zoom.png` enlarges the selected notch area.
- `*_notch_aligned.png` is the notch-angle-corrected wafer image.

`notch_detection_contact_sheet.png` collects the four enlarged detections for
historical visual comparison. It is not production-data validation for the
current geometry-edge detector.

![Four notch detections](./notch_detection_contact_sheet.png)

The user-marked target and the corrected detector output are retained together:

| User-marked reference | Corrected detector output |
|---|---|
| ![User red target](./natural_teal_bluegray_notch_zoom_red.png) | ![Corrected target](./natural_teal_bluegray_notch_zoom.png) |

The selected red point is the midpoint direction of the separated notch region
projected onto the fitted original wafer circle. This is the user-confirmed
angle reference. The small green point is the deepest notch point and is kept
only as a diagnostic. Image-space right is 0 degrees, bottom is 90 degrees, and
the default correction aligns the red reference point to bottom.

The current detector still preserves that red/green point meaning, but its
detection signal has changed: it no longer compares wafer and corner background
colours. It fits the circle from colour-gradient geometry outside the bottom
search sector and measures an inward edge deviation inside the sector. See
[`../README_NOTCH.md`](../README_NOTCH.md) for current options and diagnostics.
