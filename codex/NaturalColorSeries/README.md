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

The four `*_notch.png` files preserve their corresponding source image and add
one small wafer-orientation notch at 6 o'clock. The source files are unchanged.
Only the local notch ROI comes from the image edit; pixels outside that ROI are
restored from the exact source image.

For each sample:

- `*_notch_overlay.png` marks the fitted wafer circle, centre-to-notch
  vector, candidate notch arc, and selected deepest point.
- `*_notch_zoom.png` enlarges the selected notch area.
- `*_notch_aligned.png` is the notch-angle-corrected wafer image.

`notch_detection_contact_sheet.png` collects the four enlarged detections for
quick visual confirmation.

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
