# H-TGAPlot

**GUI tool for Thermogravimetric Analysis (TGA / DrTGA) Visualization**

Note: The software interface and menus are currently available only in Portuguese (Brazilian Portuguese).

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![Version](https://img.shields.io/badge/Version-2.95-informational)
![Platform](https://img.shields.io/badge/Platform-Windows-lightgrey?logo=windows)

H-TGAPlot is a Python desktop application for visualizing and quantitatively analyzing Thermogravimetric Analysis (TGA) and Derivative TGA (DrTGA / DTG) curves. It is part of the **H-SciTools** scientific software suite, developed to support materials engineering research.

---

## Features

**Data loading**
- Import `.txt` files exported by TGA instruments (`Time Temp TGA DrTGA` whitespace-delimited format)
- Automatic encoding detection (UTF-16, UTF-16-LE, UTF-8, Latin-1)
- Automatic extraction of the heating rate (°C/min) from the file header
- Multiple samples loaded simultaneously, managed from a sample list

**Data processing**
- TGA Y-axis as residual mass (%) or mass (mg); X-axis as temperature (°C) or time (s)
- DrTGA computed via Gaussian-filtered numerical derivative, selectable in mg/min, mg/s, mg/°C, %/min, %/s, or %/°C
- Optional Savitzky–Golay smoothing on the TGA curve (adjustable window size)
- Automatic onset/endset/mass-loss detection for up to *N* decomposition events, with automatic merging of overlapping events
- Manual interval analysis: define custom `[T_start, T_end]` ranges to compute onset, endset, midpoint, peak temperature, and both monotonic and raw mass loss

**Style and appearance**
- Color, line width, and line style configurable per sample (TGA and DrTGA curves colored independently)
- Sequential automatic color assignment (matplotlib tab10 palette)
- Independent X/Y grid toggle with automatic or fixed tick interval
- Adjustable font sizes for title, axis labels, ticks, legend, and event annotations
- 10 legend position presets (or automatic placement)

**Graph interactivity**
- Embedded matplotlib canvas directly in the main window, with the native navigation toolbar (zoom, pan, save)
- Draggable onset/endset/mass-loss annotation boxes
- Editable plot title, axis labels, and sample names from the side panel

**Export**
- PNG (300 DPI, white background, publication-ready)
- Event/interval results tables with one-click copy-to-clipboard (TSV) for pasting directly into Excel

---

## Interface

The layout is split into a scrollable control panel (left) and the plot area (right):

```
┌────────────────────┬──────────────────────────────────────────┐
│  H-TGAPlot          │                                          │
│  ─────────────      │                                          │
│  Files               │           Plot area                      │
│  Plot TGA/DrTGA      │       (interactive matplotlib)           │
│  Smoothing           │                                          │
│  Manual interval     │                                          │
│  Onset/Endset auto   │                                          │
│  ─────────────      │                                          │
│  Axes / Grid          │                                          │
│  Fonts / Legend       │                                          │
│  Style controls      │                                          │
│  Export              │                                          │
└────────────────────┴──────────────────────────────────────────┘
```

---

## Requirements

Python 3.9 or higher.

```
numpy
matplotlib
scipy
```

Install dependencies:

```bash
pip install numpy matplotlib scipy
```

> `tkinter` is included in standard Python distributions. If missing on Linux, install via `sudo apt install python3-tk`.
> `scipy` is optional but strongly recommended — without it, smoothing and automatic onset/endset detection are disabled (the app still runs, with an on-screen warning).

---

## Usage

### Running directly

```bash
python H-TGAPlot_v2_95.py
```

### Packaging with PyInstaller

```bash
pyinstaller --onefile --windowed --icon=TGAPlot.ico H-TGAPlot_v2_95.py
```

The resulting executable in `dist/` can be distributed without a Python installation.

> The `TGAPlot.ico` icon file is optional. If not found, the application starts normally without an icon.

### Workflow

1. **Load files** — choose one or more `.txt` TGA export files
2. **Plot** — render TGA, DrTGA, or both curves (dual Y-axis) for all loaded samples
3. **Adjust signal processing** — enable/tune TGA smoothing and DrTGA derivative smoothing as needed
4. **Analyze** — either enable automatic onset/endset detection or define manual temperature intervals for onset, endset, midpoint, and mass-loss calculations
5. **Customize** — adjust per-sample styles, axes, grid, fonts, and legend position
6. **Export** — save the figure as PNG, or copy event/interval result tables as TSV for Excel

---

## Supported file format

H-TGAPlot reads plain-text instrument export files (`.txt`, also tolerant of `.dat` / `.tad` extensions) containing a header line followed by four whitespace-separated numeric columns:

```
Time   Temp   TGA   DrTGA
<one blank/units line>
0.00   25.00   10.000   0.0000
0.50   25.30    9.998  -0.0012
...
```

| Column | Physical quantity |
|---|---|
| Time  | seconds |
| Temp  | °C |
| TGA   | mass (mg) |
| DrTGA | mass loss rate (mg/s) |

If a line above the header contains two numbers in the pattern `<heating rate °C/min> <hold temperature °C>` (e.g. `20.00  1000.0`), the heating rate is parsed automatically and used by the onset/endset algorithms.

---

## Code structure

```
H-TGAPlot_v2_95.py
│
├── resource_path()                   # PyInstaller-compatible asset resolver
├── analisar_intervalo_manual()       # Manual interval onset/endset/midpoint/mass-loss analysis
├── detectar_eventos_tga()            # Automatic multi-event onset/endset detection
├── _mesclar_eventos_sobrepostos()    # Merges overlapping detected events
├── _perda_monotonica()               # Monotonic accumulated mass-loss helper
├── Dataset                           # Per-sample data model (time, temp, TGA, DrTGA, style)
├── TGAModel                          # Core model: datasets, plot settings, intervals, events
├── GraficoEmbutido                   # Embedded matplotlib canvas with draggable annotations
│   ├── plotar_tga() / plotar_dtga() / plotar_ambos()
│   ├── redesenhar()
│   └── exportar_png()
└── App                                # Main tkinter window and side-panel UI
```

---

## Part of the H-SciTools suite

| Tool | Purpose |
|---|---|
| H-AnodPlot | Electrochemical anodization curves |
| H-TGAPlot  | Thermogravimetric Analysis (TGA) |
| H-DMAPlot  | Dynamic Mechanical Analysis (DMA) |
| H-DRXPlot  | X-Ray Diffraction (XRD) |

---

## Author

**Carlos Henrique Amaro da Silva**
M.Sc. in Materials Technology and Industrial Processes — Universidade Feevale (2025)
B.Sc. in Chemical Engineering (2023)

Research focus: surface treatments, anodization, and electrodeposition with biomedical applications.

GitHub: https://github.com/Leindsher

---

## License

This project is licensed under the [MIT License](LICENSE).
