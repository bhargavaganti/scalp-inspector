# Scalp Inspector Tools

Tools for detecting, mapping, and extracting scalp EEG channels from SEEG EDF files.

## Files

| Script | Purpose |
|--------|---------|
| `edf_scalp_inspector.py` | GUI + CLI tool — load EDFs, detect 10-20 scalp channels, export CSV, extract scalp-only EDF |
| `map_scalp_channels.py` | Core 10-20 channel mapping logic and anatomy dictionary |
| `map_all_patients_channels.py` | Batch scan all 15 DACTRL patients, produces `results/scalp_channel_mapping/all_patients_channel_map.csv` |

## Setup
```
pip install -r tools/scalp_inspector/requirements.txt
```
On Linux, also install tkinter: `apt install python3-tk`

## Usage

### GUI mode
```
python tools/scalp_inspector/edf_scalp_inspector.py
```

### CLI / headless mode
```
python tools/scalp_inspector/edf_scalp_inspector.py \
    --folder G:\path\to\edfs \
    --csv output.csv \
    --extract \
    --outdir G:\scalp_edfs
```

### Batch patient mapping
```
python tools/scalp_inspector/map_all_patients_channels.py
```

## Channel mapping rules (10-20 system)
- Odd suffix → Left hemisphere
- Even suffix → Right hemisphere
- Z suffix → Midline
- Mapping types: A (full 10-20 in EDF), B (partial C3/C4), C (functional projection from nucleus anatomy)
