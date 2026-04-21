# -*- coding: utf-8 -*-
"""
Scalp Channel Mapping & EDF Extraction for P2, P10, P12
========================================================
These three patients have concurrent scalp EEG (10-20 system) co-recorded
with their SEEG. This script:
  1. Produces a definitive anatomical mapping table (CSV)
  2. Extracts ONLY the scalp channels from every EDF for these patients
  3. Saves cleaned EDF files (corrected channel names, scalp-only montage)

10-20 naming rules (confirmed by co-supervisor):
  Odd number  = LEFT hemisphere
  Even number = RIGHT hemisphere
  Z suffix    = MIDLINE
  01/02 in P2 are typos for O1/O2 — corrected on export.
"""

import os, gc, warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import mne; mne.set_log_level('ERROR')
warnings.filterwarnings('ignore')

SEEG_ROOT = Path(r"G:\PHD Datasets\Data\Thalamus\SEEG Seizure Data")
OUT_ROOT  = Path(r"D:\Projects\phd\PSEG\pges_toolkit\results\scalp_channel_mapping")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

def log(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ── Definitive anatomical mapping (10-20 system) ──────────────────────────────
# Key: exact channel name as it appears in the EDF (uppercase match)
# Corrected: 01→O1, 02→O2 (P2 recording typo)
SCALP_ANATOMY = {
    # Frontal Pole
    'FP1':  {'region': 'Frontal Pole',    'hemisphere': 'Left',    'laterality': 'Lateral'},
    'FP2':  {'region': 'Frontal Pole',    'hemisphere': 'Right',   'laterality': 'Lateral'},
    # Frontal
    'F7':   {'region': 'Frontal',         'hemisphere': 'Left',    'laterality': 'Far-lateral'},
    'F3':   {'region': 'Frontal',         'hemisphere': 'Left',    'laterality': 'Lateral'},
    'FZ':   {'region': 'Frontal',         'hemisphere': 'Midline', 'laterality': 'Midline'},
    'F4':   {'region': 'Frontal',         'hemisphere': 'Right',   'laterality': 'Lateral'},
    'F8':   {'region': 'Frontal',         'hemisphere': 'Right',   'laterality': 'Far-lateral'},
    # Fronto-Temporal (P10 only)
    'FT9':  {'region': 'FrontoTemporal',  'hemisphere': 'Left',    'laterality': 'Lateral'},
    'FT10': {'region': 'FrontoTemporal',  'hemisphere': 'Right',   'laterality': 'Lateral'},
    # Temporal
    'T3':   {'region': 'Temporal',        'hemisphere': 'Left',    'laterality': 'Anterior'},
    'T4':   {'region': 'Temporal',        'hemisphere': 'Right',   'laterality': 'Anterior'},
    'T5':   {'region': 'Temporal',        'hemisphere': 'Left',    'laterality': 'Posterior'},
    'T6':   {'region': 'Temporal',        'hemisphere': 'Right',   'laterality': 'Posterior'},
    # Central
    'C3':   {'region': 'Central',         'hemisphere': 'Left',    'laterality': 'Lateral'},
    'CZ':   {'region': 'Central',         'hemisphere': 'Midline', 'laterality': 'Midline'},
    'C4':   {'region': 'Central',         'hemisphere': 'Right',   'laterality': 'Lateral'},
    # Parietal
    'P3':   {'region': 'Parietal',        'hemisphere': 'Left',    'laterality': 'Lateral'},
    'PZ':   {'region': 'Parietal',        'hemisphere': 'Midline', 'laterality': 'Midline'},
    'P4':   {'region': 'Parietal',        'hemisphere': 'Right',   'laterality': 'Lateral'},
    # Occipital
    'O1':   {'region': 'Occipital',       'hemisphere': 'Left',    'laterality': 'Lateral'},
    'O2':   {'region': 'Occipital',       'hemisphere': 'Right',   'laterality': 'Lateral'},
    # P2 typos (01/02 are O1/O2 mislabelled in recording system)
    '01':   {'region': 'Occipital',       'hemisphere': 'Left',    'laterality': 'Lateral',
             'corrected_name': 'O1', 'note': 'Typo in P2 EDF — should be O1'},
    '02':   {'region': 'Occipital',       'hemisphere': 'Right',   'laterality': 'Lateral',
             'corrected_name': 'O2', 'note': 'Typo in P2 EDF — should be O2'},
}

# All patients — script will auto-detect who has scalp channels
# Full census (April 2026):
#   Full 10-20 set: P2 (19 ch, incl. 01/02 typos), P10 (18 ch), P12 (19 ch)
#   Partial (C3/C4 only): P6, P13
#   None: P1, P3, P4, P5, P7, P8, P9, P11, P14, P15
PATIENTS_ALL = [f'P{i}' for i in range(1, 16)]


def find_scalp_channels(ch_names: list) -> dict:
    """
    Given a list of channel names from an EDF, identify which are 10-20 scalp
    channels. Returns {original_name: anatomy_dict}.
    """
    found = {}
    ch_upper = {c.upper(): c for c in ch_names}
    for key, anatomy in SCALP_ANATOMY.items():
        if key in ch_upper:
            found[ch_upper[key]] = {**anatomy, 'corrected_name': anatomy.get('corrected_name', key)}
    return found


def extract_scalp_edf(edf_path: Path, out_dir: Path) -> dict:
    """
    Read an EDF, extract scalp channels only, save a new EDF with:
      - corrected channel names (01→O1, 02→O2)
      - only scalp electrodes (no SEEG, ECG, EOG, DC, triggers)
    Returns summary dict.
    """
    raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose=False)
    scalp_map = find_scalp_channels(raw.ch_names)

    if not scalp_map:
        raw.close()
        return {'file': edf_path.name, 'n_scalp': 0, 'channels': []}

    # Pick only scalp channels
    picks = mne.pick_channels(raw.ch_names, include=list(scalp_map.keys()), ordered=True)
    raw.load_data()
    raw_scalp = raw.pick(picks)

    # Rename corrected channels (01→O1, 02→O2)
    rename_map = {orig: info['corrected_name']
                  for orig, info in scalp_map.items()
                  if orig != info['corrected_name']}
    if rename_map:
        raw_scalp.rename_channels(rename_map)

    # Set channel types to EEG
    raw_scalp.set_channel_types({ch: 'eeg' for ch in raw_scalp.ch_names})

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (edf_path.stem + '_scalp_only.edf')
    mne.export.export_raw(str(out_path), raw_scalp, fmt='edf', overwrite=True)

    result = {
        'file': edf_path.name,
        'out_file': out_path.name,
        'n_scalp': len(raw_scalp.ch_names),
        'channels': raw_scalp.ch_names,
        'renamed': rename_map,
        'duration_s': raw.n_times / raw.info['sfreq'],
    }
    del raw, raw_scalp; gc.collect()
    return result


def build_mapping_table() -> pd.DataFrame:
    """Build the full channel→anatomy mapping table for all 3 patients."""
    rows = []
    for pid in PATIENTS_ALL:
        pdir = SEEG_ROOT / f"{pid}_SEEG"
        edfs = sorted(pdir.glob('*.edf'))
        if not edfs:
            continue
        # Use first EDF to get channel list (all EDFs for a patient have same montage)
        raw = mne.io.read_raw_edf(str(edfs[0]), preload=False, verbose=False)
        scalp_map = find_scalp_channels(raw.ch_names)
        raw.close()

        for orig_name, anatomy in scalp_map.items():
            rows.append({
                'Patient':          pid,
                'Original_Channel': orig_name,
                'Corrected_Name':   anatomy.get('corrected_name', orig_name),
                'Region':           anatomy['region'],
                'Hemisphere':       anatomy['hemisphere'],
                'Laterality':       anatomy['laterality'],
                'Naming_Rule':      ('Z=Midline' if anatomy['hemisphere'] == 'Midline'
                                     else f"{'Odd' if orig_name[-1].isdigit() and int(orig_name[-1]) % 2 == 1 else 'Even'}={'Left' if anatomy['hemisphere']=='Left' else 'Right'}"),
                'Note':             anatomy.get('note', ''),
                'n_EDFs_in_patient': len(edfs),
            })
    return pd.DataFrame(rows)


def main():
    log("=" * 65)
    log("Scalp Channel Mapping & EDF Extraction — P2, P10, P12")
    log("=" * 65)

    # ── 1. Build and save mapping table ──────────────────────────────────────
    log("\n[1] Building anatomical mapping table...")
    mapping_df = build_mapping_table()
    mapping_path = OUT_ROOT / "scalp_channel_anatomy_map.csv"
    mapping_df.to_csv(mapping_path, index=False)
    log(f"  Saved: {mapping_path}")

    # Pretty-print summary
    log("\n  10-20 Scalp Channels Found per Patient:")
    log(f"  {'Patient':<8} {'Channel':<10} {'Corrected':<10} {'Region':<18} {'Hemisphere':<10} {'Note'}")
    log(f"  {'-'*80}")
    for _, row in mapping_df.iterrows():
        note = f"  ** {row['Note']}" if row['Note'] else ''
        log(f"  {row['Patient']:<8} {row['Original_Channel']:<10} {row['Corrected_Name']:<10} "
            f"{row['Region']:<18} {row['Hemisphere']:<10}{note}")

    # ── 2. Extract scalp-only EDFs ────────────────────────────────────────────
    log("\n[2] Extracting scalp-only EDF files (all patients with >= 1 scalp channel)...")
    extraction_rows = []
    patients_with_scalp = mapping_df['Patient'].unique().tolist()
    for pid in patients_with_scalp:
        pdir    = SEEG_ROOT / f"{pid}_SEEG"
        out_dir = OUT_ROOT / f"{pid}_scalp_edfs"
        edfs    = sorted(pdir.glob('*.edf'))
        log(f"\n  {pid}: {len(edfs)} EDFs → {out_dir.name}/")
        for edf in edfs:
            result = extract_scalp_edf(edf, out_dir)
            if result['n_scalp'] > 0:
                log(f"    {edf.name} → {result['out_file']}  "
                    f"({result['n_scalp']} scalp ch, {result['duration_s']:.0f}s)")
                if result['renamed']:
                    log(f"      Renamed: {result['renamed']}")
            extraction_rows.append({'Patient': pid, **result})

    # ── 3. Summary report ────────────────────────────────────────────────────
    log("\n" + "=" * 65)
    log("SUMMARY — Scalp Channels in SEEG Recordings")
    log("=" * 65)

    log("""
  Context:
  These 3 patients (P2, P10, P12) were recorded with simultaneous scalp EEG
  (standard 10-20 system) AND thalamic SEEG. The scalp channels are embedded
  in the same multi-channel EDF alongside the intracranial electrodes.

  Naming rules confirmed (per co-supervisor):
    Odd number  → LEFT hemisphere   (F3, C3, P3, T3, T5, FP1, O1 ...)
    Even number → RIGHT hemisphere  (F4, C4, P4, T4, T6, FP2, O2 ...)
    Z suffix    → MIDLINE           (FZ, CZ, PZ)

  Patients and their scalp channels:
""")
    for pid in sorted(mapping_df['Patient'].unique()):
        sub = mapping_df[mapping_df['Patient'] == pid]
        ch_list = ', '.join(sub['Corrected_Name'].tolist())
        log(f"  {pid} ({len(sub)} scalp ch): {ch_list}")

    log(f"""
  P2 typo correction:
    '01' in EDF → corrected to 'O1' (Left Occipital)
    '02' in EDF → corrected to 'O2' (Right Occipital)
    (These are labelling errors in the acquisition system, not missing channels)

  P10 extras:
    FT9  = Left FrontoTemporal  (not present in P2/P12)
    FT10 = Right FrontoTemporal (not present in P2/P12)

  Output EDFs saved to: {OUT_ROOT}
    Each patient folder contains scalp-only EDF for each original file
    (AC5, SC5, and all seizure EDFs).
""")

    log(f"Done. All outputs: {OUT_ROOT}")


if __name__ == '__main__':
    main()
