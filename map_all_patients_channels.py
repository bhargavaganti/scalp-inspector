# -*- coding: utf-8 -*-
"""
Complete Scalp-Thalamic Channel Mapping — All 15 Patients
==========================================================
Three mapping types per patient:

  TYPE A — Concurrent scalp EEG (actual 10-20 channels in EDF):
    P2  : 19 ch (FP1/2, F3/Z/4/7/8, C3/Z/4, P3/Z/4, T3/4/5/6 + 01/02 typos)
    P10 : 18 ch (FP1/2, F4/Z/7/8, FT9/10, C4/Z, P4/Z, T3/4/5/6, O1/O2)
    P12 : 19 ch (FP1/2, F3/Z/4/7/8, C3/Z/4, P3/Z/4, T3/4/5/6, O1/O2)

  TYPE B — Partial concurrent scalp (only a few 10-20 channels in EDF):
    P6  : C3, C4 only
    P13 : C3, C4 only

  TYPE C — Functional projection mapping (no scalp EEG in EDF):
    P1,P3,P5,P7,P8,P9,P11,P14,P15,P4 — derived from:
      nucleus → cortical projection region → 10-20 electrode coverage
      (same physiological logic as §1.3 of thesis)

Naming rules (10-20 system, confirmed by co-supervisor):
  Odd  number → LEFT hemisphere
  Even number → RIGHT hemisphere
  Z suffix    → MIDLINE

Outputs:
  results/scalp_channel_mapping/all_patients_channel_map.csv
  results/scalp_channel_mapping/channel_map_summary.txt
  results/scalp_channel_mapping/figures/channel_map_heatmap.png
"""

import os, gc, warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne; mne.set_log_level('ERROR')
warnings.filterwarnings('ignore')

SEEG_ROOT = Path(r"G:\PHD Datasets\Data\Thalamus\SEEG Seizure Data")
OUT_ROOT  = Path(r"D:\Projects\phd\PSEG\pges_toolkit\results\scalp_channel_mapping")
OUT_ROOT.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUT_ROOT / "figures"; FIG_DIR.mkdir(exist_ok=True)

def log(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ── Anatomy of all standard 10-20 channels ───────────────────────────────────
SCALP_10_20_ANATOMY = {
    'FP1':  {'region': 'Frontal Pole',   'hemisphere': 'Left',    'area': 'Prefrontal'},
    'FP2':  {'region': 'Frontal Pole',   'hemisphere': 'Right',   'area': 'Prefrontal'},
    'F7':   {'region': 'Frontal',        'hemisphere': 'Left',    'area': 'Prefrontal-lateral'},
    'F3':   {'region': 'Frontal',        'hemisphere': 'Left',    'area': 'Prefrontal'},
    'FZ':   {'region': 'Frontal',        'hemisphere': 'Midline', 'area': 'Prefrontal/Cingulate'},
    'F4':   {'region': 'Frontal',        'hemisphere': 'Right',   'area': 'Prefrontal'},
    'F8':   {'region': 'Frontal',        'hemisphere': 'Right',   'area': 'Prefrontal-lateral'},
    'FT9':  {'region': 'FrontoTemporal', 'hemisphere': 'Left',    'area': 'FrontoTemporal'},
    'FT10': {'region': 'FrontoTemporal', 'hemisphere': 'Right',   'area': 'FrontoTemporal'},
    'T3':   {'region': 'Temporal',       'hemisphere': 'Left',    'area': 'Temporal-anterior'},
    'T4':   {'region': 'Temporal',       'hemisphere': 'Right',   'area': 'Temporal-anterior'},
    'T5':   {'region': 'Temporal',       'hemisphere': 'Left',    'area': 'Temporal-posterior'},
    'T6':   {'region': 'Temporal',       'hemisphere': 'Right',   'area': 'Temporal-posterior'},
    'C3':   {'region': 'Central',        'hemisphere': 'Left',    'area': 'Motor cortex M1'},
    'CZ':   {'region': 'Central',        'hemisphere': 'Midline', 'area': 'Motor cortex M1 (vertex)'},
    'C4':   {'region': 'Central',        'hemisphere': 'Right',   'area': 'Motor cortex M1'},
    'P3':   {'region': 'Parietal',       'hemisphere': 'Left',    'area': 'Somatosensory/Parietal assoc.'},
    'PZ':   {'region': 'Parietal',       'hemisphere': 'Midline', 'area': 'Parietal association'},
    'P4':   {'region': 'Parietal',       'hemisphere': 'Right',   'area': 'Somatosensory/Parietal assoc.'},
    'O1':   {'region': 'Occipital',      'hemisphere': 'Left',    'area': 'Visual cortex V1'},
    'O2':   {'region': 'Occipital',      'hemisphere': 'Right',   'area': 'Visual cortex V1'},
}

# ── Thalamic nucleus → cortical projection → primary scalp electrodes ─────────
# Based on established thalamocortical circuit anatomy (see thesis §1.3)
NUCLEUS_PROJECTION = {
    'ANT': {
        'cortical_target':   'Anterior/mid cingulate cortex, parahippocampal gyrus',
        'circuit':           'Papez memory circuit (ANT→cingulate→entorhinal→hippocampus→mammillary bodies→ANT)',
        'primary_scalp':     ['FZ', 'CZ'],        # anterior/mid cingulate is directly under Fz–Cz
        'secondary_scalp':   ['F3', 'F4', 'T3', 'T4'],  # parahippocampal via temporal
        'pges_pattern':      'Limbic theta collapse; variable onset; cingulate suppression at Fz/Cz',
        'why':               'ANT→cingulate cortex projects to mid-cingulate (under Cz) and anterior '
                             'cingulate (under Fz). In CHB-MIT/TUH recordings of temporal lobe epilepsy, '
                             'post-ictal cingulate suppression is captured at Cz and Fz — the same '
                             'scalp locus that reflects ANT activity during PGES.',
    },
    'CeM': {
        'cortical_target':   'Bilateral motor cortex, premotor cortex, supplementary motor area, basal ganglia',
        'circuit':           'Thalamocortical arousal / motor synchronisation (most diffuse projection)',
        'primary_scalp':     ['CZ', 'C3', 'C4'],  # motor cortex bilateral
        'secondary_scalp':   ['FZ', 'F3', 'F4'],  # premotor/SMA
        'pges_pattern':      'Broad amplitude drop; fast SR rise; bilateral central suppression',
        'why':               'CeM is the most diffuse thalamic projector — bilateral motor, premotor, SMA, '
                             'and basal ganglia. The motor cortex suppression during PGES is the dominant '
                             'scalp PGES signature, visible at C3/C4/Cz. CHB-MIT records C3/C4/Cz extensively.',
    },
    'CL': {
        'cortical_target':   'Primary motor cortex M1, sensorimotor strip',
        'circuit':           'Motor relay; sensorimotor integration (CL→M1 direct projection)',
        'primary_scalp':     ['C3', 'C4'],         # M1 is directly under C3 (left) and C4 (right)
        'secondary_scalp':   ['CZ', 'P3', 'P4'],   # sensorimotor strip extends to parietal
        'pges_pattern':      'Consistent central suppression; most stereotyped PGES pattern',
        'why':               'CL projects specifically to primary motor cortex M1 (Brodmann area 4). '
                             'M1 is directly under C3 (left hemisphere) and C4 (right hemisphere). '
                             'Post-ictal motor cortex suppression after FBTCS is the most consistent '
                             'scalp EEG finding — visible as central amplitude reduction at C3/C4.',
    },
    'MD': {
        'cortical_target':   'Dorsolateral prefrontal cortex (DLPFC), orbitofrontal cortex',
        'circuit':           'Limbic-prefrontal circuit; executive function (MD→DLPFC reciprocal)',
        'primary_scalp':     ['F3', 'F4'],          # DLPFC under F3/F4
        'secondary_scalp':   ['FP1', 'FP2', 'FZ'], # orbitofrontal under Fp1/Fp2; anterior cingulate under Fz
        'pges_pattern':      'Frontal alpha/beta collapse; DLPFC suppression',
        'why':               'MD has reciprocal connections with DLPFC and orbitofrontal cortex. '
                             'These frontal areas generate frontal alpha/beta visible at F3/F4. '
                             'During PGES, frontal cortex suppresses, collapsing these rhythms. '
                             'F3/F4 and Fp1/Fp2 capture the MD→DLPFC suppression.',
    },
}

# ── Per-patient metadata ───────────────────────────────────────────────────────
PATIENT_META = {
    'P1':  {'nucleus': 'CeM', 'contact': 'LT2-LT3',      'sz_types': 'FBTCS+FIAS'},
    'P2':  {'nucleus': 'CL',  'contact': 'LT1-LT2',      'sz_types': 'FBTCS'},
    'P3':  {'nucleus': 'CeM', 'contact': 'LT2-LT3',      'sz_types': 'FBTCS+FIAS'},
    'P4':  {'nucleus': 'MD',  'contact': 'LT2-LT3',      'sz_types': 'FBTCS'},
    'P5':  {'nucleus': 'CeM', 'contact': 'LT2-LT3',      'sz_types': 'FBTCS+FIAS'},
    'P6':  {'nucleus': 'MD',  'contact': 'LTHAL2-LTHAL3','sz_types': 'FBTCS'},
    'P7':  {'nucleus': 'CL',  'contact': 'LT2-LT3',      'sz_types': 'FBTCS+FIAS'},
    'P8':  {'nucleus': 'CL',  'contact': 'LT1-LT2',      'sz_types': 'FBTCS+FIAS'},
    'P9':  {'nucleus': 'CeM', 'contact': 'RT1-RT2',      'sz_types': 'FIAS'},
    'P10': {'nucleus': 'ANT', 'contact': 'INS1-INS2',    'sz_types': 'FIAS'},
    'P11': {'nucleus': 'ANT', 'contact': 'RT1-RT2',      'sz_types': 'FIAS'},
    'P12': {'nucleus': 'ANT', 'contact': 'RSR1-RSR2',    'sz_types': 'FIAS'},
    'P13': {'nucleus': 'ANT', 'contact': 'RT1-RT2',      'sz_types': 'FIAS'},
    'P14': {'nucleus': 'ANT', 'contact': 'RT1-RT2',      'sz_types': 'FIAS'},
    'P15': {'nucleus': 'ANT', 'contact': 'LTHAL1-LTHAL2','sz_types': 'FIAS'},
}

# Actual scalp channels found in EDFs (from census run)
ACTUAL_SCALP_CHANNELS = {
    'P2':  {'type': 'A', 'channels': ['FP1','FP2','F7','F3','FZ','F4','F8',
                                       'C3','CZ','C4','P3','PZ','P4',
                                       'T3','T4','T5','T6','O1','O2'],
            'notes': '01 and 02 in EDF are typos for O1 and O2 — corrected'},
    'P6':  {'type': 'B', 'channels': ['C3','C4'],
            'notes': 'Only central electrodes present; partial montage'},
    'P10': {'type': 'A', 'channels': ['FP1','FP2','F7','FZ','F4','F8',
                                       'FT9','FT10','T3','T4','T5','T6',
                                       'C4','CZ','P4','PZ','O1','O2'],
            'notes': 'Has FT9/FT10 (fronto-temporal); missing F3, C3, P3 (left-side partial)'},
    'P12': {'type': 'A', 'channels': ['FP1','FP2','F7','F3','FZ','F4','F8',
                                       'C3','CZ','C4','P3','PZ','P4',
                                       'T3','T4','T5','T6','O1','O2'],
            'notes': 'Full standard 10-20 montage'},
    'P13': {'type': 'B', 'channels': ['C3','C4'],
            'notes': 'Only central electrodes present; partial montage'},
}


def build_full_mapping():
    """Build the complete channel mapping table for all 15 patients."""
    rows = []

    for pid in [f'P{i}' for i in range(1, 16)]:
        meta    = PATIENT_META[pid]
        nucleus = meta['nucleus']
        proj    = NUCLEUS_PROJECTION[nucleus]

        if pid in ACTUAL_SCALP_CHANNELS:
            actual = ACTUAL_SCALP_CHANNELS[pid]
            map_type = actual['type']
            channels = actual['channels']
            notes    = actual['notes']
        else:
            # Type C: derive from nucleus projection
            map_type = 'C'
            channels = proj['primary_scalp'] + proj['secondary_scalp']
            notes    = f'Functional projection — no concurrent scalp EEG in EDF'

        for ch in channels:
            anat = SCALP_10_20_ANATOMY.get(ch, {})
            is_primary = (map_type in ('A', 'B') or
                          ch in proj['primary_scalp'])

            rows.append({
                'Patient':            pid,
                'Nucleus':            nucleus,
                'TH_Contact':         meta['contact'],
                'Sz_Types':           meta['sz_types'],
                'Mapping_Type':       map_type,
                'Channel':            ch,
                'Hemisphere':         anat.get('hemisphere', 'Unknown'),
                'Scalp_Region':       anat.get('region', 'Unknown'),
                'Cortical_Area':      anat.get('area', 'Unknown'),
                'Is_Primary':         is_primary,
                'Cortical_Target':    proj['cortical_target'],
                'PGES_Pattern':       proj['pges_pattern'],
                'Circuit':            proj['circuit'],
                'Notes':              notes,
                'In_EDF':             map_type in ('A', 'B'),
            })

    return pd.DataFrame(rows)


def plot_channel_coverage(df: pd.DataFrame):
    """Heatmap: patients × scalp channels, coloured by mapping type."""
    all_channels = ['FP1','FP2','F7','F3','FZ','F4','F8',
                    'FT9','FT10','T3','T4','T5','T6',
                    'C3','CZ','C4','P3','PZ','P4','O1','O2']
    patients     = [f'P{i}' for i in range(1, 16)]

    # 0=none, 1=type-C (projected), 2=type-B (partial actual), 3=type-A (full actual)
    TYPE_VAL = {'C': 1, 'B': 2, 'A': 3}
    mat = np.zeros((len(patients), len(all_channels)))

    for _, row in df.iterrows():
        pi = patients.index(row['Patient'])
        if row['Channel'] in all_channels:
            ci = all_channels.index(row['Channel'])
            mat[pi, ci] = max(mat[pi, ci], TYPE_VAL.get(row['Mapping_Type'], 0))

    fig, ax = plt.subplots(figsize=(14, 8))
    cmap = plt.cm.get_cmap('RdYlGn', 4)
    im = ax.imshow(mat, cmap=cmap, vmin=-0.5, vmax=3.5, aspect='auto')

    ax.set_xticks(range(len(all_channels)))
    ax.set_xticklabels(all_channels, rotation=45, ha='right', fontsize=10)
    ax.set_yticks(range(len(patients)))
    ax.set_yticklabels([f'{p} ({PATIENT_META[p]["nucleus"]})' for p in patients], fontsize=10)

    # Add value text
    for i in range(len(patients)):
        for j in range(len(all_channels)):
            v = mat[i, j]
            if v > 0:
                label = {1: 'proj', 2: 'part', 3: 'EDF'}[int(v)]
                ax.text(j, i, label, ha='center', va='center', fontsize=6,
                        color='black' if v < 3 else 'white', fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
    cbar.set_ticklabels(['None', 'Type-C\n(projected)', 'Type-B\n(partial EDF)', 'Type-A\n(full EDF)'])

    ax.set_title('Scalp Channel Coverage — All 15 SEEG Patients\n'
                 '(Type-A: actual 10-20 channels in EDF  |  '
                 'Type-B: partial channels in EDF  |  '
                 'Type-C: nucleus projection mapping)',
                 fontsize=11)
    ax.set_xlabel('10-20 Scalp Electrode', fontsize=11)
    ax.set_ylabel('Patient (Nucleus)', fontsize=11)

    plt.tight_layout()
    fig.savefig(FIG_DIR / 'channel_map_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    log(f"  Heatmap saved: {FIG_DIR / 'channel_map_heatmap.png'}")


def write_summary(df: pd.DataFrame):
    """Write a human-readable summary report."""
    lines = []
    lines.append("=" * 70)
    lines.append("SCALP CHANNEL MAPPING — ALL 15 PATIENTS")
    lines.append("=" * 70)
    lines.append("")
    lines.append("10-20 Naming Rules (confirmed by co-supervisor):")
    lines.append("  Odd number  → LEFT hemisphere  (F3, C3, P3, T3, T5, FP1, O1)")
    lines.append("  Even number → RIGHT hemisphere (F4, C4, P4, T4, T6, FP2, O2)")
    lines.append("  Z suffix    → MIDLINE          (FZ, CZ, PZ)")
    lines.append("")
    lines.append("Mapping types:")
    lines.append("  Type A — Full concurrent 10-20 scalp EEG in EDF (actual recorded data)")
    lines.append("  Type B — Partial concurrent scalp (only some 10-20 channels in EDF)")
    lines.append("  Type C — Functional projection (nucleus→cortex→scalp electrode, no EDF data)")
    lines.append("")
    lines.append("-" * 70)

    for pid in [f'P{i}' for i in range(1, 16)]:
        meta    = PATIENT_META[pid]
        nucleus = meta['nucleus']
        proj    = NUCLEUS_PROJECTION[nucleus]
        sub     = df[df['Patient'] == pid]
        map_type = sub['Mapping_Type'].iloc[0] if not sub.empty else 'C'

        type_str = {'A': 'TYPE A — Concurrent scalp EEG in EDF',
                    'B': 'TYPE B — Partial scalp EEG in EDF',
                    'C': 'TYPE C — Functional projection (no scalp EEG in EDF)'}[map_type]

        lines.append(f"\n{pid}  |  Nucleus: {nucleus}  |  Contact: {meta['contact']}  |  "
                     f"Seizures: {meta['sz_types']}")
        lines.append(f"  [{type_str}]")

        if map_type in ('A', 'B'):
            ch_list = sub['Channel'].tolist()
            lines.append(f"  Actual channels in EDF ({len(ch_list)}): {', '.join(ch_list)}")
            notes = sub['Notes'].iloc[0]
            if notes:
                lines.append(f"  Note: {notes}")
        else:
            lines.append(f"  Cortical projection: {proj['cortical_target']}")
            lines.append(f"  Circuit: {proj['circuit']}")
            lines.append(f"  Primary scalp equiv: {', '.join(proj['primary_scalp'])}")
            lines.append(f"  Secondary scalp equiv: {', '.join(proj['secondary_scalp'])}")
            lines.append(f"  PGES pattern: {proj['pges_pattern']}")

    lines.append("\n" + "=" * 70)
    lines.append("NUCLEUS SUMMARY")
    lines.append("=" * 70)

    for nucleus in ['ANT', 'CeM', 'CL', 'MD']:
        proj = NUCLEUS_PROJECTION[nucleus]
        pts  = [p for p, m in PATIENT_META.items() if m['nucleus'] == nucleus]
        lines.append(f"\n{nucleus} ({len(pts)} patients: {', '.join(pts)})")
        lines.append(f"  Cortical target: {proj['cortical_target']}")
        lines.append(f"  Primary scalp electrodes: {', '.join(proj['primary_scalp'])}")
        lines.append(f"  Secondary:                {', '.join(proj['secondary_scalp'])}")
        lines.append(f"  PGES scalp pattern: {proj['pges_pattern']}")
        lines.append(f"  Why scalp pre-training captures this: {proj['why'][:120]}...")

    lines.append("\n" + "=" * 70)
    lines.append("IMPORTANT NOTE ON TYPE C MAPPINGS")
    lines.append("=" * 70)
    lines.append("""
  For patients P1, P3, P4, P5, P7, P8, P9, P11, P14, P15 (Type C):
  NO scalp EEG exists in their EDF recordings. The channel assignments
  listed are FUNCTIONAL EQUIVALENTS — the scalp electrodes where you
  WOULD see the cortical effect of their thalamic nucleus during PGES
  if a scalp EEG were being recorded simultaneously.

  These mappings are used for:
    1. Understanding which scalp channels in CHB-MIT/TUH capture the
       same thalamocortical suppression as each patient's nucleus
    2. Targeted feature extraction if future paired recordings are made
    3. Interpreting why scalp pre-training transfers to each nucleus

  These mappings are NOT:
    - Actual recorded data
    - Usable for additional model training
    - A substitute for concurrent scalp recording

  For clinical use, patients P2, P10, P12 provide the only actual
  concurrent scalp-thalamic paired data (with P6/P13 having partial C3/C4).
""")

    summary_path = OUT_ROOT / "channel_map_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    log(f"  Summary saved: {summary_path}")


def main():
    log("=" * 65)
    log("Scalp Channel Mapping — All 15 Patients")
    log("=" * 65)

    log("\n[1] Building complete mapping table...")
    df = build_full_mapping()

    csv_path = OUT_ROOT / "all_patients_channel_map.csv"
    df.to_csv(csv_path, index=False)
    log(f"  Saved: {csv_path} ({len(df)} rows)")

    log("\n[2] Channel coverage per patient:")
    log(f"  {'Patient':<8} {'Nucleus':<6} {'Type':<8} {'N channels':<12} {'Primary scalp electrodes'}")
    log(f"  {'-'*70}")
    for pid in [f'P{i}' for i in range(1, 16)]:
        sub      = df[df['Patient'] == pid]
        map_type = sub['Mapping_Type'].iloc[0]
        n_ch     = sub['Channel'].nunique()
        primary  = sub[sub['Is_Primary']]['Channel'].tolist()
        nucleus  = PATIENT_META[pid]['nucleus']
        type_lbl = {'A': 'Full EDF', 'B': 'Part EDF', 'C': 'Projected'}[map_type]
        log(f"  {pid:<8} {nucleus:<6} {type_lbl:<8} {n_ch:<12} {', '.join(primary)}")

    log("\n[3] Building heatmap figure...")
    plot_channel_coverage(df)

    log("\n[4] Writing summary report...")
    write_summary(df)

    log("\n" + "=" * 65)
    log("MAPPING LEGEND")
    log("=" * 65)
    log("  Type A (P2, P10, P12): Actual 10-20 scalp EEG co-recorded in EDF")
    log("  Type B (P6, P13):      Partial — only C3 and C4 in EDF")
    log("  Type C (all others):   Functional projection from nucleus anatomy")
    log("                         (no scalp EEG exists in their recordings)")
    log(f"\nDone. Results: {OUT_ROOT}")


if __name__ == '__main__':
    main()
