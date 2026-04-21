# -*- coding: utf-8 -*-
"""
EDF Scalp Channel Inspector & Extractor
========================================
GUI tool to:
  1. Load any EDF file (or batch-scan a folder)
  2. Detect which channels are 10-20 scalp electrodes
  3. Apply naming rules: Odd=Left, Even=Right, Z=Midline
  4. Export channel mapping CSV
  5. Optionally extract a scalp-only EDF

Run:  python scripts/edf_scalp_inspector.py
"""

import os, gc, warnings
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
warnings.filterwarnings('ignore')

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from datetime import datetime
import threading

import numpy as np
import pandas as pd
import mne; mne.set_log_level('ERROR')

# ── 10-20 anatomy dictionary ──────────────────────────────────────────────────
SCALP_ANATOMY = {
    'FP1':  {'region': 'Frontal Pole',    'hemisphere': 'Left',    'laterality': 'Lateral'},
    'FP2':  {'region': 'Frontal Pole',    'hemisphere': 'Right',   'laterality': 'Lateral'},
    'F7':   {'region': 'Frontal',         'hemisphere': 'Left',    'laterality': 'Far-lateral'},
    'F3':   {'region': 'Frontal',         'hemisphere': 'Left',    'laterality': 'Lateral'},
    'FZ':   {'region': 'Frontal',         'hemisphere': 'Midline', 'laterality': 'Midline'},
    'F4':   {'region': 'Frontal',         'hemisphere': 'Right',   'laterality': 'Lateral'},
    'F8':   {'region': 'Frontal',         'hemisphere': 'Right',   'laterality': 'Far-lateral'},
    'FT9':  {'region': 'FrontoTemporal',  'hemisphere': 'Left',    'laterality': 'Lateral'},
    'FT10': {'region': 'FrontoTemporal',  'hemisphere': 'Right',   'laterality': 'Lateral'},
    'T3':   {'region': 'Temporal',        'hemisphere': 'Left',    'laterality': 'Anterior'},
    'T4':   {'region': 'Temporal',        'hemisphere': 'Right',   'laterality': 'Anterior'},
    'T5':   {'region': 'Temporal',        'hemisphere': 'Left',    'laterality': 'Posterior'},
    'T6':   {'region': 'Temporal',        'hemisphere': 'Right',   'laterality': 'Posterior'},
    'C3':   {'region': 'Central',         'hemisphere': 'Left',    'laterality': 'Lateral'},
    'CZ':   {'region': 'Central',         'hemisphere': 'Midline', 'laterality': 'Midline'},
    'C4':   {'region': 'Central',         'hemisphere': 'Right',   'laterality': 'Lateral'},
    'P3':   {'region': 'Parietal',        'hemisphere': 'Left',    'laterality': 'Lateral'},
    'PZ':   {'region': 'Parietal',        'hemisphere': 'Midline', 'laterality': 'Midline'},
    'P4':   {'region': 'Parietal',        'hemisphere': 'Right',   'laterality': 'Lateral'},
    'O1':   {'region': 'Occipital',       'hemisphere': 'Left',    'laterality': 'Lateral'},
    'O2':   {'region': 'Occipital',       'hemisphere': 'Right',   'laterality': 'Lateral'},
    # P2 EDF typos — corrected on export
    '01':   {'region': 'Occipital',       'hemisphere': 'Left',    'laterality': 'Lateral',
             'corrected_name': 'O1', 'note': 'Typo in EDF (01 -> O1)'},
    '02':   {'region': 'Occipital',       'hemisphere': 'Right',   'laterality': 'Lateral',
             'corrected_name': 'O2', 'note': 'Typo in EDF (02 -> O2)'},
}

# Naming rule explanation
NAMING_RULES = {
    'Left':    'Odd number suffix',
    'Right':   'Even number suffix',
    'Midline': 'Z suffix',
}


def find_scalp_channels(ch_names: list) -> dict:
    """Return {original_name: anatomy_dict} for any 10-20 channels found."""
    found = {}
    ch_upper = {c.upper().strip(): c for c in ch_names}
    for key, anatomy in SCALP_ANATOMY.items():
        if key in ch_upper:
            orig = ch_upper[key]
            found[orig] = {
                **anatomy,
                'corrected_name': anatomy.get('corrected_name', key),
                'naming_rule': NAMING_RULES.get(anatomy['hemisphere'],
                                                 anatomy['hemisphere']),
            }
    return found


def scan_edf(edf_path: Path) -> dict:
    """Scan a single EDF: return channel info without loading data."""
    try:
        raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose=False)
        scalp = find_scalp_channels(raw.ch_names)
        result = {
            'path':        edf_path,
            'n_total':     len(raw.ch_names),
            'all_channels': raw.ch_names,
            'sfreq':       raw.info['sfreq'],
            'duration_s':  raw.n_times / raw.info['sfreq'],
            'scalp_map':   scalp,
            'n_scalp':     len(scalp),
            'error':       None,
        }
        raw.close()
    except Exception as e:
        result = {'path': edf_path, 'error': str(e),
                  'n_total': 0, 'n_scalp': 0, 'scalp_map': {}}
    return result


def extract_scalp_edf(edf_path: Path, out_path: Path) -> str:
    """Extract scalp-only channels to a new EDF. Returns status message."""
    raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose=False)
    scalp = find_scalp_channels(raw.ch_names)
    if not scalp:
        raw.close()
        return "No scalp channels found — EDF not created."

    picks = mne.pick_channels(raw.ch_names, include=list(scalp.keys()), ordered=True)
    raw.load_data()
    raw_scalp = raw.pick(picks)

    rename_map = {orig: info['corrected_name']
                  for orig, info in scalp.items()
                  if orig != info['corrected_name']}
    if rename_map:
        raw_scalp.rename_channels(rename_map)

    raw_scalp.set_channel_types({ch: 'eeg' for ch in raw_scalp.ch_names})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mne.export.export_raw(str(out_path), raw_scalp, fmt='edf', overwrite=True)
    del raw, raw_scalp; gc.collect()
    return f"Saved: {out_path.name}"


# ═══════════════════════════════════════════════════════════════════════════════
#  GUI
# ═══════════════════════════════════════════════════════════════════════════════

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("EDF Scalp Channel Inspector")
        self.geometry("1050x720")
        self.resizable(True, True)
        self.configure(bg='#f0f0f0')

        self._scanned   = []   # list of scan result dicts
        self._csv_path  = None

        self._build_ui()

    # ── UI construction ────────────────────────────────────────────────────────
    def _build_ui(self):
        # Top bar — buttons
        bar = tk.Frame(self, bg='#2c3e50', pady=6)
        bar.pack(fill='x')

        btn_kw = dict(bg='#3498db', fg='white', relief='flat',
                      padx=12, pady=4, font=('Segoe UI', 9, 'bold'), cursor='hand2')

        tk.Button(bar, text="Load EDF file",   command=self._load_file,   **btn_kw).pack(side='left', padx=6)
        tk.Button(bar, text="Scan folder",     command=self._scan_folder, **btn_kw).pack(side='left', padx=4)
        tk.Button(bar, text="⚡ Batch Process Folder", command=self._batch_process,
                  bg='#8e44ad', fg='white', relief='flat', padx=12, pady=4,
                  font=('Segoe UI', 9, 'bold'), cursor='hand2').pack(side='left', padx=4)
        tk.Button(bar, text="Clear",           command=self._clear,
                  bg='#7f8c8d', fg='white', relief='flat', padx=12, pady=4,
                  font=('Segoe UI', 9, 'bold'), cursor='hand2').pack(side='left', padx=4)

        tk.Button(bar, text="Export CSV",      command=self._export_csv,
                  bg='#27ae60', fg='white', relief='flat', padx=12, pady=4,
                  font=('Segoe UI', 9, 'bold'), cursor='hand2').pack(side='right', padx=6)
        tk.Button(bar, text="Extract scalp EDF", command=self._extract_edf,
                  bg='#e67e22', fg='white', relief='flat', padx=12, pady=4,
                  font=('Segoe UI', 9, 'bold'), cursor='hand2').pack(side='right', padx=4)

        # Status bar
        self._status_var = tk.StringVar(value="Ready. Load an EDF file or scan a folder.")
        tk.Label(self, textvariable=self._status_var, bg='#ecf0f1', anchor='w',
                 font=('Segoe UI', 9), padx=8).pack(fill='x')

        # Main pane: left = file list, right = details
        pane = tk.PanedWindow(self, orient='horizontal', bg='#f0f0f0', sashwidth=6)
        pane.pack(fill='both', expand=True, padx=6, pady=6)

        # ── Left: file list ────────────────────────────────────────────────────
        left = tk.Frame(pane, bg='#f0f0f0')
        pane.add(left, width=340)

        tk.Label(left, text="Scanned EDF Files", bg='#f0f0f0',
                 font=('Segoe UI', 10, 'bold')).pack(anchor='w', pady=(0, 2))

        cols_left = ('file', 'scalp', 'total', 'dur')
        self._file_tree = ttk.Treeview(left, columns=cols_left, show='headings',
                                       selectmode='browse')
        self._file_tree.heading('file',  text='File')
        self._file_tree.heading('scalp', text='Scalp ch')
        self._file_tree.heading('total', text='Total ch')
        self._file_tree.heading('dur',   text='Dur (s)')
        self._file_tree.column('file',  width=160)
        self._file_tree.column('scalp', width=60,  anchor='center')
        self._file_tree.column('total', width=60,  anchor='center')
        self._file_tree.column('dur',   width=60,  anchor='center')

        sb_l = ttk.Scrollbar(left, orient='vertical', command=self._file_tree.yview)
        self._file_tree.configure(yscrollcommand=sb_l.set)
        sb_l.pack(side='right', fill='y')
        self._file_tree.pack(fill='both', expand=True)
        self._file_tree.bind('<<TreeviewSelect>>', self._on_select)

        # Tag colours
        self._file_tree.tag_configure('has_scalp',  background='#d5f5e3')
        self._file_tree.tag_configure('no_scalp',   background='#fdecea')
        self._file_tree.tag_configure('error',      background='#fdf2d0')

        # ── Right: detail view ────────────────────────────────────────────────
        right = tk.Frame(pane, bg='#f0f0f0')
        pane.add(right)

        tk.Label(right, text="Channel Mapping", bg='#f0f0f0',
                 font=('Segoe UI', 10, 'bold')).pack(anchor='w', pady=(0, 2))

        # Summary labels
        self._detail_frame = tk.Frame(right, bg='#ecf0f1', bd=1, relief='groove', pady=4, padx=6)
        self._detail_frame.pack(fill='x', pady=(0, 4))

        self._lbl_file = tk.Label(self._detail_frame, text="No file selected",
                                  bg='#ecf0f1', font=('Segoe UI', 9, 'bold'), anchor='w')
        self._lbl_file.pack(fill='x')
        self._lbl_info = tk.Label(self._detail_frame, text="",
                                  bg='#ecf0f1', font=('Segoe UI', 9), anchor='w',
                                  justify='left')
        self._lbl_info.pack(fill='x')

        # Channel table
        cols_r = ('orig', 'corrected', 'region', 'hemisphere', 'laterality', 'rule', 'note')
        self._ch_tree = ttk.Treeview(right, columns=cols_r, show='headings')
        self._ch_tree.heading('orig',       text='Original name')
        self._ch_tree.heading('corrected',  text='Corrected')
        self._ch_tree.heading('region',     text='Region')
        self._ch_tree.heading('hemisphere', text='Hemisphere')
        self._ch_tree.heading('laterality', text='Laterality')
        self._ch_tree.heading('rule',       text='Naming rule')
        self._ch_tree.heading('note',       text='Note')

        self._ch_tree.column('orig',       width=90)
        self._ch_tree.column('corrected',  width=80)
        self._ch_tree.column('region',     width=110)
        self._ch_tree.column('hemisphere', width=80, anchor='center')
        self._ch_tree.column('laterality', width=90, anchor='center')
        self._ch_tree.column('rule',       width=120)
        self._ch_tree.column('note',       width=160)

        sb_r = ttk.Scrollbar(right, orient='vertical', command=self._ch_tree.yview)
        self._ch_tree.configure(yscrollcommand=sb_r.set)
        sb_r.pack(side='right', fill='y')
        self._ch_tree.pack(fill='both', expand=True)

        # Hemisphere tag colours
        self._ch_tree.tag_configure('Left',    background='#d6eaf8')
        self._ch_tree.tag_configure('Right',   background='#fde8d8')
        self._ch_tree.tag_configure('Midline', background='#eafaf1')
        self._ch_tree.tag_configure('typo',    background='#fef9e7')

        # All channels list (bottom strip)
        tk.Label(right, text="All channels in EDF (non-scalp in grey):",
                 bg='#f0f0f0', font=('Segoe UI', 8)).pack(anchor='w', pady=(4, 0))
        self._all_ch_text = tk.Text(right, height=4, font=('Courier New', 8),
                                    wrap='word', state='disabled', bg='#fafafa')
        self._all_ch_text.pack(fill='x')

    # ── Actions ────────────────────────────────────────────────────────────────
    def _load_file(self):
        path = filedialog.askopenfilename(
            title="Select EDF file",
            filetypes=[("EDF files", "*.edf"), ("All files", "*.*")])
        if not path: return
        self._status("Scanning...")
        threading.Thread(target=self._run_scan, args=([Path(path)],), daemon=True).start()

    def _scan_folder(self):
        folder = filedialog.askdirectory(title="Select folder to scan for EDF files")
        if not folder: return
        edfs = list(Path(folder).rglob("*.edf"))
        if not edfs:
            messagebox.showinfo("No EDFs", "No .edf files found in that folder.")
            return
        self._status(f"Scanning {len(edfs)} EDF files...")
        threading.Thread(target=self._run_scan, args=(edfs,), daemon=True).start()

    def _run_scan(self, paths):
        results = []
        for i, p in enumerate(paths):
            self.after(0, lambda i=i, t=len(paths): self._status(
                f"Scanning {i+1}/{t}: {Path(paths[i]).name}"))
            results.append(scan_edf(p))
        self._scanned.extend(results)
        self.after(0, self._refresh_file_list)
        self.after(0, lambda: self._status(
            f"Done. {len(results)} file(s) scanned. "
            f"{sum(1 for r in results if r['n_scalp'] > 0)} have scalp channels."))

    def _refresh_file_list(self):
        for item in self._file_tree.get_children():
            self._file_tree.delete(item)
        for r in self._scanned:
            name = r['path'].name
            n_s  = r['n_scalp']
            n_t  = r.get('n_total', '?')
            dur  = f"{r.get('duration_s', 0):.0f}" if not r['error'] else 'ERR'
            tag  = 'error' if r['error'] else ('has_scalp' if n_s > 0 else 'no_scalp')
            self._file_tree.insert('', 'end', iid=str(r['path']),
                                   values=(name, n_s, n_t, dur), tags=(tag,))

    def _on_select(self, _event=None):
        sel = self._file_tree.selection()
        if not sel: return
        iid  = sel[0]
        r    = next((x for x in self._scanned if str(x['path']) == iid), None)
        if r is None: return
        self._show_detail(r)

    def _show_detail(self, r):
        # Summary strip
        self._lbl_file.config(text=str(r['path']))
        if r['error']:
            self._lbl_info.config(text=f"ERROR: {r['error']}")
            for row in self._ch_tree.get_children(): self._ch_tree.delete(row)
            return

        n_s = r['n_scalp']
        dur = r.get('duration_s', 0)
        sfreq = r.get('sfreq', 0)
        mapping_type = ('Full 10-20' if n_s >= 16 else
                        'Partial' if n_s > 0 else 'None (projected only)')
        self._lbl_info.config(
            text=(f"Total channels: {r['n_total']}  |  Scalp channels: {n_s}  |  "
                  f"Mapping type: {mapping_type}\n"
                  f"Duration: {dur:.1f}s  |  Sampling: {sfreq:.0f} Hz\n"
                  f"Naming rules:  Odd number = LEFT hemisphere  |  "
                  f"Even number = RIGHT hemisphere  |  Z suffix = MIDLINE"))

        # Channel table
        for row in self._ch_tree.get_children():
            self._ch_tree.delete(row)

        for orig, info in sorted(r['scalp_map'].items(),
                                 key=lambda x: x[1]['region']):
            corrected = info['corrected_name']
            tag = ('typo' if orig != corrected else info['hemisphere'])
            self._ch_tree.insert('', 'end', values=(
                orig, corrected,
                info['region'], info['hemisphere'], info['laterality'],
                info['naming_rule'], info.get('note', '')
            ), tags=(tag,))

        # All channels strip
        scalp_set = set(r['scalp_map'].keys())
        parts = []
        for ch in r.get('all_channels', []):
            if ch in scalp_set:
                parts.append(f"[{ch}]")  # bracketed = scalp
            else:
                parts.append(ch)

        self._all_ch_text.config(state='normal')
        self._all_ch_text.delete('1.0', 'end')
        self._all_ch_text.insert('end', "  ".join(parts))
        self._all_ch_text.config(state='disabled')

    def _clear(self):
        self._scanned = []
        self._csv_path = None
        for item in self._file_tree.get_children(): self._file_tree.delete(item)
        for row  in self._ch_tree.get_children():  self._ch_tree.delete(row)
        self._lbl_file.config(text="No file selected")
        self._lbl_info.config(text="")
        self._all_ch_text.config(state='normal')
        self._all_ch_text.delete('1.0', 'end')
        self._all_ch_text.config(state='disabled')
        self._status("Cleared.")

    def _export_csv(self):
        if not self._scanned:
            messagebox.showwarning("Nothing to export", "Scan at least one EDF first.")
            return

        save_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile=f"scalp_channel_map_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        if not save_path: return

        rows = []
        for r in self._scanned:
            if r['error']:
                rows.append({'File': r['path'].name, 'Error': r['error']})
                continue
            if not r['scalp_map']:
                rows.append({
                    'File': r['path'].name, 'N_scalp': 0,
                    'Mapping_type': 'None',
                    'Original_channel': '', 'Corrected_name': '',
                    'Region': '', 'Hemisphere': '', 'Laterality': '',
                    'Naming_rule': '', 'Note': '',
                    'Duration_s': round(r.get('duration_s', 0), 1),
                    'Sfreq_Hz': r.get('sfreq', 0),
                })
            else:
                n_s = r['n_scalp']
                mtype = ('Full 10-20' if n_s >= 16 else 'Partial')
                for orig, info in r['scalp_map'].items():
                    rows.append({
                        'File':             r['path'].name,
                        'Full_path':        str(r['path']),
                        'N_scalp':          n_s,
                        'Mapping_type':     mtype,
                        'Original_channel': orig,
                        'Corrected_name':   info['corrected_name'],
                        'Region':           info['region'],
                        'Hemisphere':       info['hemisphere'],
                        'Laterality':       info['laterality'],
                        'Naming_rule':      info['naming_rule'],
                        'Note':             info.get('note', ''),
                        'Duration_s':       round(r.get('duration_s', 0), 1),
                        'Sfreq_Hz':         r.get('sfreq', 0),
                    })

        df = pd.DataFrame(rows)
        df.to_csv(save_path, index=False)
        self._csv_path = save_path
        self._status(f"CSV saved: {save_path}  ({len(df)} rows)")
        messagebox.showinfo("Exported", f"CSV saved to:\n{save_path}")

    def _extract_edf(self):
        """Extract scalp-only EDF for selected file(s)."""
        sel = self._file_tree.selection()
        targets = []
        if sel:
            iid = sel[0]
            r = next((x for x in self._scanned if str(x['path']) == iid), None)
            if r and r['n_scalp'] > 0:
                targets = [r]
        if not targets:
            # Extract all with scalp channels
            targets = [r for r in self._scanned if r.get('n_scalp', 0) > 0]
        if not targets:
            messagebox.showwarning("Nothing to extract",
                                   "No files with scalp channels found.\n"
                                   "Select a file or scan a folder first.")
            return

        out_dir = filedialog.askdirectory(title="Choose output folder for scalp EDFs")
        if not out_dir: return
        out_dir = Path(out_dir)

        self._status(f"Extracting {len(targets)} scalp EDF(s)...")
        threading.Thread(target=self._run_extract,
                         args=(targets, out_dir), daemon=True).start()

    def _run_extract(self, targets, out_dir):
        msgs = []
        for i, r in enumerate(targets):
            self.after(0, lambda i=i: self._status(
                f"Extracting {i+1}/{len(targets)}: {targets[i]['path'].name}"))
            out_path = out_dir / (r['path'].stem + '_scalp_only.edf')
            msg = extract_scalp_edf(r['path'], out_path)
            msgs.append(f"{r['path'].name}: {msg}")

        self.after(0, lambda: self._status(
            f"Done. {len(targets)} EDF(s) extracted to {out_dir}"))
        self.after(0, lambda: messagebox.showinfo(
            "Extraction complete",
            "\n".join(msgs[:10]) + ("\n..." if len(msgs) > 10 else "")))

    # ── Batch Process Folder ───────────────────────────────────────────────────
    def _batch_process(self):
        """One-shot: pick input folder → scan all EDFs → extract scalp EDFs → save CSV."""
        folder = filedialog.askdirectory(title="Select folder containing EDF files")
        if not folder:
            return
        edfs = sorted(Path(folder).rglob("*.edf"))
        if not edfs:
            messagebox.showinfo("No EDFs found",
                                "No .edf files found in that folder (or subfolders).")
            return

        out_dir = filedialog.askdirectory(
            title=f"Select output folder  ({len(edfs)} EDFs found)")
        if not out_dir:
            return

        prog = _BatchProgressWin(self, total_files=len(edfs))
        threading.Thread(
            target=self._run_batch,
            args=(edfs, Path(out_dir), prog),
            daemon=True,
        ).start()

    def _run_batch(self, edfs: list, out_dir: Path, prog: '_BatchProgressWin'):
        n = len(edfs)

        # ── Phase 1: Scan ──────────────────────────────────────────────────────
        results = []
        for i, p in enumerate(edfs):
            self.after(0, lambda i=i: prog.update(
                phase="Scanning",
                step=i + 1, total=n,
                filename=edfs[i].name,
            ))
            r = scan_edf(p)
            results.append(r)
            self._scanned.append(r)
            self.after(0, self._refresh_file_list)

        # ── Phase 2: Extract scalp EDFs ────────────────────────────────────────
        to_extract = [r for r in results if r.get('n_scalp', 0) > 0]
        scalp_dir  = out_dir / "scalp_edfs"
        scalp_dir.mkdir(parents=True, exist_ok=True)

        extract_msgs = []
        for i, r in enumerate(to_extract):
            self.after(0, lambda i=i: prog.update(
                phase="Extracting",
                step=i + 1, total=len(to_extract),
                filename=to_extract[i]['path'].name,
            ))
            out_path = scalp_dir / (r['path'].stem + '_scalp_only.edf')
            msg = extract_scalp_edf(r['path'], out_path)
            extract_msgs.append(f"{r['path'].name}: {msg}")

        # ── Phase 3: Save CSV ──────────────────────────────────────────────────
        self.after(0, lambda: prog.update(
            phase="Saving CSV", step=1, total=1, filename=""))
        csv_path = out_dir / f"scalp_channel_map_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        rows = []
        for r in results:
            if r['error']:
                rows.append({'File': r['path'].name, 'Error': r['error']})
                continue
            if not r['scalp_map']:
                rows.append({'File': r['path'].name, 'N_scalp': 0,
                             'Mapping_type': 'None', 'Original_channel': '',
                             'Corrected_name': '', 'Region': '', 'Hemisphere': '',
                             'Laterality': '', 'Naming_rule': '', 'Note': '',
                             'Duration_s': round(r.get('duration_s', 0), 1),
                             'Sfreq_Hz': r.get('sfreq', 0)})
            else:
                n_s   = r['n_scalp']
                mtype = 'Full 10-20' if n_s >= 16 else 'Partial'
                for orig, info in r['scalp_map'].items():
                    rows.append({
                        'File':             r['path'].name,
                        'Full_path':        str(r['path']),
                        'N_scalp':          n_s,
                        'Mapping_type':     mtype,
                        'Original_channel': orig,
                        'Corrected_name':   info['corrected_name'],
                        'Region':           info['region'],
                        'Hemisphere':       info['hemisphere'],
                        'Laterality':       info['laterality'],
                        'Naming_rule':      info['naming_rule'],
                        'Note':             info.get('note', ''),
                        'Duration_s':       round(r.get('duration_s', 0), 1),
                        'Sfreq_Hz':         r.get('sfreq', 0),
                    })
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        # ── Done ───────────────────────────────────────────────────────────────
        n_ok  = sum(1 for r in results if not r['error'])
        n_err = sum(1 for r in results if r['error'])
        summary = (
            f"Scanned    : {n} file(s)  ({n_ok} OK, {n_err} error(s))\n"
            f"Scalp EDFs : {len(to_extract)} extracted → {scalp_dir}\n"
            f"CSV        : {csv_path.name}\n"
        )
        if extract_msgs:
            summary += "\nExtraction detail:\n" + "\n".join(extract_msgs[:20])
            if len(extract_msgs) > 20:
                summary += f"\n  … and {len(extract_msgs) - 20} more"

        self.after(0, lambda: prog.done(summary))
        self.after(0, lambda: self._status(
            f"Batch complete — {n_ok}/{n} files OK, "
            f"{len(to_extract)} scalp EDFs extracted, CSV saved"))

    def _status(self, msg):
        self._status_var.set(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Batch progress window
# ═══════════════════════════════════════════════════════════════════════════════

class _BatchProgressWin(tk.Toplevel):
    """Modal progress dialog shown during batch processing."""

    def __init__(self, parent, total_files: int):
        super().__init__(parent)
        self.title("Batch Processing")
        self.geometry("560x380")
        self.resizable(False, False)
        self.configure(bg='#f0f0f0')
        self.grab_set()                 # modal
        self.protocol("WM_DELETE_WINDOW", lambda: None)  # disable X during run

        # Phase label
        self._phase_var = tk.StringVar(value="Starting…")
        tk.Label(self, textvariable=self._phase_var, bg='#f0f0f0',
                 font=('Segoe UI', 10, 'bold')).pack(pady=(14, 2))

        # Progress bar
        self._pbar = ttk.Progressbar(self, orient='horizontal',
                                     length=500, mode='determinate')
        self._pbar.pack(pady=4)

        # Current filename
        self._file_var = tk.StringVar(value="")
        tk.Label(self, textvariable=self._file_var, bg='#f0f0f0',
                 font=('Segoe UI', 8), fg='#555', wraplength=520,
                 justify='left').pack(anchor='w', padx=28)

        # Scrollable log
        log_frame = tk.Frame(self, bg='#f0f0f0')
        log_frame.pack(fill='both', expand=True, padx=14, pady=6)
        sb = ttk.Scrollbar(log_frame, orient='vertical')
        self._log = tk.Text(log_frame, height=10, font=('Courier New', 8),
                            bg='#1e1e1e', fg='#d4d4d4', state='disabled',
                            yscrollcommand=sb.set, wrap='word')
        sb.config(command=self._log.yview)
        sb.pack(side='right', fill='y')
        self._log.pack(fill='both', expand=True)

        # Close button (disabled until done)
        self._close_btn = tk.Button(self, text="Close", state='disabled',
                                    bg='#27ae60', fg='white', relief='flat',
                                    padx=20, pady=4,
                                    font=('Segoe UI', 9, 'bold'),
                                    command=self.destroy)
        self._close_btn.pack(pady=(0, 12))

    # called from background thread via parent.after(0, …)
    def update(self, phase: str, step: int, total: int, filename: str):
        pct = int(step / max(total, 1) * 100)
        self._phase_var.set(f"{phase}  —  {step} / {total}  ({pct}%)")
        self._pbar['value'] = pct
        self._file_var.set(filename)
        self._append_log(f"[{phase}] {step}/{total}  {filename}")

    def done(self, summary: str):
        self._phase_var.set("✅  Done!")
        self._pbar['value'] = 100
        self._file_var.set("")
        self._append_log("\n" + "─" * 60 + "\n" + summary)
        self._close_btn.config(state='normal')
        self.protocol("WM_DELETE_WINDOW", self.destroy)  # re-enable X

    def _append_log(self, text: str):
        self._log.config(state='normal')
        self._log.insert('end', text + "\n")
        self._log.see('end')
        self._log.config(state='disabled')


# ── CLI / headless mode ────────────────────────────────────────────────────────
def run_headless(folder: str, out_csv: str = None, extract: bool = False,
                 out_dir: str = None):
    """
    Non-GUI mode: scan a folder, print results, save CSV, optionally extract EDFs.
    Usage:
        python edf_scalp_inspector.py --folder G:/path/to/edfs
                                      --csv results.csv
                                      --extract --outdir G:/scalp_edfs
    """
    folder_path = Path(folder)
    edfs = sorted(folder_path.rglob("*.edf"))
    print(f"Found {len(edfs)} EDF files in {folder_path}")

    results = []
    for i, edf in enumerate(edfs):
        print(f"  [{i+1}/{len(edfs)}] {edf.name}", end=' ')
        r = scan_edf(edf)
        results.append(r)
        if r['error']:
            print(f"ERROR: {r['error']}")
        else:
            print(f"→ {r['n_scalp']} scalp ch / {r['n_total']} total")
            for orig, info in r['scalp_map'].items():
                corr = info['corrected_name']
                arrow = f" → {corr}" if orig != corr else ""
                print(f"       {orig}{arrow}: {info['region']} ({info['hemisphere']})")

    # CSV
    csv_path = out_csv or f"scalp_channel_map_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    rows = []
    for r in results:
        if r['error'] or not r['scalp_map']:
            rows.append({'File': r['path'].name, 'N_scalp': r['n_scalp'],
                         'Error': r.get('error', ''), 'Original_channel': '',
                         'Corrected_name': '', 'Region': '', 'Hemisphere': '',
                         'Laterality': '', 'Naming_rule': '', 'Note': ''})
        else:
            for orig, info in r['scalp_map'].items():
                rows.append({'File': r['path'].name, 'N_scalp': r['n_scalp'],
                             'Mapping_type': 'Full 10-20' if r['n_scalp'] >= 16 else 'Partial',
                             'Original_channel': orig, 'Corrected_name': info['corrected_name'],
                             'Region': info['region'], 'Hemisphere': info['hemisphere'],
                             'Laterality': info['laterality'], 'Naming_rule': info['naming_rule'],
                             'Note': info.get('note', ''),
                             'Duration_s': round(r.get('duration_s', 0), 1),
                             'Sfreq_Hz': r.get('sfreq', 0)})
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nCSV saved: {csv_path}  ({len(rows)} rows)")

    # Extract EDFs
    if extract:
        out_dir_path = Path(out_dir) if out_dir else folder_path / "scalp_edfs"
        out_dir_path.mkdir(parents=True, exist_ok=True)
        to_extract = [r for r in results if r['n_scalp'] > 0]
        print(f"\nExtracting {len(to_extract)} scalp EDF(s) → {out_dir_path}")
        for r in to_extract:
            out_path = out_dir_path / (r['path'].stem + '_scalp_only.edf')
            msg = extract_scalp_edf(r['path'], out_path)
            print(f"  {r['path'].name}: {msg}")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    if '--folder' in sys.argv:
        # Headless CLI mode
        import argparse
        ap = argparse.ArgumentParser(description='EDF scalp channel inspector (headless)')
        ap.add_argument('--folder',  required=True, help='Folder to scan for EDF files')
        ap.add_argument('--csv',     default=None,  help='Output CSV path')
        ap.add_argument('--extract', action='store_true', help='Extract scalp-only EDFs')
        ap.add_argument('--outdir',  default=None,  help='Output folder for scalp EDFs')
        args = ap.parse_args()
        run_headless(args.folder, args.csv, args.extract, args.outdir)
    else:
        # GUI mode
        app = App()
        app.mainloop()
