"""
Dofus Resource Farming – GUI Manager
=====================================
Tkinter-based dark UI for managing maps, node positions, and transitions.
Press F8/F9 to capture coordinates while the window is open.
"""

import json
import os
import signal
import subprocess
import sys
import tkinter as tk
from tkinter import messagebox, simpledialog

import cv2
import mss
import numpy as np
import pyautogui
from pynput import keyboard

# ── Dark theme palette ──────────────────────────────────────────────────────
BG       = "#0f0f1a"
SURFACE  = "#1a1a2e"
SURFACE2 = "#252545"
PRIMARY  = "#7c3aed"
ACCENT   = "#06b6d4"
GREEN    = "#22c55e"
RED      = "#ef4444"
ORANGE   = "#f59e0b"
TEXT     = "#e2e8f0"
DIM      = "#64748b"

CONFIG_FILE = "fer_config.json"
FONT       = ("Segoe UI", 10)
FONT_BOLD  = ("Segoe UI", 10, "bold")
FONT_TITLE = ("Segoe UI", 16, "bold")
FONT_SM    = ("Segoe UI", 9, "bold")

DEFAULT_CONFIG = {
    "general": {
        "mining_duration": 5.0,
        "map_transition_duration": 4.0,
        "mouse_move_duration": 0.2,
        "node_check_delay": 0.5,
        "iron_hsv_target": [20, 27, 101],
        "iron_hsv_tolerance": 50,
        "collect_offset_x": 40,
        "collect_offset_y": 40,
        "failsafe": False,
    },
    "maps": [],
}


class FerGUI:
    """Main application window."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("⛏️ Dofus Farming Manager")
        self.root.geometry("800x650")
        self.root.configure(bg=BG)

        self.config = dict(DEFAULT_CONFIG)
        self.selected_map_idx: int | None = None
        self.capture_mode: str | None = None  # "node" | "transition" | None
        self.bot_process: subprocess.Popen | None = None

        self._load_config()
        self._build_ui()
        self._start_key_listener()

        # Check bot status periodically
        self._poll_bot_status()

    # ── Config I/O ──────────────────────────────────────────────────────────

    def _load_config(self):
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                self.config = json.load(f)

    def _save_config(self):
        try:
            self.config["general"]["mining_duration"] = float(self.var_mining.get())
        except ValueError:
            pass
        try:
            self.config["general"]["collect_offset_x"] = int(self.var_off_x.get())
        except ValueError:
            pass
        try:
            self.config["general"]["collect_offset_y"] = int(self.var_off_y.get())
        except ValueError:
            pass
        try:
            self.config["general"]["iron_hsv_tolerance"] = int(self.var_tolerance.get())
        except ValueError:
            pass
        try:
            self.config["general"]["map_transition_duration"] = float(self.var_trans_dur.get())
        except ValueError:
            pass
        self.config["general"]["failsafe"] = self.var_failsafe.get()
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2)

    # ── UI Construction ─────────────────────────────────────────────────────

    def _build_ui(self):
        # Header
        hdr = tk.Frame(self.root, bg=BG)
        hdr.pack(fill="x", padx=10, pady=10)
        tk.Label(hdr, text="⛏️ Dofus Farming Manager", font=FONT_TITLE, bg=BG, fg=TEXT).pack(side="left")

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=10, pady=5)

        # Left – Maps list
        maps_box = tk.LabelFrame(body, text="Maps", bg=SURFACE, fg=ACCENT, font=FONT_BOLD, bd=1)
        maps_box.pack(side="left", fill="y", padx=(0, 10))

        self.map_list = tk.Listbox(maps_box, bg=SURFACE2, fg=TEXT, selectbackground=PRIMARY, width=25, font=FONT)
        self.map_list.pack(fill="both", expand=True, padx=5, pady=5)
        self.map_list.bind("<<ListboxSelect>>", self._on_map_select)

        bf1 = tk.Frame(maps_box, bg=SURFACE)
        bf1.pack(fill="x", padx=5, pady=5)
        tk.Button(bf1, text="➕ Add", bg=SURFACE2, fg=TEXT, command=self._add_map).pack(side="left", expand=True, fill="x", padx=(0, 2))
        tk.Button(bf1, text="❌ Del", bg=SURFACE2, fg=TEXT, command=self._del_map).pack(side="right", expand=True, fill="x", padx=(2, 0))

        # Right – Nodes & Transition
        right = tk.Frame(body, bg=BG)
        right.pack(side="left", fill="both", expand=True)

        nodes_box = tk.LabelFrame(right, text="Nodes (selected map)", bg=SURFACE, fg=ACCENT, font=FONT_BOLD, bd=1)
        nodes_box.pack(fill="both", expand=True, pady=(0, 10))

        self.node_list = tk.Listbox(nodes_box, bg=SURFACE2, fg=TEXT, selectbackground=PRIMARY, font=FONT)
        self.node_list.pack(fill="both", expand=True, padx=5, pady=5)

        bf2 = tk.Frame(nodes_box, bg=SURFACE)
        bf2.pack(fill="x", padx=5, pady=5)
        self.btn_cap_node = tk.Button(bf2, text="📷 Capture Node (F8)", bg=PRIMARY, fg="white", font=FONT_SM, command=lambda: self._set_capture("node"))
        self.btn_cap_node.pack(side="left", expand=True, fill="x", padx=(0, 2))
        tk.Button(bf2, text="❌ Delete", bg=SURFACE2, fg=TEXT, command=self._del_node).pack(side="right", expand=True, fill="x", padx=(2, 0))

        trans_box = tk.LabelFrame(right, text="Transition Zone", bg=SURFACE, fg=ACCENT, font=FONT_BOLD, bd=1)
        trans_box.pack(fill="x")

        self.lbl_trans = tk.Label(trans_box, text="Not set", bg=SURFACE, fg=DIM, font=FONT)
        self.lbl_trans.pack(pady=5)

        bf3 = tk.Frame(trans_box, bg=SURFACE)
        bf3.pack(fill="x", padx=5, pady=5)
        self.btn_cap_trans = tk.Button(bf3, text="📷 Capture Transition (F9)", bg=ACCENT, fg="black", font=FONT_SM, command=lambda: self._set_capture("transition"))
        self.btn_cap_trans.pack(side="left", expand=True, fill="x", padx=(0, 2))
        tk.Button(bf3, text="❌ Del", bg=SURFACE2, fg=TEXT, command=self._del_trans).pack(side="right", padx=(2, 0))

        # ── Settings bar ────────────────────────────────────────────────────
        settings = tk.LabelFrame(self.root, text="Settings", bg=SURFACE, fg=ACCENT, font=FONT_BOLD, bd=1)
        settings.pack(fill="x", padx=10, pady=(5, 0))

        row1 = tk.Frame(settings, bg=SURFACE)
        row1.pack(fill="x", padx=5, pady=3)

        gen = self.config["general"]

        tk.Label(row1, text="Mining (s):", bg=SURFACE, fg=TEXT, font=FONT).pack(side="left")
        self.var_mining = tk.StringVar(value=str(gen.get("mining_duration", 14.0)))
        tk.Entry(row1, textvariable=self.var_mining, bg=SURFACE2, fg=TEXT, width=5).pack(side="left", padx=(2, 10))

        tk.Label(row1, text="Transition (s):", bg=SURFACE, fg=TEXT, font=FONT).pack(side="left")
        self.var_trans_dur = tk.StringVar(value=str(gen.get("map_transition_duration", 4.0)))
        tk.Entry(row1, textvariable=self.var_trans_dur, bg=SURFACE2, fg=TEXT, width=5).pack(side="left", padx=(2, 10))

        tk.Label(row1, text="Tolerance:", bg=SURFACE, fg=TEXT, font=FONT).pack(side="left")
        self.var_tolerance = tk.StringVar(value=str(gen.get("iron_hsv_tolerance", 50)))
        tk.Entry(row1, textvariable=self.var_tolerance, bg=SURFACE2, fg=TEXT, width=5).pack(side="left", padx=(2, 10))

        row2 = tk.Frame(settings, bg=SURFACE)
        row2.pack(fill="x", padx=5, pady=3)

        tk.Label(row2, text="Collect Offset X:", bg=SURFACE, fg=TEXT, font=FONT).pack(side="left")
        self.var_off_x = tk.StringVar(value=str(gen.get("collect_offset_x", 40)))
        tk.Entry(row2, textvariable=self.var_off_x, bg=SURFACE2, fg=TEXT, width=5).pack(side="left", padx=(2, 10))

        tk.Label(row2, text="Y:", bg=SURFACE, fg=TEXT, font=FONT).pack(side="left")
        self.var_off_y = tk.StringVar(value=str(gen.get("collect_offset_y", 40)))
        tk.Entry(row2, textvariable=self.var_off_y, bg=SURFACE2, fg=TEXT, width=5).pack(side="left", padx=(2, 10))

        self.var_failsafe = tk.BooleanVar(value=gen.get("failsafe", False))
        tk.Checkbutton(row2, text="Failsafe (corner stop)", variable=self.var_failsafe, bg=SURFACE, fg=TEXT, selectcolor=SURFACE2, activebackground=SURFACE, activeforeground=TEXT, font=FONT).pack(side="left", padx=(10, 0))

        # ── Bottom action bar ───────────────────────────────────────────────
        bar = tk.Frame(self.root, bg=BG)
        bar.pack(fill="x", padx=10, pady=10)

        tk.Button(bar, text="💾 Save", bg=SURFACE2, fg=TEXT, font=FONT_BOLD, command=self._save_config).pack(side="left")

        self.btn_stop = tk.Button(bar, text="⏹ Stop Bot", bg=RED, fg="white", font=FONT_BOLD, command=self._stop_bot, state="disabled")
        self.btn_stop.pack(side="right", padx=(5, 0))

        self.btn_run = tk.Button(bar, text="▶ Run Bot", bg=GREEN, fg="black", font=FONT_BOLD, command=self._run_bot)
        self.btn_run.pack(side="right")

        self._refresh_maps()

    # ── Refresh helpers ─────────────────────────────────────────────────────

    def _refresh_maps(self):
        self.map_list.delete(0, tk.END)
        for m in self.config["maps"]:
            self.map_list.insert(tk.END, m.get("map_name", "Unnamed"))
        self._refresh_details()

    def _refresh_details(self):
        self.node_list.delete(0, tk.END)
        self.lbl_trans.config(text="Not set", fg=DIM)

        if self.selected_map_idx is None or self.selected_map_idx >= len(self.config["maps"]):
            return

        m = self.config["maps"][self.selected_map_idx]
        for i, n in enumerate(m.get("nodes", []), 1):
            self.node_list.insert(tk.END, f"Node {i}: ({n['x']}, {n['y']})")

        t = m.get("transition_zone")
        if t:
            self.lbl_trans.config(text=f"X: {t['x']}  |  Y: {t['y']}", fg=GREEN)

    # ── Map / node CRUD ─────────────────────────────────────────────────────

    def _on_map_select(self, _event):
        sel = self.map_list.curselection()
        if sel:
            self.selected_map_idx = sel[0]
            self._refresh_details()

    def _add_map(self):
        name = simpledialog.askstring("Add Map", "Map name:")
        if not name:
            return
        self.config["maps"].append({"map_name": name, "nodes": []})
        self._save_config()
        self._refresh_maps()
        self.map_list.selection_set(tk.END)
        self.selected_map_idx = len(self.config["maps"]) - 1
        self._refresh_details()

    def _del_map(self):
        if self.selected_map_idx is not None and messagebox.askyesno("Confirm", "Delete this map?"):
            self.config["maps"].pop(self.selected_map_idx)
            self.selected_map_idx = None
            self._save_config()
            self._refresh_maps()

    def _del_node(self):
        if self.selected_map_idx is None:
            return
        sel = self.node_list.curselection()
        if sel:
            self.config["maps"][self.selected_map_idx]["nodes"].pop(sel[0])
            self._save_config()
            self._refresh_details()

    def _del_trans(self):
        if self.selected_map_idx is None:
            return
        self.config["maps"][self.selected_map_idx].pop("transition_zone", None)
        self._save_config()
        self._refresh_details()

    # ── Capture mode ────────────────────────────────────────────────────────

    def _set_capture(self, mode: str | None):
        if self.selected_map_idx is None:
            messagebox.showwarning("Warning", "Select a map first!")
            return
        self.capture_mode = mode
        if mode == "node":
            self.btn_cap_node.config(text="⏳ Press F8…", bg=RED)
            self.btn_cap_trans.config(text="📷 Capture Transition (F9)", bg=ACCENT)
        elif mode == "transition":
            self.btn_cap_trans.config(text="⏳ Press F9…", bg=RED)
            self.btn_cap_node.config(text="📷 Capture Node (F8)", bg=PRIMARY)

    @staticmethod
    def _pixel_hsv(x: int, y: int) -> list[int]:
        with mss.mss() as sct:
            shot = sct.grab({"left": x, "top": y, "width": 1, "height": 1})
            bgr = cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            return [int(c) for c in hsv[0][0]]

    def _on_key(self, key):
        if self.selected_map_idx is None:
            return

        if key == keyboard.Key.f8 and self.capture_mode == "node":
            x, y = pyautogui.position()
            hsv = self._pixel_hsv(x, y)
            self.config["maps"][self.selected_map_idx].setdefault("nodes", []).append({"x": x, "y": y, "hsv": hsv})
            self._save_config()
            self.root.after(0, self._refresh_details)
            self.root.after(0, lambda: self._set_capture(None))
            self.root.after(0, lambda: self.btn_cap_node.config(text="📷 Capture Node (F8)", bg=PRIMARY))

        elif key == keyboard.Key.f9 and self.capture_mode == "transition":
            x, y = pyautogui.position()
            self.config["maps"][self.selected_map_idx]["transition_zone"] = {"x": x, "y": y}
            self._save_config()
            self.root.after(0, self._refresh_details)
            self.root.after(0, lambda: self._set_capture(None))
            self.root.after(0, lambda: self.btn_cap_trans.config(text="📷 Capture Transition (F9)", bg=ACCENT))

    def _start_key_listener(self):
        self._listener = keyboard.Listener(on_press=self._on_key)
        self._listener.start()

    # ── Bot process management ──────────────────────────────────────────────

    def _run_bot(self):
        if self.bot_process and self.bot_process.poll() is None:
            messagebox.showinfo("Info", "Bot is already running!")
            return
        self._save_config()
        self.bot_process = subprocess.Popen(
            [sys.executable, "farm_fer.py"],
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
        self.btn_run.config(state="disabled", bg=DIM)
        self.btn_stop.config(state="normal")

    def _stop_bot(self):
        if self.bot_process and self.bot_process.poll() is None:
            self.bot_process.terminate()
            self.bot_process.wait(timeout=5)
        self.bot_process = None
        self.btn_run.config(state="normal", bg=GREEN)
        self.btn_stop.config(state="disabled")

    def _poll_bot_status(self):
        """Check every second if the bot process is still alive."""
        if self.bot_process and self.bot_process.poll() is not None:
            # Bot exited on its own
            self.bot_process = None
            self.btn_run.config(state="normal", bg=GREEN)
            self.btn_stop.config(state="disabled")
        self.root.after(1000, self._poll_bot_status)


if __name__ == "__main__":
    app = FerGUI()
    app.root.mainloop()
