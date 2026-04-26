"""
Screen Automation GUI
=====================
Visual interface for configuring and running screen automation.
- Two detection modes: Template Match or Color Detection
- Target image selection (browse file or capture region from screen)
- Color picker (eyedropper) for color-based mob detection
- Scan region limiter (restrict search to game area only)
- Action recording (captures mouse clicks & keyboard input with timing)
- Visual action list editor (add, delete, reorder)
- Run / Stop controls with live log
- Saves & loads JSON config

Hotkeys (work even when GUI is minimised):
    F6  — Start / Stop recording
    F7  — Stop automation run
"""

import json
import logging
import os
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

import cv2
import mss
import numpy as np
import pyautogui
from PIL import Image, ImageTk, ImageDraw
from pynput import mouse as pmouse, keyboard as pkeyboard

# ============================================================
# Constants
# ============================================================
CONFIG_FILE = "config.json"
IMAGES_DIR = "images"
TOGGLE_KEY = pkeyboard.Key.f6
STOP_RUN_KEY = pkeyboard.Key.f7

# --- Dark theme palette ---
BG       = "#0f0f1a"
SURFACE  = "#1a1a2e"
SURFACE2 = "#252545"
BORDER   = "#333355"
PRIMARY  = "#7c3aed"
ACCENT   = "#06b6d4"
GREEN    = "#22c55e"
RED      = "#ef4444"
AMBER    = "#f59e0b"
TEXT     = "#e2e8f0"
MUTED    = "#94a3b8"
DIM      = "#64748b"

FONT         = ("Segoe UI", 10)
FONT_BOLD    = ("Segoe UI", 10, "bold")
FONT_SMALL   = ("Segoe UI", 9)
FONT_TITLE   = ("Segoe UI", 14, "bold")
FONT_HEADING = ("Segoe UI", 11, "bold")
FONT_MONO    = ("Consolas", 9)

log = logging.getLogger("gui")


# ============================================================
# Helpers — pynput → pyautogui key names
# ============================================================
_KEY_MAP = {}

def _build_key_map():
    base = {
        pkeyboard.Key.enter: "enter", pkeyboard.Key.tab: "tab",
        pkeyboard.Key.space: "space", pkeyboard.Key.backspace: "backspace",
        pkeyboard.Key.delete: "delete", pkeyboard.Key.esc: "escape",
        pkeyboard.Key.up: "up", pkeyboard.Key.down: "down",
        pkeyboard.Key.left: "left", pkeyboard.Key.right: "right",
        pkeyboard.Key.home: "home", pkeyboard.Key.end: "end",
        pkeyboard.Key.page_up: "pageup", pkeyboard.Key.page_down: "pagedown",
        pkeyboard.Key.insert: "insert", pkeyboard.Key.caps_lock: "capslock",
        pkeyboard.Key.num_lock: "numlock", pkeyboard.Key.scroll_lock: "scrolllock",
        pkeyboard.Key.pause: "pause", pkeyboard.Key.menu: "apps",
    }
    for i in range(1, 13):
        fk = getattr(pkeyboard.Key, f"f{i}", None)
        if fk:
            base[fk] = f"f{i}"
    _KEY_MAP.update(base)

_build_key_map()

MODIFIER_KEYS = {
    pkeyboard.Key.ctrl_l: "ctrl", pkeyboard.Key.ctrl_r: "ctrl",
    pkeyboard.Key.shift: "shift", pkeyboard.Key.shift_r: "shift",
    pkeyboard.Key.alt_l: "alt", pkeyboard.Key.alt_r: "alt",
    pkeyboard.Key.alt_gr: "alt",
    pkeyboard.Key.cmd: "win", pkeyboard.Key.cmd_r: "win",
}


def pynput_key_to_str(key):
    if isinstance(key, pkeyboard.KeyCode):
        if key.char:
            return key.char.lower()
        if key.vk and 32 <= key.vk <= 126:
            return chr(key.vk).lower()
        return None
    return _KEY_MAP.get(key)


def is_modifier(key):
    return key in MODIFIER_KEYS


# ============================================================
# Template matching (multi-scale + grayscale)
# ============================================================

def find_template_multiscale(screen_bgr, template_bgr, threshold=0.65,
                              use_grayscale=True,
                              scales=(0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5)):
    if use_grayscale:
        screen = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2GRAY)
        templ = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    else:
        screen = screen_bgr
        templ = template_bgr

    best_val = -1
    best_loc = None
    best_scale = 1.0
    best_tw, best_th = templ.shape[1], templ.shape[0]

    for scale in scales:
        sw, sh = int(templ.shape[1] * scale), int(templ.shape[0] * scale)
        if sw < 10 or sh < 10 or sw > screen.shape[1] or sh > screen.shape[0]:
            continue
        resized = cv2.resize(templ, (sw, sh), interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(screen, resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val > best_val:
            best_val = max_val
            best_loc = max_loc
            best_scale = scale
            best_tw, best_th = sw, sh

    if best_val >= threshold and best_loc is not None:
        cx = best_loc[0] + best_tw // 2
        cy = best_loc[1] + best_th // 2
        return (cx, cy, best_val, best_scale)
    return None


# ============================================================
# Color detection (HSV-based)
# ============================================================

def find_color_clusters(screen_bgr, target_hsv, tolerance=25, min_area=200,
                        scan_region=None):
    """
    Find clusters of pixels matching target_hsv color within tolerance.
    Returns list of (cx, cy, area) sorted by area descending.
    """
    hsv = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2HSV)

    h_target, s_target, v_target = target_hsv

    # Build HSV range with tolerance
    h_tol = min(tolerance, 20)  # Hue wraps at 180, keep tight
    s_tol = tolerance * 2
    v_tol = tolerance * 2

    lower = np.array([max(0, h_target - h_tol),
                      max(0, s_target - s_tol),
                      max(0, v_target - v_tol)], dtype=np.uint8)
    upper = np.array([min(179, h_target + h_tol),
                      min(255, s_target + s_tol),
                      min(255, v_target + v_tol)], dtype=np.uint8)

    # Handle hue wrapping (e.g., red hues near 0/180)
    if h_target - h_tol < 0:
        mask1 = cv2.inRange(hsv, np.array([0, lower[1], lower[2]]), upper)
        mask2 = cv2.inRange(hsv, np.array([180 + h_target - h_tol, lower[1], lower[2]]),
                            np.array([179, upper[1], upper[2]]))
        mask = cv2.bitwise_or(mask1, mask2)
    elif h_target + h_tol > 179:
        mask1 = cv2.inRange(hsv, lower, np.array([179, upper[1], upper[2]]))
        mask2 = cv2.inRange(hsv, np.array([0, lower[1], lower[2]]),
                            np.array([h_target + h_tol - 180, upper[1], upper[2]]))
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        mask = cv2.inRange(hsv, lower, upper)

    # Apply scan region mask if specified
    if scan_region:
        region_mask = np.zeros(mask.shape, dtype=np.uint8)
        x, y, w, h = scan_region["x"], scan_region["y"], scan_region["w"], scan_region["h"]
        region_mask[y:y+h, x:x+w] = 255
        mask = cv2.bitwise_and(mask, region_mask)

    # Clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    clusters = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                clusters.append((cx, cy, int(area)))

    clusters.sort(key=lambda c: c[2], reverse=True)
    return clusters, mask


# ============================================================
# InputRecorder
# ============================================================
class InputRecorder:
    def __init__(self):
        self.events: list[dict] = []
        self.recording = False
        self.start_time = 0.0
        self.active_mods: set[str] = set()
        self._mouse_listener = None
        self._kb_listener = None
        self.on_stop_callback = None

    def start(self, on_stop=None):
        self.events.clear()
        self.active_mods.clear()
        self.recording = True
        self.start_time = time.time()
        self.on_stop_callback = on_stop
        self._mouse_listener = pmouse.Listener(on_click=self._on_click)
        self._kb_listener = pkeyboard.Listener(
            on_press=self._on_key_press, on_release=self._on_key_release)
        self._mouse_listener.start()
        self._kb_listener.start()

    def stop(self):
        self.recording = False
        if self._mouse_listener: self._mouse_listener.stop()
        if self._kb_listener: self._kb_listener.stop()

    def get_actions(self):
        if not self.events: return []
        actions = []
        prev_time = 0.0
        for evt in self.events:
            gap = evt["time"] - prev_time
            if gap > 0.05:
                actions.append({"type": "wait", "duration": round(gap, 2)})
            actions.append({k: v for k, v in evt.items() if k != "time"})
            prev_time = evt["time"]
        return actions

    def _elapsed(self):
        return time.time() - self.start_time

    def _on_click(self, x, y, button, pressed):
        if not self.recording or not pressed: return
        btn_map = {pmouse.Button.left: "left", pmouse.Button.right: "right",
                   pmouse.Button.middle: "middle"}
        self.events.append({"time": self._elapsed(), "type": "click",
                            "position": {"x": int(x), "y": int(y)},
                            "button": btn_map.get(button, "left")})

    def _on_key_press(self, key):
        if not self.recording: return
        if key == TOGGLE_KEY:
            self.stop()
            if self.on_stop_callback: self.on_stop_callback()
            return
        if is_modifier(key):
            self.active_mods.add(MODIFIER_KEYS[key]); return
        name = pynput_key_to_str(key)
        if name is None: return
        if self.active_mods:
            self.events.append({"time": self._elapsed(), "type": "hotkey",
                                "keys": sorted(self.active_mods) + [name]})
        else:
            self.events.append({"time": self._elapsed(), "type": "key", "value": name})

    def _on_key_release(self, key):
        if is_modifier(key):
            self.active_mods.discard(MODIFIER_KEYS.get(key))


# ============================================================
# RegionSelector — full-screen overlay for rectangle selection
# ============================================================
class RegionSelector:
    def __init__(self, root, callback, mode="capture"):
        """mode: 'capture' returns PIL Image crop, 'region' returns {x,y,w,h} dict"""
        self.callback = callback
        self.root = root
        self.mode = mode
        self.start_x = self.start_y = 0
        self.rect_id = None

        with mss.mss() as sct:
            monitor = sct.monitors[1]
            shot = sct.grab(monitor)
            self.screenshot = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
            self.monitor = monitor

        self.overlay = tk.Toplevel(root)
        self.overlay.attributes("-fullscreen", True)
        self.overlay.attributes("-topmost", True)
        self.overlay.configure(cursor="crosshair")

        dim = Image.blend(self.screenshot, Image.new("RGB", self.screenshot.size, (0, 0, 0)), 0.35)
        self._bg_photo = ImageTk.PhotoImage(dim)

        self.canvas = tk.Canvas(self.overlay, width=self.monitor["width"],
                                height=self.monitor["height"], highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.create_image(0, 0, anchor="nw", image=self._bg_photo)

        if mode == "capture":
            txt = "Click and drag to select target region  •  ESC to cancel"
        elif mode == "region":
            txt = "Click and drag to define SCAN REGION (game area only)  •  ESC to cancel"
        elif mode == "color":
            txt = "Click on a MOB to pick its color  •  ESC to cancel"
        else:
            txt = "Select area  •  ESC to cancel"

        self.canvas.create_text(self.monitor["width"] // 2, 30, text=txt,
                                fill="white", font=("Segoe UI", 14, "bold"))

        if mode == "color":
            self.canvas.bind("<ButtonPress-1>", self._on_color_pick)
            # Show live color under cursor
            self._color_indicator = self.canvas.create_rectangle(0, 0, 0, 0, outline="white", width=2)
            self._color_text = self.canvas.create_text(0, 0, text="", fill="white",
                                                        font=("Segoe UI", 11, "bold"))
            self.canvas.bind("<Motion>", self._on_color_hover)
        else:
            self.canvas.bind("<ButtonPress-1>", self._on_press)
            self.canvas.bind("<B1-Motion>", self._on_drag)
            self.canvas.bind("<ButtonRelease-1>", self._on_release)

        self.overlay.bind("<Escape>", lambda e: self._cancel())

    def _on_press(self, event):
        self.start_x, self.start_y = event.x, event.y
        if self.rect_id: self.canvas.delete(self.rect_id)

    def _on_drag(self, event):
        if self.rect_id: self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(
            self.start_x, self.start_y, event.x, event.y,
            outline="#7c3aed", width=2, dash=(6, 3))

    def _on_release(self, event):
        x1, y1 = min(self.start_x, event.x), min(self.start_y, event.y)
        x2, y2 = max(self.start_x, event.x), max(self.start_y, event.y)
        if x2 - x1 < 10 or y2 - y1 < 10: return
        self.overlay.destroy()
        if self.mode == "capture":
            self.callback(self.screenshot.crop((x1, y1, x2, y2)))
        elif self.mode == "region":
            self.callback({"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1})

    def _on_color_hover(self, event):
        """Show color preview under cursor."""
        x, y = event.x, event.y
        try:
            pixel = self.screenshot.getpixel((x, y))
            r, g, b = pixel[0], pixel[1], pixel[2]
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            # Show color swatch
            self.canvas.coords(self._color_indicator, x + 20, y - 30, x + 50, y)
            self.canvas.itemconfig(self._color_indicator, fill=hex_color, outline="white")
            self.canvas.coords(self._color_text, x + 60, y - 15)
            self.canvas.itemconfig(self._color_text, text=f"RGB({r},{g},{b})")
        except Exception:
            pass

    def _on_color_pick(self, event):
        """Pick color from clicked pixel."""
        x, y = event.x, event.y
        try:
            pixel = self.screenshot.getpixel((x, y))
            r, g, b = pixel[0], pixel[1], pixel[2]
            # Convert RGB to HSV (OpenCV format)
            bgr = np.uint8([[[b, g, r]]])
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0][0]
            self.overlay.destroy()
            self.callback({
                "rgb": [int(r), int(g), int(b)],
                "hsv": [int(hsv[0]), int(hsv[1]), int(hsv[2])],
            })
        except Exception:
            pass

    def _cancel(self):
        self.overlay.destroy()
        self.root.deiconify()


# ============================================================
# AutomationGUI
# ============================================================
class AutomationGUI:

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Screen Automation")
        self.root.geometry("800x960")
        self.root.configure(bg=BG)
        self.root.minsize(750, 850)

        # State
        self.actions: list[dict] = []
        self.recorder = InputRecorder()
        self.is_recording = False
        self.is_running = False
        self.run_thread = None
        self.stop_event = threading.Event()
        self.global_kb_listener = None

        # Color detection state
        self.picked_color_hsv = None  # (H, S, V) in OpenCV range
        self.picked_color_rgb = None  # (R, G, B)
        self.scan_region = None       # {"x","y","w","h"} or None

        self._build_ui()
        self._load_config_if_exists()
        self._start_global_hotkeys()

    # ----------------------------------------------------------------
    # UI
    # ----------------------------------------------------------------

    def _build_ui(self):
        container = tk.Frame(self.root, bg=BG)
        container.pack(fill="both", expand=True, padx=16, pady=12)

        title_row = tk.Frame(container, bg=BG)
        title_row.pack(fill="x", pady=(0, 10))
        tk.Label(title_row, text="⚡ Screen Automation", font=FONT_TITLE,
                 bg=BG, fg=TEXT).pack(side="left")
        tk.Label(title_row, text="F6: Record  │  F7: Stop Run", font=FONT_SMALL,
                 bg=BG, fg=DIM).pack(side="right")

        self._build_detection_section(container)
        self._build_actions_section(container)
        self._build_settings_section(container)
        self._build_controls(container)
        self._build_log_section(container)

    def _section(self, parent, title):
        f = tk.LabelFrame(parent, text=f"  {title}  ", font=FONT_HEADING,
                          bg=SURFACE, fg=ACCENT, bd=1, relief="solid",
                          highlightbackground=BORDER, highlightthickness=1, padx=12, pady=8)
        f.pack(fill="x", pady=(0, 8))
        return f

    def _btn(self, parent, text, cmd, color=PRIMARY, fg_="white", **kw):
        return tk.Button(parent, text=text, command=cmd, bg=color, fg=fg_,
                         activebackground=color, font=FONT_BOLD, relief="flat",
                         cursor="hand2", padx=12, pady=5, bd=0, **kw)

    def _entry(self, parent, var=None, width=10):
        e = tk.Entry(parent, bg=SURFACE2, fg=TEXT, insertbackground=TEXT,
                     font=FONT, relief="flat", bd=0, highlightthickness=1,
                     highlightbackground=BORDER, highlightcolor=PRIMARY, width=width)
        if var: e.configure(textvariable=var)
        return e

    # ---- Detection section (replaces old Target) -------------------

    def _build_detection_section(self, parent):
        sec = self._section(parent, "🎯 Detection")

        # Row 0: Mode selection
        row0 = tk.Frame(sec, bg=SURFACE); row0.pack(fill="x", pady=(0, 6))
        tk.Label(row0, text="Mode:", font=FONT_BOLD, bg=SURFACE, fg=TEXT).pack(side="left")
        self.detect_mode_var = tk.StringVar(value="color")
        mode_combo = ttk.Combobox(row0, textvariable=self.detect_mode_var, width=18,
                                   values=["color", "template"],
                                   state="readonly")
        mode_combo.pack(side="left", padx=(8, 16))
        mode_combo.bind("<<ComboboxSelected>>", lambda e: self._on_mode_change())

        tk.Label(row0, text="On Find:", font=FONT, bg=SURFACE, fg=TEXT).pack(side="left")
        self.action_on_find_var = tk.StringVar(value="right_click")
        ttk.Combobox(row0, textvariable=self.action_on_find_var, width=12,
                     values=["click", "right_click", "double_click", "none"],
                     state="readonly").pack(side="left", padx=(8, 0))

        # --- Color detection frame ---
        self.color_frame = tk.Frame(sec, bg=SURFACE)

        crow1 = tk.Frame(self.color_frame, bg=SURFACE); crow1.pack(fill="x", pady=(0, 6))
        self._btn(crow1, "🎨 Pick Color", self._pick_color, color=ACCENT, fg_="#000").pack(side="left", padx=(0, 4))
        self._btn(crow1, "📐 Set Scan Region", self._set_scan_region, color=PRIMARY).pack(side="left", padx=(0, 4))
        self._btn(crow1, "🔍 Test", self._test_detection, color=AMBER, fg_="#000").pack(side="left", padx=(0, 4))

        # Region label
        self.region_label = tk.Label(crow1, text="Region: Full screen", font=FONT_SMALL,
                                      bg=SURFACE, fg=MUTED)
        self.region_label.pack(side="left", padx=(8, 0))

        crow2 = tk.Frame(self.color_frame, bg=SURFACE); crow2.pack(fill="x", pady=(0, 4))

        # Color preview swatch
        self.color_swatch = tk.Canvas(crow2, width=30, height=30, bg=SURFACE2,
                                       highlightthickness=1, highlightbackground=BORDER)
        self.color_swatch.pack(side="left", padx=(0, 8))

        self.color_info_label = tk.Label(crow2, text="No color picked — click 🎨 Pick Color",
                                          font=FONT, bg=SURFACE, fg=DIM)
        self.color_info_label.pack(side="left")

        crow3 = tk.Frame(self.color_frame, bg=SURFACE); crow3.pack(fill="x", pady=(0, 4))
        tk.Label(crow3, text="Tolerance:", font=FONT, bg=SURFACE, fg=TEXT).pack(side="left")
        self.color_tol_var = tk.IntVar(value=25)
        tk.Scale(crow3, from_=5, to=60, resolution=1, orient="horizontal",
                 variable=self.color_tol_var, bg=SURFACE, fg=TEXT,
                 troughcolor=SURFACE2, highlightthickness=0, length=150,
                 font=FONT_SMALL, activebackground=PRIMARY,
                 sliderrelief="flat").pack(side="left", padx=(8, 16))

        tk.Label(crow3, text="Min area:", font=FONT, bg=SURFACE, fg=TEXT).pack(side="left")
        self.min_area_var = tk.StringVar(value="300")
        self._entry(crow3, self.min_area_var, 6).pack(side="left", padx=(8, 0))
        tk.Label(crow3, text="px²", font=FONT_SMALL, bg=SURFACE, fg=MUTED).pack(side="left")

        tk.Label(self.color_frame,
                 text="💡 Click 🎨 Pick Color then click on a MOB in your game. The script will find all similar colored objects.",
                 font=FONT_SMALL, bg=SURFACE, fg=DIM, wraplength=700, justify="left").pack(fill="x", pady=(4, 0))

        self.color_frame.pack(fill="x")

        # --- Template detection frame ---
        self.template_frame = tk.Frame(sec, bg=SURFACE)

        trow1 = tk.Frame(self.template_frame, bg=SURFACE); trow1.pack(fill="x", pady=(0, 6))
        tk.Label(trow1, text="Image:", font=FONT, bg=SURFACE, fg=TEXT).pack(side="left")
        self.target_path_var = tk.StringVar(value="./images/target.png")
        self._entry(trow1, self.target_path_var, 28).pack(side="left", padx=8, fill="x", expand=True)
        self._btn(trow1, "📂", self._browse_target).pack(side="left", padx=(0, 4))
        self._btn(trow1, "📷 Capture", self._capture_target, color=ACCENT, fg_="#000").pack(side="left", padx=(0, 4))
        self._btn(trow1, "🔍 Test", self._test_detection, color=AMBER, fg_="#000").pack(side="left")

        trow2 = tk.Frame(self.template_frame, bg=SURFACE); trow2.pack(fill="x", pady=(0, 4))
        tk.Label(trow2, text="Confidence:", font=FONT, bg=SURFACE, fg=TEXT).pack(side="left")
        self.confidence_var = tk.DoubleVar(value=0.65)
        tk.Scale(trow2, from_=0.40, to=1.0, resolution=0.05, orient="horizontal",
                 variable=self.confidence_var, bg=SURFACE, fg=TEXT,
                 troughcolor=SURFACE2, highlightthickness=0, length=150,
                 font=FONT_SMALL, activebackground=PRIMARY,
                 sliderrelief="flat").pack(side="left", padx=(8, 16))

        self.grayscale_var = tk.BooleanVar(value=True)
        tk.Checkbutton(trow2, text="Grayscale", variable=self.grayscale_var, bg=SURFACE, fg=TEXT,
                       selectcolor=SURFACE2, activebackground=SURFACE, activeforeground=TEXT,
                       font=FONT_SMALL).pack(side="left")

        self.preview_frame = tk.Frame(self.template_frame, bg=SURFACE2, width=200, height=60)
        self.preview_frame.pack(fill="x", pady=(6, 0))
        self.preview_frame.pack_propagate(False)
        self.preview_label = tk.Label(self.preview_frame, text="No target image loaded",
                                      font=FONT_SMALL, bg=SURFACE2, fg=DIM)
        self.preview_label.pack(expand=True)

        # Initially hide template frame (color mode is default)
        self._on_mode_change()

    def _on_mode_change(self):
        mode = self.detect_mode_var.get()
        if mode == "color":
            self.template_frame.pack_forget()
            self.color_frame.pack(fill="x")
        else:
            self.color_frame.pack_forget()
            self.template_frame.pack(fill="x")
            self._update_preview()

    # ---- Actions section -------------------------------------------

    def _build_actions_section(self, parent):
        sec = self._section(parent, "🎬 Actions (after detection)")

        btn_row = tk.Frame(sec, bg=SURFACE); btn_row.pack(fill="x", pady=(0, 6))
        self.record_btn = self._btn(btn_row, "🔴 Record (F6)", self._toggle_record, color=RED)
        self.record_btn.pack(side="left", padx=(0, 4))
        self._btn(btn_row, "🗑 Clear", self._clear_actions, color=SURFACE2, fg_=TEXT).pack(side="left", padx=(0, 4))
        self._btn(btn_row, "➕ Add", self._add_action_dialog, color=SURFACE2, fg_=TEXT).pack(side="left")
        self.record_status = tk.Label(btn_row, text="", font=FONT_SMALL, bg=SURFACE, fg=AMBER)
        self.record_status.pack(side="right")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("A.Treeview", background=SURFACE2, foreground=TEXT,
                        fieldbackground=SURFACE2, font=FONT_SMALL, rowheight=24)
        style.configure("A.Treeview.Heading", background=SURFACE, foreground=ACCENT, font=FONT_BOLD)
        style.map("A.Treeview", background=[("selected", PRIMARY)])

        tree_frame = tk.Frame(sec, bg=SURFACE); tree_frame.pack(fill="both", expand=True)
        cols = ("type", "details", "delay")
        self.tree = ttk.Treeview(tree_frame, columns=cols, show="headings", height=7, style="A.Treeview")
        self.tree.heading("type", text="Type"); self.tree.column("type", width=80, minwidth=60)
        self.tree.heading("details", text="Details"); self.tree.column("details", width=380, minwidth=200)
        self.tree.heading("delay", text="Delay"); self.tree.column("delay", width=80, minwidth=60)
        sb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        edit_row = tk.Frame(sec, bg=SURFACE); edit_row.pack(fill="x", pady=(6, 0))
        self._btn(edit_row, "❌ Del", self._del_action, color=SURFACE2, fg_=TEXT).pack(side="left", padx=(0, 4))
        self._btn(edit_row, "⬆", self._move_up, color=SURFACE2, fg_=TEXT).pack(side="left", padx=(0, 4))
        self._btn(edit_row, "⬇", self._move_down, color=SURFACE2, fg_=TEXT).pack(side="left")

    # ---- Settings ---------------------------------------------------

    def _build_settings_section(self, parent):
        sec = self._section(parent, "⚙️ Settings")
        row = tk.Frame(sec, bg=SURFACE); row.pack(fill="x")

        self.loop_var = tk.BooleanVar(value=True)
        tk.Checkbutton(row, text="Loop", variable=self.loop_var, bg=SURFACE, fg=TEXT,
                       selectcolor=SURFACE2, activebackground=SURFACE, activeforeground=TEXT,
                       font=FONT).pack(side="left")

        tk.Label(row, text="  Delay:", font=FONT, bg=SURFACE, fg=TEXT).pack(side="left")
        self.delay_var = tk.StringVar(value="5.0")
        self._entry(row, self.delay_var, 5).pack(side="left", padx=(4, 0))
        tk.Label(row, text="s", font=FONT, bg=SURFACE, fg=MUTED).pack(side="left")

        tk.Label(row, text="  Scan:", font=FONT, bg=SURFACE, fg=TEXT).pack(side="left", padx=(16, 0))
        self.scan_var = tk.StringVar(value="0.5")
        self._entry(row, self.scan_var, 5).pack(side="left", padx=(4, 0))
        tk.Label(row, text="s", font=FONT, bg=SURFACE, fg=MUTED).pack(side="left")

        tk.Label(row, text="  Max:", font=FONT, bg=SURFACE, fg=TEXT).pack(side="left", padx=(16, 0))
        self.maxloop_var = tk.StringVar(value="0")
        self._entry(row, self.maxloop_var, 4).pack(side="left", padx=(4, 0))
        tk.Label(row, text="(0=∞)", font=FONT_SMALL, bg=SURFACE, fg=DIM).pack(side="left", padx=(4, 0))

    # ---- Controls ---------------------------------------------------

    def _build_controls(self, parent):
        f = tk.Frame(parent, bg=BG); f.pack(fill="x", pady=(4, 6))
        self.run_btn = self._btn(f, "▶  Run", self._start_run, color=GREEN, fg_="#000")
        self.run_btn.pack(side="left", padx=(0, 6))
        self.stop_btn = self._btn(f, "⏹  Stop (F7)", self._stop_run, color=RED)
        self.stop_btn.pack(side="left", padx=(0, 6))
        self.stop_btn.configure(state="disabled")
        self._btn(f, "💾 Save", self._save_config, color=SURFACE2, fg_=TEXT).pack(side="right", padx=(6, 0))
        self._btn(f, "📂 Load", self._load_config_dialog, color=SURFACE2, fg_=TEXT).pack(side="right")

    # ---- Log --------------------------------------------------------

    def _build_log_section(self, parent):
        sec = self._section(parent, "📋 Log")
        self.log_text = tk.Text(sec, height=5, bg=SURFACE2, fg=TEXT, font=FONT_MONO,
                                relief="flat", bd=0, insertbackground=TEXT, wrap="word",
                                highlightthickness=1, highlightbackground=BORDER)
        self.log_text.pack(fill="both", expand=True)
        self.log_text.configure(state="disabled")

    def _log(self, msg, level="INFO"):
        ts = time.strftime("%H:%M:%S")
        icon = {"INFO": "ℹ️", "SUCCESS": "✅", "WARNING": "⚠️",
                "ERROR": "❌", "RECORD": "🔴"}.get(level, "")
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"{ts} {icon} {msg}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    # ----------------------------------------------------------------
    # Color picking
    # ----------------------------------------------------------------

    def _pick_color(self):
        self.root.withdraw()
        self.root.after(500, lambda: RegionSelector(self.root, self._on_color_picked, mode="color"))

    def _on_color_picked(self, color_data):
        self.root.deiconify()
        self.picked_color_rgb = tuple(color_data["rgb"])
        self.picked_color_hsv = tuple(color_data["hsv"])
        r, g, b = self.picked_color_rgb
        h, s, v = self.picked_color_hsv
        hex_c = f"#{r:02x}{g:02x}{b:02x}"

        self.color_swatch.configure(bg=hex_c)
        self.color_info_label.configure(
            text=f"RGB({r},{g},{b})  HSV({h},{s},{v})  {hex_c}",
            fg=TEXT
        )
        self._log(f"Color picked: RGB({r},{g},{b}) HSV({h},{s},{v})", "SUCCESS")

    def _set_scan_region(self):
        self.root.withdraw()
        self.root.after(500, lambda: RegionSelector(self.root, self._on_region_set, mode="region"))

    def _on_region_set(self, region):
        self.root.deiconify()
        self.scan_region = region
        self.region_label.configure(
            text=f"Region: ({region['x']},{region['y']}) {region['w']}×{region['h']}",
            fg=GREEN
        )
        self._log(f"Scan region set: {region['w']}×{region['h']} at ({region['x']},{region['y']})", "SUCCESS")

    # ----------------------------------------------------------------
    # Target (template mode)
    # ----------------------------------------------------------------

    def _browse_target(self):
        p = filedialog.askopenfilename(title="Select Target Image", initialdir=IMAGES_DIR,
                                       filetypes=[("Images", "*.png *.jpg *.bmp"), ("All", "*.*")])
        if p:
            try: self.target_path_var.set(f"./{os.path.relpath(p)}")
            except ValueError: self.target_path_var.set(p)
            self._update_preview()
            self._log(f"Target: {self.target_path_var.get()}")

    def _capture_target(self):
        self.root.withdraw()
        self.root.after(500, lambda: RegionSelector(self.root, self._on_region_captured, mode="capture"))

    def _on_region_captured(self, crop):
        self.root.deiconify()
        os.makedirs(IMAGES_DIR, exist_ok=True)
        fname = f"capture_{int(time.time())}.png"
        fpath = os.path.join(IMAGES_DIR, fname)
        crop.save(fpath)
        self.target_path_var.set(f"./{fpath}")
        self._update_preview()
        self._log(f"Captured target → {fpath}", "SUCCESS")

    def _update_preview(self):
        path = self.target_path_var.get()
        try:
            img = Image.open(path)
            img.thumbnail((780, 55), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.preview_label.configure(image=photo, text="")
            self.preview_label._photo = photo
        except Exception:
            self.preview_label.configure(image="", text="⚠️ Could not load image")

    # ----------------------------------------------------------------
    # Test detection (visual debug)
    # ----------------------------------------------------------------

    def _test_detection(self):
        mode = self.detect_mode_var.get()
        self._log(f"Testing {mode} detection...")

        with mss.mss() as sct:
            mon = sct.monitors[1]
            shot = sct.grab(mon)
            screen = cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)

        if mode == "color":
            self._test_color(screen)
        else:
            self._test_template(screen)

    def _test_color(self, screen):
        if self.picked_color_hsv is None:
            self._log("No color picked! Click 🎨 Pick Color first.", "ERROR")
            return

        tol = self.color_tol_var.get()
        min_area = int(self.min_area_var.get() or 300)
        clusters, mask = find_color_clusters(
            screen, self.picked_color_hsv,
            tolerance=tol, min_area=min_area,
            scan_region=self.scan_region
        )

        display = screen.copy()

        # Draw scan region if set
        if self.scan_region:
            r = self.scan_region
            cv2.rectangle(display, (r["x"], r["y"]),
                          (r["x"]+r["w"], r["y"]+r["h"]), (255, 255, 0), 2)

        # Draw the color mask as an overlay
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_colored[mask > 0] = [0, 255, 0]  # Green overlay
        display = cv2.addWeighted(display, 0.8, mask_colored, 0.4, 0)

        if clusters:
            for i, (cx, cy, area) in enumerate(clusters[:10]):
                color = (0, 255, 0) if i == 0 else (0, 200, 200)
                cv2.drawMarker(display, (cx, cy), color, cv2.MARKER_CROSS, 30, 3)
                cv2.circle(display, (cx, cy), int(area**0.5 / 2), color, 2)
                label = f"#{i+1} area={area}"
                cv2.putText(display, label, (cx + 15, cy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            best = clusters[0]
            self._log(f"✅ Found {len(clusters)} cluster(s). Best: ({best[0]},{best[1]}) area={best[2]}", "SUCCESS")
        else:
            self._log(f"❌ No color clusters found. Try increasing tolerance or picking a different color.", "WARNING")

        self._show_debug_window(display, f"Color Detection — {len(clusters)} cluster(s) found")

    def _test_template(self, screen):
        tp = self.target_path_var.get()
        if not tp or not os.path.exists(tp):
            self._log("No target image!", "ERROR"); return
        template = cv2.imread(tp, cv2.IMREAD_COLOR)
        if template is None:
            self._log(f"Cannot read: {tp}", "ERROR"); return

        conf = self.confidence_var.get()
        use_gray = self.grayscale_var.get()
        match = find_template_multiscale(screen, template, threshold=conf, use_grayscale=use_gray)

        display = screen.copy()
        th, tw = template.shape[:2]

        if match:
            cx, cy, best_val, best_scale = match
            stw, sth = int(tw * best_scale), int(th * best_scale)
            x1, y1 = cx - stw // 2, cy - sth // 2
            cv2.rectangle(display, (x1, y1), (x1+stw, y1+sth), (0, 255, 0), 3)
            cv2.drawMarker(display, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            cv2.putText(display, f"MATCH conf={best_val:.3f} scale={best_scale:.1f}x",
                        (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            self._log(f"✅ Match! conf={best_val:.3f} scale={best_scale:.1f}x at ({cx},{cy})", "SUCCESS")
        else:
            if use_gray:
                sc = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
                tm = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            else:
                sc, tm = screen, template
            result = cv2.matchTemplate(sc, tm, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            cv2.rectangle(display, max_loc, (max_loc[0]+tw, max_loc[1]+th), (0, 0, 255), 3)
            cv2.putText(display, f"BEST (no match) conf={max_val:.3f} < threshold={conf:.2f}",
                        (max_loc[0], max_loc[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            self._log(f"❌ No match. Best conf={max_val:.3f} (needs >={conf:.2f})", "WARNING")

        self._show_debug_window(display, "Template Match Result")

    def _show_debug_window(self, screen_bgr, title="Debug"):
        win = tk.Toplevel(self.root)
        win.title(f"🔍 {title}")
        win.configure(bg=BG)
        win.attributes("-topmost", True)

        h, w = screen_bgr.shape[:2]
        scale = min(1100 / w, 700 / h, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(screen_bgr, (new_w, new_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(img)

        canvas = tk.Canvas(win, width=new_w, height=new_h, bg=BG, highlightthickness=0)
        canvas.pack(padx=8, pady=8)
        canvas.create_image(0, 0, anchor="nw", image=photo)
        canvas._photo = photo

        bottom = tk.Frame(win, bg=SURFACE)
        bottom.pack(fill="x", padx=8, pady=(0, 8))
        tip = "Green ✅ = detected  •  Red ❌ = best guess (too low confidence)  •  Yellow = scan region"
        tk.Label(bottom, text=tip, font=FONT_SMALL, bg=SURFACE, fg=DIM,
                 wraplength=700).pack(side="left", padx=8, fill="x", expand=True)
        self._btn(bottom, "Close", win.destroy, color=SURFACE2, fg_=TEXT).pack(side="right", padx=8)

    # ----------------------------------------------------------------
    # Recording
    # ----------------------------------------------------------------

    def _toggle_record(self):
        if self.is_recording: self._stop_record()
        else: self._start_record()

    def _start_record(self):
        if self.is_running:
            self._log("Cannot record while running", "WARNING"); return
        self.is_recording = True
        self.record_btn.configure(text="⏹ Stop Recording", bg=AMBER)
        self.record_status.configure(text="🔴 Recording… press F6 when done")
        self._log("Recording started — perform your actions, press F6 when done", "RECORD")
        self.root.iconify()
        self.root.after(800, self._actually_start_record)

    def _actually_start_record(self):
        self.recorder.start(on_stop=lambda: self.root.after(0, self._stop_record))

    def _stop_record(self):
        if not self.is_recording: return
        self.is_recording = False
        self.recorder.stop()
        self.root.deiconify(); self.root.lift()
        self.record_btn.configure(text="🔴 Record (F6)", bg=RED)
        self.record_status.configure(text="")
        new = self.recorder.get_actions()
        if new:
            self.actions.extend(new)
            self._refresh_tree()
            self._log(f"Recorded {len(new)} actions", "SUCCESS")
        else:
            self._log("No actions recorded", "WARNING")

    def _start_global_hotkeys(self):
        def on_press(key):
            if key == TOGGLE_KEY and not self.is_recording:
                self.root.after(0, self._toggle_record)
            elif key == STOP_RUN_KEY and self.is_running:
                self.root.after(0, self._stop_run)
        self.global_kb_listener = pkeyboard.Listener(on_press=on_press)
        self.global_kb_listener.daemon = True
        self.global_kb_listener.start()

    # ----------------------------------------------------------------
    # Action list
    # ----------------------------------------------------------------

    def _refresh_tree(self):
        self.tree.delete(*self.tree.get_children())
        for i, a in enumerate(self.actions):
            t = a.get("type", "?")
            d = self._detail(a)
            dl = f'{a["duration"]}s' if t == "wait" else ""
            self.tree.insert("", "end", iid=str(i), values=(t, d, dl))

    @staticmethod
    def _detail(a):
        t = a["type"]
        if t in ("click", "double_click", "right_click"):
            p = a.get("position", {})
            extra = f' {a["button"]}' if "button" in a else ""
            return f'({p.get("x","?")}, {p.get("y","?")}){extra}'
        if t == "key": return a.get("value", "?")
        if t == "hotkey": return " + ".join(a.get("keys", []))
        if t == "type_text": return a.get("value", "")[:50]
        if t == "wait": return f'{a.get("duration",0)}s pause'
        if t == "move":
            p = a.get("position", {}); return f'({p.get("x","?")}, {p.get("y","?")})'
        if t == "scroll": return f'{a.get("amount",0)}'
        return str(a)

    def _clear_actions(self):
        if self.actions and messagebox.askyesno("Clear", "Clear all recorded actions?"):
            self.actions.clear(); self._refresh_tree(); self._log("Actions cleared")

    def _del_action(self):
        s = self.tree.selection()
        if s: self.actions.pop(int(s[0])); self._refresh_tree()

    def _move_up(self):
        s = self.tree.selection()
        if not s: return
        i = int(s[0])
        if i > 0:
            self.actions[i], self.actions[i-1] = self.actions[i-1], self.actions[i]
            self._refresh_tree(); self.tree.selection_set(str(i-1))

    def _move_down(self):
        s = self.tree.selection()
        if not s: return
        i = int(s[0])
        if i < len(self.actions)-1:
            self.actions[i], self.actions[i+1] = self.actions[i+1], self.actions[i]
            self._refresh_tree(); self.tree.selection_set(str(i+1))

    def _add_action_dialog(self):
        dlg = tk.Toplevel(self.root); dlg.title("Add Action"); dlg.geometry("420x300")
        dlg.configure(bg=SURFACE); dlg.transient(self.root); dlg.grab_set()

        tk.Label(dlg, text="Type:", font=FONT, bg=SURFACE, fg=TEXT).pack(anchor="w", padx=12, pady=(12,4))
        tv = tk.StringVar(value="key")
        ttk.Combobox(dlg, textvariable=tv, width=20,
                     values=["click","right_click","double_click","key","hotkey","type_text","wait","scroll","move"],
                     state="readonly").pack(anchor="w", padx=12)

        tk.Label(dlg, text="Value / Key (for hotkey use +):", font=FONT, bg=SURFACE, fg=TEXT).pack(anchor="w", padx=12, pady=(8,4))
        vv = tk.StringVar(); self._entry(dlg, vv, 30).pack(anchor="w", padx=12)

        tk.Label(dlg, text="Position X, Y:", font=FONT, bg=SURFACE, fg=TEXT).pack(anchor="w", padx=12, pady=(8,4))
        pf = tk.Frame(dlg, bg=SURFACE); pf.pack(anchor="w", padx=12)
        xv = tk.StringVar(); yv = tk.StringVar()
        self._entry(pf, xv, 8).pack(side="left", padx=(0,4))
        self._entry(pf, yv, 8).pack(side="left")

        tk.Label(dlg, text="Duration / Amount:", font=FONT, bg=SURFACE, fg=TEXT).pack(anchor="w", padx=12, pady=(8,4))
        dv = tk.StringVar(value="1.0"); self._entry(dlg, dv, 10).pack(anchor="w", padx=12)

        def add():
            t = tv.get(); a = {"type": t}
            if t in ("click","double_click","right_click","move"):
                try: a["position"] = {"x": int(xv.get()), "y": int(yv.get())}
                except ValueError: messagebox.showerror("Error","X/Y must be integers"); return
                if t in ("click","double_click"): a["button"] = "left"
            elif t == "key": a["value"] = vv.get()
            elif t == "hotkey": a["keys"] = [k.strip() for k in vv.get().split("+")]
            elif t == "type_text": a["value"] = vv.get()
            elif t == "wait":
                try: a["duration"] = float(dv.get())
                except ValueError: a["duration"] = 1.0
            elif t == "scroll":
                try: a["amount"] = int(dv.get())
                except ValueError: a["amount"] = -3
                try: a["position"] = {"x": int(xv.get()), "y": int(yv.get())}
                except ValueError: pass
            self.actions.append(a); self._refresh_tree(); dlg.destroy()

        self._btn(dlg, "✅ Add", add, color=GREEN, fg_="#000").pack(pady=12)

    # ----------------------------------------------------------------
    # Config
    # ----------------------------------------------------------------

    def _build_config(self):
        cfg = {
            "general": {
                "scan_interval": float(self.scan_var.get() or 0.5),
                "confidence_threshold": self.confidence_var.get(),
                "monitor_index": 1,
                "loop": self.loop_var.get(),
                "loop_delay": float(self.delay_var.get() or 5.0),
                "max_loops": int(self.maxloop_var.get() or 0),
                "failsafe": True,
                "grayscale": self.grayscale_var.get(),
                "detect_mode": self.detect_mode_var.get(),
            },
            "color_detection": {
                "target_hsv": list(self.picked_color_hsv) if self.picked_color_hsv else None,
                "target_rgb": list(self.picked_color_rgb) if self.picked_color_rgb else None,
                "tolerance": self.color_tol_var.get(),
                "min_area": int(self.min_area_var.get() or 300),
                "scan_region": self.scan_region,
            },
            "tasks": [{
                "name": "Recorded Task",
                "enabled": True,
                "reference_image": self.target_path_var.get(),
                "confidence": self.confidence_var.get(),
                "action_on_find": self.action_on_find_var.get(),
                "click_offset": {"x": 0, "y": 0},
                "post_actions": [a.copy() for a in self.actions],
            }],
        }
        return cfg

    def _save_config(self, path=None):
        if not path:
            path = filedialog.asksaveasfilename(title="Save Config", initialfile="config.json",
                                                defaultextension=".json", filetypes=[("JSON","*.json")])
        if not path: return
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._build_config(), f, indent=2, ensure_ascii=False)
        self._log(f"Config saved → {path}", "SUCCESS")

    def _load_config_dialog(self):
        p = filedialog.askopenfilename(title="Load Config", filetypes=[("JSON","*.json")])
        if p: self._load_config(p)

    def _load_config(self, path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            g = cfg.get("general", {})
            self.loop_var.set(g.get("loop", True))
            self.delay_var.set(str(g.get("loop_delay", 5.0)))
            self.scan_var.set(str(g.get("scan_interval", 0.5)))
            self.maxloop_var.set(str(g.get("max_loops", 0)))
            self.confidence_var.set(g.get("confidence_threshold", 0.65))
            self.grayscale_var.set(g.get("grayscale", True))
            self.detect_mode_var.set(g.get("detect_mode", "color"))

            cd = cfg.get("color_detection", {})
            if cd.get("target_hsv"):
                self.picked_color_hsv = tuple(cd["target_hsv"])
                self.picked_color_rgb = tuple(cd.get("target_rgb", [0, 0, 0]))
                r, g_, b = self.picked_color_rgb
                self.color_swatch.configure(bg=f"#{r:02x}{g_:02x}{b:02x}")
                h, s, v = self.picked_color_hsv
                self.color_info_label.configure(
                    text=f"RGB({r},{g_},{b})  HSV({h},{s},{v})", fg=TEXT)
            self.color_tol_var.set(cd.get("tolerance", 25))
            self.min_area_var.set(str(cd.get("min_area", 300)))
            self.scan_region = cd.get("scan_region")
            if self.scan_region:
                r = self.scan_region
                self.region_label.configure(
                    text=f"Region: ({r['x']},{r['y']}) {r['w']}×{r['h']}", fg=GREEN)

            tasks = cfg.get("tasks", [])
            if tasks:
                t = tasks[0]
                self.target_path_var.set(t.get("reference_image", ""))
                self.action_on_find_var.set(t.get("action_on_find", "right_click"))
                self.confidence_var.set(t.get("confidence", 0.65))
                self.actions = t.get("post_actions", [])
            else:
                self.actions = []

            self._refresh_tree()
            self._on_mode_change()
            self._log(f"Loaded {path}", "SUCCESS")
        except Exception as e:
            self._log(f"Load error: {e}", "ERROR")

    def _load_config_if_exists(self):
        if os.path.exists(CONFIG_FILE):
            self._load_config(CONFIG_FILE)

    # ----------------------------------------------------------------
    # Run automation
    # ----------------------------------------------------------------

    def _start_run(self):
        if self.is_running or self.is_recording: return

        mode = self.detect_mode_var.get()
        if mode == "color" and self.picked_color_hsv is None:
            self._log("No color picked! Click 🎨 Pick Color first.", "ERROR"); return
        if mode == "template":
            tp = self.target_path_var.get()
            if not tp or not os.path.exists(tp):
                self._log(f"Target image not found: {tp}", "ERROR"); return

        self.is_running = True
        self.stop_event.clear()
        self.run_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")

        cfg = self._build_config()
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)

        self._log("▶ Automation started", "SUCCESS")
        self.run_thread = threading.Thread(target=self._run_loop, daemon=True)
        self.run_thread.start()

    def _stop_run(self):
        if not self.is_running: return
        self.stop_event.set()
        self.is_running = False
        self.run_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self._log("⏹ Automation stopped", "WARNING")

    def _run_loop(self):
        cfg = self._build_config()
        g = cfg["general"]
        mode = g.get("detect_mode", "color")
        tasks = cfg.get("tasks", [])
        task = tasks[0] if tasks else {}

        on_find      = task.get("action_on_find", "right_click")
        offset       = task.get("click_offset", {})
        post_actions = task.get("post_actions", [])
        scan_iv      = g.get("scan_interval", 0.5)
        do_loop      = g.get("loop", True)
        loop_delay   = g.get("loop_delay", 5.0)
        max_loops    = g.get("max_loops", 0)
        mon_idx      = g.get("monitor_index", 1)

        # Color detection params
        cd = cfg.get("color_detection", {})
        target_hsv = tuple(cd["target_hsv"]) if cd.get("target_hsv") else None
        color_tol = cd.get("tolerance", 25)
        min_area = cd.get("min_area", 300)
        color_scan_region = cd.get("scan_region")

        # Template params
        template = None
        if mode == "template":
            template = cv2.imread(task.get("reference_image", ""), cv2.IMREAD_COLOR)
            if template is None:
                self.root.after(0, lambda: self._log("Cannot read target image", "ERROR"))
                self.root.after(0, self._stop_run); return

        confidence = task.get("confidence", 0.65)
        use_gray = g.get("grayscale", True)

        pyautogui.FAILSAFE = g.get("failsafe", True)
        pyautogui.PAUSE = 0.05
        iteration = 0

        try:
            with mss.mss() as sct:
                while not self.stop_event.is_set():
                    iteration += 1
                    self.root.after(0, lambda n=iteration: self._log(f"Scan #{n}"))

                    mon = sct.monitors[mon_idx]
                    shot = sct.grab(mon)
                    screen = cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)

                    found = False
                    click_x, click_y = 0, 0

                    if mode == "color" and target_hsv:
                        clusters, _ = find_color_clusters(
                            screen, target_hsv,
                            tolerance=color_tol, min_area=min_area,
                            scan_region=color_scan_region
                        )
                        if clusters:
                            cx, cy, area = clusters[0]
                            click_x = cx + mon["left"]
                            click_y = cy + mon["top"]
                            found = True
                            self.root.after(0, lambda x=click_x, y=click_y, a=area:
                                self._log(f"Color cluster at ({x},{y}) area={a}", "SUCCESS"))

                    elif mode == "template" and template is not None:
                        match = find_template_multiscale(
                            screen, template, threshold=confidence,
                            use_grayscale=use_gray)
                        if match:
                            cx, cy, best_val, best_scale = match
                            click_x = cx + mon["left"] + offset.get("x", 0)
                            click_y = cy + mon["top"]  + offset.get("y", 0)
                            found = True
                            self.root.after(0, lambda v=best_val, x=click_x, y=click_y:
                                self._log(f"Template at ({x},{y}) conf={v:.2f}", "SUCCESS"))

                    if found:
                        if on_find == "click":         pyautogui.click(click_x, click_y)
                        elif on_find == "right_click": pyautogui.rightClick(click_x, click_y)
                        elif on_find == "double_click": pyautogui.doubleClick(click_x, click_y)

                        for action in post_actions:
                            if self.stop_event.is_set(): break
                            self._exec(action)

                        if not do_loop: break
                        self._interruptible_sleep(loop_delay)
                    else:
                        self._interruptible_sleep(scan_iv)

                    if 0 < max_loops <= iteration:
                        self.root.after(0, lambda: self._log("Max loops reached", "WARNING"))
                        break

        except pyautogui.FailSafeException:
            self.root.after(0, lambda: self._log("Failsafe triggered!", "WARNING"))
        except Exception as e:
            self.root.after(0, lambda err=str(e): self._log(f"Error: {err}", "ERROR"))

        self.root.after(0, self._stop_run)

    def _interruptible_sleep(self, seconds):
        end = time.time() + seconds
        while time.time() < end and not self.stop_event.is_set():
            time.sleep(0.1)

    def _exec(self, action):
        t = action["type"]
        if t == "wait":
            self._interruptible_sleep(action.get("duration", 1.0))
        elif t == "click":
            p = action.get("position", {}); x, y = p.get("x"), p.get("y")
            b = action.get("button", "left")
            pyautogui.click(x, y, button=b) if x is not None else pyautogui.click(button=b)
        elif t == "double_click":
            p = action.get("position", {}); x, y = p.get("x"), p.get("y")
            b = action.get("button", "left")
            pyautogui.doubleClick(x, y, button=b) if x is not None else pyautogui.doubleClick(button=b)
        elif t == "right_click":
            p = action.get("position", {}); x, y = p.get("x"), p.get("y")
            pyautogui.rightClick(x, y) if x is not None else pyautogui.rightClick()
        elif t == "move":
            p = action["position"]; pyautogui.moveTo(p["x"], p["y"])
        elif t == "key":
            pyautogui.press(action["value"])
        elif t == "hotkey":
            pyautogui.hotkey(*action["keys"])
        elif t == "type_text":
            pyautogui.typewrite(action["value"], interval=action.get("interval", 0.05))
        elif t == "scroll":
            amt = action.get("amount", -3); p = action.get("position", {})
            x, y = p.get("x"), p.get("y")
            pyautogui.scroll(amt, x, y) if x is not None else pyautogui.scroll(amt)

    # ----------------------------------------------------------------
    # Main
    # ----------------------------------------------------------------

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()

    def _on_close(self):
        if self.is_running: self.stop_event.set()
        if self.is_recording: self.recorder.stop()
        if self.global_kb_listener: self.global_kb_listener.stop()
        self.root.destroy()


# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(IMAGES_DIR, exist_ok=True)
    AutomationGUI().run()
