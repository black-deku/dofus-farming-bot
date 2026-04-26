"""
Dofus Resource Farming Bot
==========================
Scans predefined node locations via pixel-color matching (HSV),
queues harvests instantly, and cycles through configured maps.

Features:
  - Single-screenshot fast scan per map
  - Mouse parked before capture to avoid hover-highlight false negatives
  - Configurable cooldown pause every N minutes to look human
"""

import json
import logging
import sys
import time
from pathlib import Path

import cv2
import mss
import numpy as np
import pyautogui

# ── Config ──────────────────────────────────────────────────────────────────
CONFIG_PATH = Path("fer_config.json")
COOLDOWN_INTERVAL = 4 * 60  # seconds between cooldowns
COOLDOWN_DURATION = 10       # seconds to pause during cooldown
MOUSE_PARK = (10, 10)        # neutral corner to park the cursor before scanning

# ── Setup ───────────────────────────────────────────────────────────────────
pyautogui.PAUSE = 0.01

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("farm_fer")


# ── Helpers ─────────────────────────────────────────────────────────────────

def grab_hsv(sct: mss.mss) -> np.ndarray:
    """Full-screen capture returned as an HSV numpy array."""
    shot = sct.grab(sct.monitors[1])
    bgr = cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)


def hsv_match(pixel, target, tolerance: int) -> bool:
    """Return True if *pixel* HSV is within *tolerance* of *target* HSV."""
    h, s, v = [int(c) for c in pixel]
    th, ts, tv = [int(c) for c in target]
    h_diff = min(abs(h - th), 180 - abs(h - th))
    return (
        h_diff <= min(tolerance, 15)
        and abs(s - ts) <= min(tolerance, 50)
        and abs(v - tv) <= min(tolerance, 50)
    )


# ── Map Processing ─────────────────────────────────────────────────────────

def process_map(map_data: dict, cfg: dict, sct: mss.mss) -> int:
    """Scan & harvest all nodes on one map, then transition.

    Returns the number of nodes that were queued for collection.
    """
    gen = cfg.get("general", {})
    tolerance = gen.get("iron_hsv_tolerance", 30)
    default_hsv = gen.get("iron_hsv_target", [0, 0, 100])
    mining_dur = gen.get("mining_duration", 5.0)
    move_dur = gen.get("mouse_move_duration", 0.2)
    off_x = gen.get("collect_offset_x", 40)
    off_y = gen.get("collect_offset_y", 40)

    map_name = map_data.get("map_name", "?")
    nodes = map_data.get("nodes", [])
    log.info("🗺️  Map: %s (%d nodes)", map_name, len(nodes))

    # Park cursor away from nodes so the screenshot has no hover highlight
    pyautogui.moveTo(*MOUSE_PARK, duration=0.0)
    time.sleep(0.05)

    hsv_frame = grab_hsv(sct)
    queued = 0

    for i, node in enumerate(nodes, 1):
        x, y = node["x"], node["y"]
        target = node.get("hsv", default_hsv)
        pixel = hsv_frame[y][x]

        if not hsv_match(pixel, target, tolerance):
            continue

        log.info("  ⛏️  [%d/%d] Iron at (%d, %d) → collecting", i, len(nodes), x, y)
        pyautogui.moveTo(x, y, duration=0.0)
        pyautogui.click(x, y)
        time.sleep(0.15)
        pyautogui.click(x + off_x, y + off_y)
        queued += 1

    # Wait for all queued mining to finish before leaving the map
    transition = map_data.get("transition_zone")
    if transition:
        if queued > 0:
            wait = queued * mining_dur
            log.info("  ⏳ Waiting %.1fs for %d node(s) to finish mining…", wait, queued)
            time.sleep(wait)

        tx, ty = transition["x"], transition["y"]
        log.info("  🏃 Transition → (%d, %d)", tx, ty)
        pyautogui.moveTo(tx, ty, duration=move_dur)
        pyautogui.click(tx, ty)
        time.sleep(gen.get("map_transition_duration", 4.0))
    else:
        if queued > 0:
            wait = queued * mining_dur
            log.info("  ⏳ Waiting %.1fs for %d node(s) to finish mining…", wait, queued)
            time.sleep(wait)
        else:
            time.sleep(0.01)

    return queued


# ── Main Loop ──────────────────────────────────────────────────────────────

def main() -> None:
    if not CONFIG_PATH.exists():
        log.error("Config not found: %s", CONFIG_PATH)
        sys.exit(1)

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    pyautogui.FAILSAFE = cfg.get("general", {}).get("failsafe", True)

    maps = cfg.get("maps", [])
    if not maps:
        log.error("No maps in config")
        sys.exit(1)

    log.info("=" * 50)
    log.info("Resource Farming Bot Started")
    log.info("Maps: %d | Cooldown: %ds every %dm", len(maps), COOLDOWN_DURATION, COOLDOWN_INTERVAL // 60)
    log.info("=" * 50)
    log.info("Ctrl+C to stop | Mouse → top-left corner for failsafe\n")

    try:
        with mss.mss() as sct:
            cycle = 0
            last_cooldown = time.time()

            while True:
                cycle += 1
                log.info("── Cycle #%d ──", cycle)

                for m in maps:
                    process_map(m, cfg, sct)

                # Periodic cooldown to look human
                elapsed = time.time() - last_cooldown
                if elapsed >= COOLDOWN_INTERVAL:
                    log.info("💤 Cooldown: pausing %ds…", COOLDOWN_DURATION)
                    time.sleep(COOLDOWN_DURATION)
                    last_cooldown = time.time()
                    log.info("💤 Cooldown over, resuming.")

    except KeyboardInterrupt:
        log.info("\n🛑 Stopped by user")
    except pyautogui.FailSafeException:
        log.info("\n🛑 Failsafe triggered")

    log.info("Done.")


if __name__ == "__main__":
    main()
