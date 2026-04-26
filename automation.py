"""
Screen Automation Script
========================
Reads instructions from a JSON config file.
Uses 'mss' for fast screen capture and OpenCV for template matching.
Uses 'pyautogui' for mouse/keyboard actions.

Usage:
    python automation.py                   # uses config.json
    python automation.py -c my_config.json # uses custom config
    python automation.py --dry-run         # preview without executing
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import mss
import numpy as np
import pyautogui

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("automation")

# ---------------------------------------------------------------------------
# Screen capture helpers
# ---------------------------------------------------------------------------

def grab_screen(sct: mss.mss, monitor_index: int = 1, region: dict | None = None) -> np.ndarray:
    """Capture the screen (or a sub-region) and return a BGR numpy array."""
    if region:
        shot = sct.grab(region)
    else:
        shot = sct.grab(sct.monitors[monitor_index])
    # mss returns BGRA; convert to BGR for OpenCV
    frame = np.array(shot)
    return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)


def find_template(
    screen: np.ndarray,
    template: np.ndarray,
    threshold: float = 0.8,
) -> tuple[int, int] | None:
    """
    Run template matching. Returns the (x, y) center of the best match
    on the *screen* coordinate system, or None if below threshold.
    """
    result = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val >= threshold:
        h, w = template.shape[:2]
        cx = max_loc[0] + w // 2
        cy = max_loc[1] + h // 2
        return (cx, cy)
    return None


# ---------------------------------------------------------------------------
# Action executor
# ---------------------------------------------------------------------------

def execute_action(action: dict, dry_run: bool = False) -> None:
    """Execute a single action dictionary."""
    a_type = action["type"]

    if a_type == "wait":
        duration = action.get("duration", 1.0)
        log.info("  ⏳ Wait %.2fs", duration)
        if not dry_run:
            time.sleep(duration)

    elif a_type == "click":
        pos = action.get("position", {})
        x, y = pos.get("x"), pos.get("y")
        btn = action.get("button", "left")
        log.info("  🖱️  Click %s at (%s, %s)", btn, x, y)
        if not dry_run:
            if x is not None and y is not None:
                pyautogui.click(x, y, button=btn)
            else:
                pyautogui.click(button=btn)

    elif a_type == "double_click":
        pos = action.get("position", {})
        x, y = pos.get("x"), pos.get("y")
        btn = action.get("button", "left")
        log.info("  🖱️  Double-click %s at (%s, %s)", btn, x, y)
        if not dry_run:
            if x is not None and y is not None:
                pyautogui.doubleClick(x, y, button=btn)
            else:
                pyautogui.doubleClick(button=btn)

    elif a_type == "right_click":
        pos = action.get("position", {})
        x, y = pos.get("x"), pos.get("y")
        log.info("  🖱️  Right-click at (%s, %s)", x, y)
        if not dry_run:
            if x is not None and y is not None:
                pyautogui.rightClick(x, y)
            else:
                pyautogui.rightClick()

    elif a_type == "move":
        pos = action["position"]
        log.info("  🖱️  Move to (%s, %s)", pos["x"], pos["y"])
        if not dry_run:
            pyautogui.moveTo(pos["x"], pos["y"])

    elif a_type == "key":
        key = action["value"]
        log.info("  ⌨️  Key press: %s", key)
        if not dry_run:
            pyautogui.press(key)

    elif a_type == "hotkey":
        keys = action["keys"]
        log.info("  ⌨️  Hotkey: %s", " + ".join(keys))
        if not dry_run:
            pyautogui.hotkey(*keys)

    elif a_type == "type_text":
        text = action["value"]
        interval = action.get("interval", 0.05)
        log.info("  ⌨️  Type: %s", text[:40])
        if not dry_run:
            pyautogui.typewrite(text, interval=interval)

    elif a_type == "scroll":
        amount = action.get("amount", -3)
        pos = action.get("position", {})
        x, y = pos.get("x"), pos.get("y")
        log.info("  🖱️  Scroll %d at (%s, %s)", amount, x, y)
        if not dry_run:
            if x is not None and y is not None:
                pyautogui.scroll(amount, x, y)
            else:
                pyautogui.scroll(amount)

    else:
        log.warning("  ❓ Unknown action type: %s", a_type)


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------

def run_task(
    task: dict,
    sct: mss.mss,
    general_cfg: dict,
    dry_run: bool = False,
) -> bool:
    """
    Run a single task: scan for the reference image, click it, then
    execute post_actions.  Returns True if the image was found & acted on.
    """
    name = task.get("name", "Unnamed")
    ref_path = Path(task["reference_image"])
    confidence = task.get("confidence", general_cfg.get("confidence_threshold", 0.8))

    if not ref_path.exists():
        log.error("❌ Reference image not found: %s", ref_path)
        return False

    template = cv2.imread(str(ref_path), cv2.IMREAD_COLOR)
    if template is None:
        log.error("❌ Failed to read image: %s", ref_path)
        return False

    monitor_idx = general_cfg.get("monitor_index", 1)
    region = general_cfg.get("scan_region")
    screen = grab_screen(sct, monitor_idx, region)

    match = find_template(screen, template, threshold=confidence)
    if match is None:
        log.debug("   [%s] not found on screen", name)
        return False

    cx, cy = match
    # Apply monitor offset if not using a custom region
    if not region:
        mon = sct.monitors[monitor_idx]
        cx += mon["left"]
        cy += mon["top"]

    # Apply user offset
    offset = task.get("click_offset", {})
    cx += offset.get("x", 0)
    cy += offset.get("y", 0)

    log.info("✅ [%s] Found at (%d, %d)", name, cx, cy)

    # Primary click on the found location
    action_on_find = task.get("action_on_find", "click")
    if action_on_find == "click":
        log.info("  🖱️  Clicking found location")
        if not dry_run:
            pyautogui.click(cx, cy)
    elif action_on_find == "double_click":
        if not dry_run:
            pyautogui.doubleClick(cx, cy)
    elif action_on_find == "right_click":
        if not dry_run:
            pyautogui.rightClick(cx, cy)
    elif action_on_find == "none":
        log.info("  (no click on find)")

    # Execute post-actions
    post_actions = task.get("post_actions", [])
    for i, action in enumerate(post_actions, 1):
        log.info("  Action %d/%d", i, len(post_actions))
        execute_action(action, dry_run=dry_run)

    return True


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Screen automation via JSON config")
    parser.add_argument("-c", "--config", default="config.json", help="Path to config JSON")
    parser.add_argument("--dry-run", action="store_true", help="Preview actions without executing")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    args = parser.parse_args()

    if args.verbose:
        log.setLevel(logging.DEBUG)

    config_path = Path(args.config)
    if not config_path.exists():
        log.error("Config file not found: %s", config_path)
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    general = config.get("general", {})
    tasks = config.get("tasks", [])

    # Filter to enabled tasks only
    tasks = [t for t in tasks if t.get("enabled", True)]
    if not tasks:
        log.warning("No enabled tasks found in config.")
        sys.exit(0)

    # PyAutoGUI settings
    if general.get("failsafe", True):
        pyautogui.FAILSAFE = True   # move mouse to top-left corner to abort
    else:
        pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0.1

    loop = general.get("loop", False)
    loop_delay = general.get("loop_delay", 5.0)
    max_loops = general.get("max_loops", 0)  # 0 = infinite
    scan_interval = general.get("scan_interval", 0.5)

    log.info("=" * 50)
    log.info("Screen Automation Started")
    log.info("Config  : %s", config_path)
    log.info("Tasks   : %d enabled", len(tasks))
    log.info("Loop    : %s (delay %.1fs)", loop, loop_delay)
    log.info("Dry run : %s", args.dry_run)
    log.info("=" * 50)
    log.info("Press Ctrl+C to stop  |  Move mouse to top-left corner for failsafe")
    log.info("")

    iteration = 0
    try:
        with mss.mss() as sct:
            while True:
                iteration += 1
                log.info("--- Scan #%d ---", iteration)

                for task in tasks:
                    found = run_task(task, sct, general, dry_run=args.dry_run)
                    if found:
                        time.sleep(scan_interval)
                        if task.get("break_on_find", False):
                            break

                if not loop:
                    break
                if max_loops > 0 and iteration >= max_loops:
                    log.info("Reached max loops (%d). Stopping.", max_loops)
                    break

                log.info("Waiting %.1fs before next scan...\n", loop_delay)
                time.sleep(loop_delay)

    except KeyboardInterrupt:
        log.info("\n🛑 Stopped by user (Ctrl+C)")
    except pyautogui.FailSafeException:
        log.info("\n🛑 Failsafe triggered (mouse moved to corner)")

    log.info("Done.")


if __name__ == "__main__":
    main()
