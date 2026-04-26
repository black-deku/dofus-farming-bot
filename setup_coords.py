import time
import cv2
import mss
import numpy as np
import pyautogui
from pynput import keyboard

print("=" * 50)
print("Dofus Coordinate & Color Setup Tool")
print("=" * 50)
print("Instructions:")
print("1. Hover over an IRON VEIN and press F8 to get node coordinates.")
print("2. Hover over the MAP TRANSITION edge/sun and press F9.")
print("3. Press 'Esc' to exit.")
print("=" * 50)

def get_pixel_hsv(sct: mss.mss, x: int, y: int) -> list:
    region = {"left": x, "top": y, "width": 1, "height": 1}
    shot = sct.grab(region)
    frame = np.array(shot)
    bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return [int(hsv[0][0][0]), int(hsv[0][0][1]), int(hsv[0][0][2])]

def on_press(key):
    if key == keyboard.Key.esc:
        print("Exiting...")
        return False
    elif key == keyboard.Key.f8:
        x, y = pyautogui.position()
        with mss.mss() as sct:
            hsv = get_pixel_hsv(sct, x, y)
        print(f"\n[Captured Node]")
        print(f'{{"x": {x}, "y": {y}}},')
        print(f'# (Optional) specific color: {hsv}')
        print("-" * 30)
    elif key == keyboard.Key.f9:
        x, y = pyautogui.position()
        print(f"\n[Captured Map Transition]")
        print(f'"transition_zone": {{')
        print(f'  "x": {x},')
        print(f'  "y": {y}')
        print(f'}}')
        print("-" * 30)

with keyboard.Listener(on_press=on_press) as listener:
    listener.join()
