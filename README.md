# Dofus Resource Farming Bot

Automated resource farming bot for Dofus. Uses pixel-color detection (HSV) to locate harvestable nodes, queues collections instantly, and cycles through maps.

## Features

- **Fast scanning** – single screenshot per map, instant pixel checks
- **Queue-based harvesting** – clicks all available nodes rapidly, then waits once
- **Multi-map cycling** – auto-transitions between configured maps
- **Anti-detection cooldown** – 10 s pause every 4 minutes
- **Dark-themed GUI** – configure maps, nodes, and transitions visually
- **Failsafe** – move mouse to top-left corner to abort instantly

---

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### GUI (recommended)

```bash
python fer_gui.py
```

1. **Add a map** with ➕
2. **Select the map**, then capture node positions with **F8** (hover over a resource and press F8)
3. **Capture a transition zone** with **F9** (hover over the map-change click point)
4. Set **Mining Duration** to match your gathering time
5. Click **▶ Run Bot**

### CLI (headless)

```bash
python farm_fer.py
```

Reads `fer_config.json` directly.

### Safety

| Method | Action |
|--------|--------|
| **Failsafe** | Move mouse to top-left corner → instant abort |
| **Ctrl+C** | Stop from terminal |

---

## Config Reference (`fer_config.json`)

### `general`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `mining_duration` | float | `5.0` | Seconds to wait per mined node |
| `map_transition_duration` | float | `4.0` | Seconds to wait for map load |
| `mouse_move_duration` | float | `0.2` | Cursor travel time to transition zone |
| `iron_hsv_target` | [H,S,V] | `[20,27,101]` | Default HSV color to detect |
| `iron_hsv_tolerance` | int | `50` | Color matching tolerance |
| `collect_offset_x/y` | int | `40` | Offset from node to "Collecter" menu item |
| `failsafe` | bool | `true` | Enable corner failsafe |

### `maps[]`

| Key | Type | Description |
|-----|------|-------------|
| `map_name` | string | Label for logs |
| `nodes` | array | `[{x, y, hsv}]` – positions captured via F8 |
| `transition_zone` | object | `{x, y}` – click target to change map |

---

## File Structure

```
├── farm_fer.py        # Bot script (main loop)
├── fer_gui.py         # GUI manager
├── fer_config.json    # Your saved configuration
├── requirements.txt   # Python dependencies
└── README.md
```

---

## Tips

1. **Capture colors without highlight** – the bot parks the cursor before scanning, but when capturing via F8, make sure you're hovering directly on the resource pixel
2. **Adjust tolerance** if nodes are missed (`iron_hsv_tolerance`)
3. **Mining duration** should match your character's actual gather time
