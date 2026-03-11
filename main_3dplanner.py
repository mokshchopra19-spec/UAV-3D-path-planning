import osmnx as ox
import geopandas as gpd
import math
import json
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from pyproj import Transformer
# ===============================
# STEP 1: Load OSM file
# ===============================

osm_file = "map.osm"   # file must be in same folder

print("Loading OSM file...")

buildings = ox.features_from_xml(
    osm_file,
    tags={"building": True}
)

# Keep only building polygons
buildings = buildings[
    buildings.geometry.type.isin(["Polygon", "MultiPolygon"])
]

print("STEP 1 DONE")
print("Number of buildings loaded:", len(buildings))
def nearest_free_cell(cell, obstacle_set):
    x, y, z = cell

    if cell not in obstacle_set:
        return cell

    # search nearby cells
    for r in range(1, 10):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    candidate = (x + dx, y + dy, z + dz)
                    if candidate not in obstacle_set:
                        return candidate

    return cell
# ===============================
# STEP 2: Extract building heights
# ===============================

def extract_building_height(row):
    """
    Returns building height in meters.
    Priority:
    1) explicit height tag
    2) building:levels
    3) inferred from building type
    4) safe fallback
    """

    # --- Case 1: Explicit height tag ---
    h = row.get("height")
    if h:
        try:
            return float(str(h).replace("m", "").strip())
        except:
            pass

    # --- Case 2: Building levels ---
    levels = row.get("building:levels")
    if levels:
        try:
            return float(levels) * 3.0  # 1 level ≈ 3 meters
        except:
            pass

    # --- Case 3: Infer from building type ---
    btype = row.get("building")

    if btype in ["apartments", "residential", "dormitory"]:
        return 30.0
    elif btype in ["commercial", "office"]:
        return 45.0
    elif btype in ["house"]:
        return 9.0
    elif btype in ["school", "university", "hospital", "college"]:
        return 18.0
    elif btype == "roof":
        return 6.0
    elif btype == "yes":
        return 15.0

    # --- Case 4: Absolute fallback ---
    return 12.0


# 🔹 Apply height extraction to ALL buildings
buildings["height_m"] = buildings.apply(extract_building_height, axis=1)

# 🔹 Safety: ensure no NaN survives
buildings["height_m"] = buildings["height_m"].fillna(12.0)

# Reproject buildings to a metric CRS (UTM zone 43N) for meter-based grids
buildings = buildings.to_crs(epsg=32643)

print("\nSTEP 2 DONE")
print("Building height statistics (meters):")
print(buildings["height_m"].describe())


# 🔹 Print height of EVERY building (what you asked for)
print("\nFINAL BUILDING HEIGHTS:")
for i, (_, row) in enumerate(buildings.iterrows(), start=1):
    print(
        f"Building {i:02d} | "
        f"Height = {row['height_m']:.1f} m | "
        f"Type = {row.get('building')}"
    )
# ===============================
# STEP 3: Convert buildings to 3D obstacles
# ===============================

print("\nSTEP 3: Creating 3D obstacles...")

# Create simple axis-aligned 3D obstacle boxes using projected building bounds
obstacles_3d = []
for _, row in buildings.iterrows():
    geom = row.geometry
    height = row["height_m"]

    if geom is None or geom.is_empty:
        continue

    minx, miny, maxx, maxy = geom.bounds
    obstacle = {
        "x_min": minx,
        "x_max": maxx,
        "y_min": miny,
        "y_max": maxy,
        "z_min": 0.0,
        "z_max": height,
    }
    obstacles_3d.append(obstacle)

print(f"Total 3D obstacles created: {len(obstacles_3d)}")

print("\nSample 3D Obstacles:")
for i, obs in enumerate(obstacles_3d[:5], start=1):
    print(
        f"Obs {i}: "
        f"X[{obs['x_min']:.1f}, {obs['x_max']:.1f}] | "
        f"Y[{obs['y_min']:.1f}, {obs['y_max']:.1f}] | "
        f"Z[0, {obs['z_max']:.1f}]"
    )
# ===============================
# STEP 4: Fly-Over vs Fly-Around decision
# ===============================

print("\nSTEP 4: Classifying obstacles (fly-over vs fly-around)...")

# Drone parameters
CRUISE_ALTITUDE = 60.0   # meters
SAFETY_MARGIN = 5.0      # meters above building

fly_over = []
fly_around = []

for obs in obstacles_3d:
    required_clearance = obs["z_max"] + SAFETY_MARGIN

    if CRUISE_ALTITUDE >= required_clearance:
        fly_over.append(obs)
    else:
        fly_around.append(obs)

print(f"Buildings fly-over allowed: {len(fly_over)}")
print(f"Buildings requiring avoidance: {len(fly_around)}")

# Show examples
print("\nSample Fly-OVER buildings:")
for i, obs in enumerate(fly_over[:3], start=1):
    print(f"{i}. Height = {obs['z_max']} m → OK to overfly")

print("\nSample Fly-AROUND buildings:")
for i, obs in enumerate(fly_around[:3], start=1):
    print(f"{i}. Height = {obs['z_max']} m → Must avoid")
# ==================================================
# STEP 5.5: Create 3D Obstacle Grid (FIXED)
# ==================================================

print("\nSTEP 5.5: Creating 3D obstacle grid...")

import numpy as np
from shapely.geometry import Point

# -----------------------------
# Grid resolution (IMPORTANT)
# -----------------------------
GRID_RES_XY = 5.0    # meters (horizontal)
GRID_RES_Z  = 5.0    # meters (vertical)

MAX_ALTITUDE = 80.0  # meters (drone ceiling)

# -----------------------------
# Compute grid bounds from buildings
# -----------------------------
xmin, ymin, xmax, ymax = buildings.total_bounds

zmin = 0.0
zmax = MAX_ALTITUDE

nx = int((xmax - xmin) / GRID_RES_XY) + 1
ny = int((ymax - ymin) / GRID_RES_XY) + 1
nz = int((zmax - zmin) / GRID_RES_Z) + 1

print(f"3D grid size: {nx} x {ny} x {nz}")

# -----------------------------
# Create EMPTY 3D grid
# -----------------------------
grid_3d = np.zeros((nx, ny, nz), dtype=np.uint8)

# -----------------------------
# Fill obstacles from buildings
# -----------------------------
for _, row in buildings.iterrows():
    poly = row.geometry
    h = row["height_m"]

    if np.isnan(h):
        continue  # skip unknown height

    z_cells = int(h / GRID_RES_Z)

    for ix in range(nx):
        for iy in range(ny):
            x = xmin + ix * GRID_RES_XY
            y = ymin + iy * GRID_RES_XY
            pt = Point(x, y)

            if poly.contains(pt):
                grid_3d[ix, iy, :z_cells] = 1

print("3D obstacle grid created successfully.")
# ===============================
# STEP 6: Grid parameters & transforms
# ===============================

# --- Grid resolution ---
GRID_RES = 10.0     # meters (XY resolution)
ALT_RES  = 5.0      # meters (Z resolution)

# --- Altitude limits ---
MIN_ALT = 0.0
MAX_ALT = 80.0      # drone won't fly above this

# --- Bounding box from buildings ---
xmin, ymin, xmax, ymax = buildings.total_bounds

# --- Ensure grid is not collapsing ---
nx = max(3, int((xmax - xmin) / GRID_RES) + 1)
ny = max(3, int((ymax - ymin) / GRID_RES) + 1)
nz = int((MAX_ALT - MIN_ALT) / ALT_RES) + 1

print("\nSTEP 6 DONE")
print(f"3D grid size: {nx} x {ny} x {nz}")


# --- World → Grid ---
def world_to_grid(x, y, z):
    gx = int((x - xmin) / GRID_RES)
    gy = int((y - ymin) / GRID_RES)
    gz = int((z - MIN_ALT) / ALT_RES)

    gx = max(0, min(nx - 1, gx))
    gy = max(0, min(ny - 1, gy))
    gz = max(0, min(nz - 1, gz))

    return gx, gy, gz


# --- Grid → World ---
def grid_to_world(gx, gy, gz):
    x = xmin + gx * GRID_RES
    y = ymin + gy * GRID_RES
    z = MIN_ALT + gz * ALT_RES
    return x, y, z

# -----------------------------
# Build obstacle occupancy set
# -----------------------------
obstacle_set = set()
for _, row in buildings.iterrows():
    poly = row.geometry
    h = row["height_m"]

    if poly is None or poly.is_empty:
        continue

    # number of vertical cells occupied by this building
    z_cells = int(math.ceil(float(h) / ALT_RES))

    minx, miny, maxx, maxy = poly.bounds
    gx_min, gy_min, _ = world_to_grid(minx, miny, MIN_ALT)
    gx_max, gy_max, _ = world_to_grid(maxx, maxy, MIN_ALT)

    for gx in range(gx_min, gx_max + 1):
        for gy in range(gy_min, gy_max + 1):
            # test point at cell center
            cx = xmin + gx * GRID_RES + GRID_RES * 0.5
            cy = ymin + gy * GRID_RES + GRID_RES * 0.5
            if poly.contains(Point(cx, cy)):
                for gz in range(0, min(nz, z_cells)):
                    obstacle_set.add((gx, gy, gz))

print(f"Total occupied cells in obstacle set: {len(obstacle_set)}")
# ===============================
# STEP 7: Define start & goal (3D)
# ===============================

# Start near south-west corner, safe altitude
start_xyz = (
    xmin + 2 * GRID_RES,
    ymin + 2 * GRID_RES,
    30.0          # meters
)

# Goal near north-east corner, slightly higher
goal_xyz = (
    xmax - 2 * GRID_RES,
    ymax - 2 * GRID_RES,
    40.0
)

start = world_to_grid(*start_xyz)
goal  = world_to_grid(*goal_xyz)

print("\nSTEP 7 DONE")
print("Start (world):", start_xyz)
print("Goal  (world):", goal_xyz)

original_start = start
original_goal = goal

start = nearest_free_cell(start, obstacle_set)
goal = nearest_free_cell(goal, obstacle_set)

if start != original_start:
    print(f"Adjusted start to nearest free cell: {start}")

if goal != original_goal:
    print(f"Adjusted goal to nearest free cell: {goal}")

print("Start (grid): ", start)
print("Goal  (grid): ", goal)
# ===============================
# STEP 8: 3D A* Path Planning
# ===============================

import heapq
import math

print("\nSTEP 8: Running 3D A* path planner...")

# --- 3D heuristic (Euclidean) ---
def heuristic_3d(a, b):
    return math.sqrt(
        (a[0] - b[0])**2 +
        (a[1] - b[1])**2 +
        (a[2] - b[2])**2
    )

# --- 26-connected 3D neighbors ---
NEIGHBORS_3D = [
    (dx, dy, dz)
    for dx in [-1, 0, 1]
    for dy in [-1, 0, 1]
    for dz in [-1, 0, 1]
    if not (dx == dy == dz == 0)
]

def nearest_free_cell(cell, obstacle_set):
    if cell not in obstacle_set:
        return cell

    for radius in range(1, max(nx, ny, nz)):
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    candidate = (cell[0] + dx, cell[1] + dy, cell[2] + dz)

                    if not (0 <= candidate[0] < nx and 0 <= candidate[1] < ny and 0 <= candidate[2] < nz):
                        continue

                    if candidate not in obstacle_set:
                        return candidate

    raise RuntimeError(f"No free cell found near {cell}")

def astar_3d(start, goal, obstacle_set):
    open_set = []
    heapq.heappush(open_set, (0.0, start))

    came_from = {}
    g_score = {start: 0.0}

    closed = set()

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        if current in closed:
            continue
        closed.add(current)

        for dx, dy, dz in NEIGHBORS_3D:
            nx_ = current[0] + dx
            ny_ = current[1] + dy
            nz_ = current[2] + dz

            # bounds check
            if not (0 <= nx_ < nx and 0 <= ny_ < ny and 0 <= nz_ < nz):
                continue

            neighbor = (nx_, ny_, nz_)

            # obstacle check
            if neighbor in obstacle_set:
                continue

            move_cost = math.sqrt(dx*dx + dy*dy + dz*dz)
            altitude_penalty = 0.5 * abs(dz)
            tentative_g = g_score[current] + move_cost + altitude_penalty

            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + heuristic_3d(neighbor, goal)
                heapq.heappush(open_set, (f, neighbor))

    return None


# --- Run planner ---
path_3d = astar_3d(start, goal, obstacle_set)

if path_3d is None:
    raise RuntimeError("No 3D path found.")
else:
    print(f"Path found with {len(path_3d)} nodes.")
# ===============================
# STEP 9: Convert grid path to world coordinates
# ===============================

print("\nSTEP 9: Converting path to world coordinates...")

path_world = [grid_to_world(gx, gy, gz) for gx, gy, gz in path_3d]

print("Sample world path points:")
for p in path_world[:5]:
    print(f"x={p[0]:.2f}, y={p[1]:.2f}, z={p[2]:.2f}")
# ===============================
# STEP 10: Export Mission Planner Waypoints
# ===============================

print("\nSTEP 10: Exporting mission file...")

OUTPUT_FILE = "mission1.waypoints"

# Transformer: UTM → WGS84 (lat/lon)
transformer = Transformer.from_crs(32643, 4326, always_xy=True)

with open(OUTPUT_FILE, "w") as f:

    # Mission planner header
    f.write("QGC WPL 110\n")

    seq = 0

    for i, (x, y, z) in enumerate(path_world):

        lon, lat = transformer.transform(x, y)

        current = 1 if i == 0 else 0
        frame = 3              # GLOBAL_RELATIVE_ALT
        command = 16           # NAV_WAYPOINT
        autocontinue = 1

        altitude = z if z > 0 else 20

        line = f"{seq}\t{current}\t{frame}\t{command}\t0\t0\t0\t0\t{lat}\t{lon}\t{altitude}\t{autocontinue}\n"

        f.write(line)

        seq += 1


print("✅ Mission Planner file created:", OUTPUT_FILE)
print("Total waypoints:", len(path_world))