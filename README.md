# 3D UAV Path Planning Using OpenStreetMap

This project implements a 3D voxel-based path planning system for drones
using OpenStreetMap data and the A* algorithm.

## Features
- OSM building extraction
- Building height estimation
- 3D voxel grid generation
- 3D A* path planning
- Fly-over vs fly-around logic
- Mission Planner waypoint export

## Technologies
- Python
- OSMnx
- GeoPandas
- NumPy
- PyProj
- Matplotlib
## System Pipeline

OpenStreetMap Data

        ↓
        
Building Footprint Extraction

        ↓
        
Building Height Estimation

        ↓
        
3D Obstacle Modeling

        ↓
        
Voxel Grid Generation

        ↓
        
3D A* Path Planning

        ↓
        
Waypoint Generation

        ↓
        
QGroundControl Visualization

## Run the Project
Install dependencies:

pip install -r requirements.txt

Run the planner:

python main_3dplanner.py

## Author
Moksh, Ananya, Abhi
