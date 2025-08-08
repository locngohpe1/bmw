# bmw/MCTA/run_mcta.py
"""
MCTA Launcher Script - Fixes path issues and provides easy startup
Run this file to start MCTA simulation
"""

import sys
import os

# Add project paths to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
bmw_dir = os.path.dirname(current_dir)  # bmw folder
mcta_dir = current_dir  # MCTA folder

# Add both directories to Python path
sys.path.insert(0, bmw_dir)
sys.path.insert(0, mcta_dir)


def check_dependencies():
    """Check if required files exist"""
    required_files = [
        os.path.join(bmw_dir, 'grid_map.py'),
        os.path.join(bmw_dir, 'dynamic_obstacles_manager.py'),
        os.path.join(bmw_dir, 'a_star.py'),
        os.path.join(mcta_dir, 'mcta_logic.py'),
        os.path.join(mcta_dir, 'mcta_coordinator.py'),
        os.path.join(mcta_dir, 'mcta_uav_robot.py'),
    ]

    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)

    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   {file_path}")
        return False

    return True


def find_available_maps():
    """Find available map files"""
    map_folder = os.path.join(bmw_dir, 'map')
    available_maps = []

    if os.path.exists(map_folder):
        for root, dirs, files in os.walk(map_folder):
            for file in files:
                if file.endswith('.txt'):
                    rel_path = os.path.relpath(os.path.join(root, file), bmw_dir)
                    available_maps.append(rel_path)

    return available_maps


def main():
    print("🚁 MCTA Multi-UAV Coverage Path Planning")
    print("=" * 50)

    # Check dependencies
    if not check_dependencies():
        print("❌ Cannot start - missing required files")
        return

    # Show available maps
    available_maps = find_available_maps()
    if available_maps:
        print("📍 Available maps:")
        for i, map_path in enumerate(available_maps[:5]):  # Show first 5
            print(f"   {i + 1}. {map_path}")
        if len(available_maps) > 5:
            print(f"   ... and {len(available_maps) - 5} more")
    else:
        print("⚠️  No map files found in bmw/map/")

    print()

    # Import and run MCTA
    try:
        # Import MCTA simulation
        from mcta_simulation import MCTASimulation
        import argparse

        # Parse arguments
        parser = argparse.ArgumentParser(description='MCTA Multi-UAV Coverage Simulation')
        parser.add_argument('--map', type=str, default='map/real_map/denmark.txt', help='Path to map file')
        parser.add_argument('--uavs', type=int, default=4, help='Number of UAVs')
        parser.add_argument('--energy', type=float, default=1000, help='Energy capacity per UAV')
        parser.add_argument('--speed', type=float, default=0.1, help='Dynamic obstacle speed factor')
        parser.add_argument('--edit', action='store_true', help='Start in map editing mode')

        args = parser.parse_args()

        # Validate map file
        map_path = args.map if os.path.isabs(args.map) else os.path.join(bmw_dir, args.map)
        if not os.path.exists(map_path):
            print(f"❌ Map file not found: {map_path}")
            if available_maps:
                print("💡 Try one of these maps:")
                for map_file in available_maps[:3]:
                    print(f"   python run_mcta.py --map {map_file}")
            return

        print(f"🗺️  Loading map: {args.map}")
        print(f"🚁 UAVs: {args.uavs}")
        print(f"⚡ Energy: {args.energy}")
        print(f"💨 Obstacle speed: {args.speed}")
        print("=" * 50)

        # Create and run simulation
        simulation = MCTASimulation(
            map_file=args.map,
            num_uavs=args.uavs,
            energy_capacity=args.energy
        )

        if args.edit:
            print("🎨 Starting map editor...")
            simulation.edit_map_mode()
        else:
            print("🚀 Starting MCTA simulation...")
            simulation.initialize_uavs()
            simulation.initialize_dynamic_obstacles(args.speed)
            print("✅ Initialization complete")
            print()
            print("Controls:")
            print("  SPACE - Pause/Resume")
            print("  LEFT/RIGHT - Adjust speed")
            print("  R - Reset simulation")
            print("  ESC/Close - Exit")
            print("=" * 50)

        # Run simulation
        simulation.run_simulation()

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all required files are in the correct locations")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()