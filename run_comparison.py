#!/usr/bin/env python3
"""
Main script to run algorithm comparison
Usage: python run_comparison.py [options]
"""

import os
import sys
import math
import argparse
import traceback
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import helper functions
try:
    from helper_functions import setup_comparison_environment, create_test_map
    from compare_algorithms import AlgorithmComparison
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all required files are in the same directory")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Compare Coverage Path Planning Algorithms')

    # Map configuration
    parser.add_argument('--maps', nargs='+',
                        default=None,
                        help='Map files to test (if not provided, will create test maps)')
    parser.add_argument('--create-test-maps', action='store_true',
                        help='Create test maps automatically')
    parser.add_argument('--map-sizes', nargs='+', type=int,
                        default=[20],
                        help='Sizes for auto-generated test maps')
    parser.add_argument('--obstacle-density', type=float, default=0.2,
                        help='Obstacle density for test maps (0.0-1.0)')

    # Robot configuration
    parser.add_argument('--energy', type=float, default=1000,
                        help='Robot energy capacity')
    parser.add_argument('--speed', type=float, default=0.1,
                        help='Dynamic obstacles speed factor')

    # Output configuration
    parser.add_argument('--output', type=str, default='comparison_results',
                        help='Output directory for results')
    parser.add_argument('--no-charts', action='store_true',
                        help='Skip generating charts')

    # Algorithm selection
    parser.add_argument('--algorithms', nargs='+',
                        choices=['A', 'D', 'C', 'all'],
                        default=['all'],
                        help='Algorithms to test (A=BWave, D=Neural, C=Auction)')

    # Execution configuration
    parser.add_argument('--max-steps', type=int, default=5000,
                        help='Maximum steps per algorithm')
    parser.add_argument('--headless', action='store_true', default=True,
                        help='Run in headless mode (no GUI)')

    args = parser.parse_args()

    print("🚀 COVERAGE PATH PLANNING ALGORITHM COMPARISON")
    print("=" * 60)
    print(f"📅 Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Setup environment
    print("\n🔧 Setting up comparison environment...")
    if not setup_comparison_environment():
        print("❌ Failed to setup environment")
        return 1

    # Determine which algorithms to run
    if 'all' in args.algorithms:
        algorithms_to_run = ['A', 'D', 'C']
    else:
        algorithms_to_run = args.algorithms

    print(f"🎯 Testing algorithms: {', '.join(algorithms_to_run)}")
    # Determine maps to test
    if args.maps:
        map_files = args.maps
        print(f"📋 Using provided maps: {map_files}")
        print("ℹ️ Setup environment ONCE, then all 3 algorithms run automatically")
        print("ℹ️ Press ENTER in map editor to continue to algorithm testing")
    else:
        print("📋 Creating test maps...")
        map_files = []

        for size in args.map_sizes:
            map_file = create_test_map(
                size=(size, size),
                obstacle_density=args.obstacle_density,
                output_path="test_maps"
            )
            map_files.append(map_file)

        print(f"📋 Created {len(map_files)} test maps")

    # Validate map files
    valid_maps = []
    for map_file in map_files:
        if os.path.exists(map_file):
            valid_maps.append(map_file)
            print(f"✅ Found map: {map_file}")
        else:
            print(f"⚠️ Map not found: {map_file}")

    if not valid_maps:
        print("❌ No valid maps found")
        return 1

    if len(valid_maps) > 1:
        print(f"ℹ️ Multiple maps found, testing only FIRST map: {valid_maps[0]}")
        valid_maps = [valid_maps[0]]

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Run comparison
    try:
        print(f"\n🚀 Starting comparison with {len(valid_maps)} maps...")
        print(f"⚙️ Configuration:")
        print(f"   • Energy capacity: {args.energy}")
        print(f"   • Dynamic speed: {args.speed}")
        print(f"   • Max steps: {args.max_steps}")
        print(f"   • Output directory: {args.output}")

        comparison = AlgorithmComparison()

        # Configure comparison
        comparison.algorithms_to_run = algorithms_to_run
        comparison.max_steps = args.max_steps
        comparison.no_charts = args.no_charts

        # Run comparison
        results = comparison.run_comparison(
            map_files=valid_maps,
            energy_capacity=args.energy,
            dynamic_speed=args.speed,
            output_dir=args.output
        )

        print(f"\n✅ Comparison completed successfully!")
        print(f"📊 Results saved in: {args.output}")

        # Print quick summary
        print(f"\n📈 QUICK SUMMARY:")
        print("=" * 40)

        successful_runs = 0
        total_runs = 0

        for map_result in results:
            map_name = os.path.basename(map_result['map_file'])
            print(f"\n📋 {map_name}:")

            for alg_name in algorithms_to_run:
                alg_key = f'Project {alg_name}'
                total_runs += 1

                if alg_key in map_result['algorithms']:
                    alg_result = map_result['algorithms'][alg_key]
                    if alg_result.get('success', False):
                        successful_runs += 1
                        print(f"  ✅ Project {alg_name}: {alg_result['execution_time']:.2f}s")
                    else:
                        print(f"  ❌ Project {alg_name}: Failed")
                else:
                    print(f"  ⚠️ Project {alg_name}: Not run")

        success_rate = (successful_runs / total_runs * 100) if total_runs > 0 else 0
        print(f"\n🎯 Success rate: {successful_runs}/{total_runs} ({success_rate:.1f}%)")

        return 0

    except Exception as e:
        print(f"\n❌ Comparison failed with error:")
        print(f"   {type(e).__name__}: {e}")
        if args.headless:
            print(f"\n🔍 Full traceback:")
            traceback.print_exc()
        return 1


def quick_test():
    """Run a quick test with minimal configuration"""
    print("🧪 RUNNING QUICK TEST")
    print("=" * 30)

    # Setup environment
    if not setup_comparison_environment():
        print("❌ Quick test failed - environment setup")
        return False

    # Create a small test map
    test_map = create_test_map(
        size=(10, 10),
        obstacle_density=0.1,
        output_path="quick_test"
    )

    # Run minimal comparison
    try:
        comparison = AlgorithmComparison()
        comparison.algorithms_to_run = ['A']  # Test only Project A
        comparison.max_steps = 100  # Minimal steps

        results = comparison.run_comparison(
            map_files=[test_map],
            energy_capacity=200,
            dynamic_speed=0.05,
            output_dir="quick_test_results"
        )

        print("✅ Quick test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False


if __name__ == "__main__":
    # Check if running in test mode
    if len(sys.argv) > 1 and sys.argv[1] == '--quick-test':
        sys.exit(0 if quick_test() else 1)
    else:
        sys.exit(main())