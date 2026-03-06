"""Example script demonstrating Parser functionality."""

import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from src.data.parser import Parser, ConfigurationError, DataFormatError

def main():
    """Demonstrate Parser functionality."""
    parser = Parser()
    
    # Example 1: Parse configuration file
    print("=" * 60)
    print("Example 1: Parsing configuration file")
    print("=" * 60)
    
    config_path = "configs/example_config.json"
    try:
        config = parser.parse_config(config_path)
        print(f"✓ Successfully parsed configuration: {config.experiment_name}")
        print(f"  - Forecast horizon: {config.forecast_horizon.value}")
        print(f"  - Model type: {config.model_configuration.model_type.value}")
        print(f"  - Microgrid mode: {config.microgrid_configuration.mode.value}")
        print(f"  - PV capacity: {config.microgrid_configuration.pv_capacity_kw} kW")
        print(f"  - Battery capacity: {config.microgrid_configuration.battery_capacity_kwh} kWh")
    except ConfigurationError as e:
        print(f"✗ Configuration error: {e}")
    
    # Example 2: Validate configuration
    print("\n" + "=" * 60)
    print("Example 2: Validating configuration")
    print("=" * 60)
    
    try:
        config = parser.parse_config(config_path)
        result = parser.validate_config(config)
        
        if result.valid:
            print("✓ Configuration is valid")
        else:
            print("✗ Configuration validation failed:")
            for error in result.errors:
                print(f"  - {error}")
    except ConfigurationError as e:
        print(f"✗ Configuration error: {e}")
    
    # Example 3: Handle invalid configuration
    print("\n" + "=" * 60)
    print("Example 3: Handling invalid configuration")
    print("=" * 60)
    
    try:
        parser.parse_config("nonexistent.json")
    except ConfigurationError as e:
        print(f"✓ Caught expected error: {e}")
    
    print("\n" + "=" * 60)
    print("Parser demonstration complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
