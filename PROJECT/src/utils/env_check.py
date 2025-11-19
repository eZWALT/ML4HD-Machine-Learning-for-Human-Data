
"""
Environment verification script.
Run this after setting up the conda environment to verify all dependencies.
"""

import importlib
import sys

def check_package(package_name, import_name=None):
    """Check if a package is properly installed"""
    import_name = import_name or package_name
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown version')
        print(f"{package_name}: {version}")
        return True
    except ImportError as e:
        print(f"{package_name}: Failed to import - {e}")
        return False
    except Exception as e:
        print(f"{package_name}: Imported with warnings - {e}")
        return True

def main():
    print("Checking environment setup...")
    print("=" * 50)
    
    # Check Python version
    print(f"Python: {sys.version}")
    
    # Core packages
    packages = [
        ("polars", "polars"),
        ("pyodbc", "pyodbc"),
        ("pandas", "pandas"),
        ("jupyter", "jupyter"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("scikit-learn", "sklearn"),
        ("numpy", "numpy"),
    ]
    
    print("\nPackages:")
    core_success = all(check_package(pkg, imp) for pkg, imp in packages)
    
    
    print("\n" + "=" * 50)
    
    
    if core_success:
        print("\n Environment is ready!")
    else:
        print("\n Some core packages failed to load.")
        print("   Please check your environment setup.")
        sys.exit(1)

if __name__ == "__main__":
    main()