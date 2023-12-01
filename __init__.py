import sys
import subprocess
import importlib

def ensure_package(package, install_package_name=None):
    # Try to import the package
    try:
        importlib.import_module(package)
    except ImportError:
        print(f"Package {package} is not installed. Installing now...")

        if "python_embeded" in sys.executable or "python_embedded" in sys.executable:
            pip_install = [sys.executable, "-s", "-m", "pip", "install"]
        else:
            pip_install = [sys.executable, "-m", "pip", "install"]

        subprocess.check_call(pip_install + [install_package_name or package])
#    else:
#        print(f"Package {package} is already installed.")

ensure_package("dotenv", "python-dotenv")
ensure_package("argon2", "argon2-cffi")


from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
