"""Install Predictor Explainer Python dependencies for JMP 19.

This file is submitted from JSL with Python Submit File(). It intentionally
does not read or write any project data files.
"""

import importlib.metadata
import re
import subprocess
import sys
import traceback

try:
    import jmputils
except ImportError:
    jmputils = None


REQUIRED_PACKAGES = [
    {"spec": "numpy", "dist": "numpy", "import_name": "numpy", "only_binary": True},
    {"spec": "pandas", "dist": "pandas", "import_name": "pandas", "only_binary": True},
    {"spec": "shap", "dist": "shap", "import_name": "shap", "only_binary": True},
    {"spec": "lightgbm", "dist": "lightgbm", "import_name": "lightgbm", "only_binary": True},
]


def _version_tuple(value):
    parts = re.findall(r"\d+", str(value).split("+")[0])
    return tuple(int(part) for part in parts[:4])


def _minimum_version(spec):
    match = re.search(r">=\s*([^,; ]+)", spec)
    return match.group(1) if match else None


def _installed_version(dist_name):
    try:
        if jmputils is not None:
            version = jmputils.package_version(dist_name)
            if version:
                return str(version)
        return importlib.metadata.version(dist_name)
    except Exception:
        return None


def _is_satisfied(package):
    installed = _installed_version(package["dist"])
    if installed is None:
        return False, None

    minimum = _minimum_version(package["spec"])
    if minimum is None:
        return True, installed

    return _version_tuple(installed) >= _version_tuple(minimum), installed


def _run_pip(command_args, package_args):
    if jmputils is not None:
        result = jmputils.jpip(command_args, package_args)
        result.check_returncode()
        return

    subprocess.check_call([sys.executable, "-m", "pip"] + command_args + package_args)


def _verify_import(import_name):
    __import__(import_name)


def install_and_verify(package):
    satisfied, installed = _is_satisfied(package)
    if satisfied:
        try:
            _verify_import(package["import_name"])
            print(f"{package['dist']} {installed} already installed")
            return None
        except Exception:
            print(f"{package['dist']} is installed but could not be imported; reinstalling")
            print(traceback.format_exc())

    command_args = ["install"]
    if package["only_binary"]:
        command_args.append("--only-binary=:all:")

    print(f"Installing {package['spec']}...")
    _run_pip(command_args, [package["spec"]])
    _verify_import(package["import_name"])
    installed = _installed_version(package["dist"]) or "unknown version"
    print(f"{package['dist']} {installed} installed")
    return None


def _install_group(packages):
    failures = []
    for package in packages:
        try:
            install_and_verify(package)
        except Exception:
            failures.append(package["spec"])
            print(f"Package setup failed for {package['spec']}")
            print(traceback.format_exc())
    return failures


try:
    pe_install_failures = _install_group(REQUIRED_PACKAGES)

    if pe_install_failures:
        pe_install_ok = 0
        pe_install_message = (
            "Required package installation failed: "
            + ", ".join(pe_install_failures)
            + ". Review the JMP log for pip output."
        )
    else:
        pe_install_ok = 1
        pe_install_message = "Required Python packages are installed."
except Exception:
    pe_install_ok = 0
    pe_install_message = "Unexpected package installation failure. Review the JMP log."
    print(traceback.format_exc())

print(pe_install_message)
