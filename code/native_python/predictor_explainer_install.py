"""Install Predictor Explainer Python dependencies for JMP 19.

This file is submitted from JSL with Python Submit File(). It intentionally
does not read or write any project data files.

Optional global supplied by JSL:
    pe_install_update  (1 = also update already-working packages to the latest
                        versions; 0 or absent = leave working packages alone)
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
    {"spec": "scikit-learn", "dist": "scikit-learn", "import_name": "sklearn", "only_binary": True},
    {"spec": "shap", "dist": "shap", "import_name": "shap", "only_binary": True},
]

# LightGBM is the preferred (faster) model, but on Mac its wheel needs the
# OpenMP runtime (brew install libomp); without it the analysis falls back
# to scikit-learn's RandomForest, so a LightGBM failure is not fatal.
OPTIONAL_PACKAGES = [
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


def install_and_verify(package, update=False):
    satisfied, installed = _is_satisfied(package)
    force_reinstall = False

    if satisfied:
        try:
            _verify_import(package["import_name"])
            if not update:
                print(f"{package['dist']} {installed} already installed")
                return None
            print(f"{package['dist']} {installed} installed; updating to the latest version")
        except Exception:
            # A plain "pip install" would answer "Requirement already satisfied"
            # and change nothing, so force pip to replace the broken install.
            print(f"{package['dist']} is installed but could not be imported; reinstalling")
            print(traceback.format_exc())
            force_reinstall = True

    command_args = ["install"]
    if force_reinstall:
        # replace only the broken package itself; upgrading its dependencies
        # here can break other packages (e.g. numpy past what numba supports)
        command_args.extend(["--force-reinstall", "--no-deps"])
    elif update:
        command_args.append("--upgrade")
    if package["only_binary"]:
        command_args.append("--only-binary=:all:")

    print(f"Installing {package['spec']}...")
    _run_pip(command_args, [package["spec"]])
    _verify_import(package["import_name"])
    installed = _installed_version(package["dist"]) or "unknown version"
    print(f"{package['dist']} {installed} installed")
    return None


def _failure_hint(error_text):
    if "libomp" in error_text and sys.platform == "darwin":
        return (
            "To enable the faster LightGBM on Mac, install the OpenMP runtime once by "
            "running 'brew install libomp' in Terminal (requires Homebrew, see brew.sh). "
            "No reinstall is needed afterwards."
        )
    return None


def _install_group(packages, update=False):
    failures = []
    hints = []
    for package in packages:
        try:
            install_and_verify(package, update=update)
        except Exception:
            failures.append(package["spec"])
            error_text = traceback.format_exc()
            print(f"Package setup failed for {package['spec']}")
            print(error_text)
            hint = _failure_hint(error_text)
            if hint and hint not in hints:
                hints.append(hint)
    return failures, hints


try:
    pe_do_update = bool(globals().get("pe_install_update", 0))
    pe_install_failures, pe_install_hints = _install_group(REQUIRED_PACKAGES, update=pe_do_update)
    pe_optional_failures, pe_optional_hints = _install_group(OPTIONAL_PACKAGES, update=pe_do_update)

    if pe_install_failures:
        pe_install_ok = 0
        pe_install_message = (
            "Required package installation failed: "
            + ", ".join(pe_install_failures)
            + ". Review the JMP log for pip output."
        )
        if pe_install_hints:
            pe_install_message += " " + " ".join(pe_install_hints)
    elif pe_optional_failures:
        pe_install_ok = 1
        pe_install_message = (
            "Core packages are installed, but LightGBM is not available, so the SHAP "
            "analysis will use the slower scikit-learn RandomForest instead."
        )
        if pe_optional_hints:
            pe_install_message += " " + " ".join(pe_optional_hints)
    elif pe_do_update:
        pe_install_ok = 1
        pe_install_message = "Required Python packages are updated to the latest versions."
    else:
        pe_install_ok = 1
        pe_install_message = "Required Python packages are installed."
except Exception:
    pe_install_ok = 0
    pe_install_message = "Unexpected package installation failure. Review the JMP log."
    print(traceback.format_exc())

print(pe_install_message)
