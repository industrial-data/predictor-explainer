"""Build Latest_PredictorExplainer.jmpaddin from the JMP App source.

Packages the app source (code/pred_explainer_addin_*.jmpappsource), the
native_python scripts and the example data tables into the .jmpaddin zip,
reproducing the format JMP's Add-In Builder exports (Addin.def plus an
addin.jmpcust menu file that embeds the app source as an XML text action).

Usage (from anywhere, no third-party packages needed):
    python3 code/build_addin.py
"""

import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

REPO_ROOT = Path(__file__).resolve().parent.parent

APP_SOURCE = REPO_ROOT / "code" / "pred_explainer_addin_v2.0.jmpappsource"
OUTPUT = REPO_ROOT / "Latest_PredictorExplainer.jmpaddin"

DISPLAY_VERSION = "v.2.0"
ADDIN_VERSION = "260705"  # numeric date so JMP replaces older installed versions
MIN_JMP_VERSION = "19"  # embedded Python with jmp/jmputils modules

PYTHON_SCRIPTS = [
    REPO_ROOT / "code" / "native_python" / "predictor_explainer_install.py",
    REPO_ROOT / "code" / "native_python" / "predictor_explainer_shap.py",
]

DATA_FILES = [
    REPO_ROOT / "data" / "dist_tower_na.jmp",
    REPO_ROOT / "data" / "dryer_dataset_imanol_et_al_2022.jmp",
    REPO_ROOT / "data" / "dryer_KPI_dataset_imanol_et_al_2022.jmp",
    REPO_ROOT / "data" / "Fermentation_Batch_Process.jmp",
    REPO_ROOT / "data" / "MiningProcess_Flotation_hourly_data.jmp",
    REPO_ROOT / "data" / "Tennesse MDMSPC - PCA - Eastman Chemical.jmp",
]

ADDIN_DEF = (
    "id=Predictor_Explainer\n"
    "name=Predictor Explainer\n"
    "supportJmpSE=0\n"
    f"addinVersion={ADDIN_VERSION}\n"
    f"minJmpVersion={MIN_JMP_VERSION}"
)

JMPCUST_TEMPLATE = """<!-- JMP Add-In Builder created --><jm:menu_and_toolbar_customizations xmlns:jm="http://www.jmp.com/ns/menu" version="3">
  <jm:insert_in_main_menu>
    <jm:insert_in_menu>
      <jm:name>ADD-INS</jm:name>
      <jm:insert_after>
        <jm:name></jm:name>
        <jm:command>
          <jm:name>PREDICTOR EXPLAINER</jm:name>
          <jm:caption>Predictor Explainer</jm:caption>
          <jm:action type="text">{action}</jm:action>
          <jm:tip>{tip}</jm:tip>
          <jm:icon type="none"></jm:icon>
        </jm:command>
      </jm:insert_after>
    </jm:insert_in_menu>
  </jm:insert_in_main_menu>
</jm:menu_and_toolbar_customizations>
"""


def build():
    app_source = APP_SOURCE.read_text(encoding="utf-8-sig")
    jmpcust = JMPCUST_TEMPLATE.format(
        action=escape(app_source + " << Run"),
        tip=DISPLAY_VERSION,
    )

    with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED) as addin:
        addin.writestr("Addin.def", ADDIN_DEF)
        addin.writestr("addin.jmpcust", jmpcust)
        for script in PYTHON_SCRIPTS:
            addin.write(script, f"native_python/{script.name}")
        for data_file in DATA_FILES:
            addin.write(data_file, data_file.name)

    size_mb = OUTPUT.stat().st_size / 1e6
    print(f"Built {OUTPUT} ({size_mb:.1f} MB, {DISPLAY_VERSION}, addinVersion {ADDIN_VERSION})")


if __name__ == "__main__":
    build()
