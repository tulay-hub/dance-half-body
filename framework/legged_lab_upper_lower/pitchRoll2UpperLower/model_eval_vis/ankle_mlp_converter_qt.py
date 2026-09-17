"""Qt UI for Lens110 ankle PR/UL conversion.

The UI supports both directions:
    PR -> UL: pitch/roll position+velocity to upper/lower position+velocity
    UL -> PR: upper/lower position+velocity to pitch/roll position+velocity

It can select the old pkl models, weighted MLP models, or poly degree3 models.

Install one Qt binding if needed, for example:
    pip install PySide6

Run:
    python -u pitchRoll2UpperLower/model_eval_vis/ankle_mlp_converter_qt.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from ankle_conversion_backends import (
    DEFAULT_OLD_PKL_DIR,
    DEFAULT_POLY_DEGREE3_DIR,
    DEFAULT_SKLEARN_PYTHON,
    DEFAULT_TORCH_PYTHON,
    DEFAULT_WEIGHTED_MLP_DIR,
    ConversionBackends,
    PR_POS,
    PR_VEL,
    UL_POS,
    UL_VEL,
)

PR_INPUT_NAMES = PR_POS + PR_VEL
UL_INPUT_NAMES = UL_POS + UL_VEL
UL_OUTPUT_TO_JOINT = {
    "left_upper_pos": "left_ankle_upper_joint",
    "left_lower_pos": "left_ankle_lower_joint",
    "right_upper_pos": "right_ankle_upper_joint",
    "right_lower_pos": "right_ankle_lower_joint",
}
PR_OUTPUT_TO_JOINT = {
    "left_pitch_pos": "left_ankle_pitch_joint",
    "left_roll_pos": "left_ankle_roll_joint",
    "right_pitch_pos": "right_ankle_pitch_joint",
    "right_roll_pos": "right_ankle_roll_joint",
}
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_POSE_VIEWER_SCRIPT = SCRIPT_DIR / "view_lens110_initial_poses.py"
DEFAULT_POSE_VIEWER_PYTHON = Path(os.environ.get("LENS110_POSE_VIEWER_PYTHON", sys.executable))
QT_FALLBACK_PYTHON = DEFAULT_POSE_VIEWER_PYTHON


def clean_python_env() -> dict[str, str]:
    env = os.environ.copy()
    # IsaacSim/IsaacLab often injects a Python-version-specific pip_prebundle into
    # PYTHONPATH. Carrying that into another conda env can break numpy extensions.
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    return env


def import_qt():
    try:
        from PySide6 import QtCore, QtWidgets

        return QtCore, QtWidgets
    except Exception:
        pass
    try:
        from PyQt6 import QtCore, QtWidgets

        return QtCore, QtWidgets
    except Exception:
        pass
    try:
        from PySide2 import QtCore, QtWidgets

        return QtCore, QtWidgets
    except Exception:
        pass
    try:
        from PyQt5 import QtCore, QtWidgets

        return QtCore, QtWidgets
    except Exception:
        pass
    if (
        os.environ.get("ANKLE_QT_NO_REEXEC") != "1"
        and os.environ.get("ANKLE_QT_REEXECED") != "1"
        and QT_FALLBACK_PYTHON.exists()
        and Path(sys.executable) != QT_FALLBACK_PYTHON
    ):
        env = clean_python_env()
        env["ANKLE_QT_REEXECED"] = "1"
        print(f"[INFO] current Python has no Qt binding; relaunching with {QT_FALLBACK_PYTHON}")
        os.execve(
            str(QT_FALLBACK_PYTHON),
            [str(QT_FALLBACK_PYTHON), "-B", str(Path(__file__).resolve()), *sys.argv[1:]],
            env,
        )
    raise RuntimeError(
        "No Qt Python binding found. Install one of: PySide6, PyQt6, PySide2, PyQt5. "
        f"Fallback Python checked: {QT_FALLBACK_PYTHON}"
    )


def build_main_window(
    QtCore,
    QtWidgets,
    converter: ConversionBackends,
    pose_viewer_python: str,
    pose_viewer_script: str,
):
    class MainWindow(QtWidgets.QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Lens110 Ankle PR/UL Converter")
            self.resize(860, 520)
            self.pose_viewers: list[subprocess.Popen] = []

            self.backend_combo = QtWidgets.QComboBox()
            self.backend_combo.addItems(converter.labels())
            self.backend_combo.setCurrentText("weighted_mlp")
            self.backend_combo.currentIndexChanged.connect(self.convert)

            self.mode_combo = QtWidgets.QComboBox()
            self.mode_combo.addItems(["PR -> UL", "UL -> PR"])
            self.mode_combo.currentIndexChanged.connect(self.render_names)

            self.convert_button = QtWidgets.QPushButton("Convert")
            self.convert_button.clicked.connect(self.convert)

            self.urdf_default_button = QtWidgets.QPushButton("URDF PR default")
            self.urdf_default_button.clicked.connect(self.set_urdf_default)
            self.deploy_default_button = QtWidgets.QPushButton("Deploy UL 0.29")
            self.deploy_default_button.clicked.connect(self.set_deploy_default)
            self.deploy_stand_button = QtWidgets.QPushButton("Deploy UL 0.26")
            self.deploy_stand_button.clicked.connect(self.set_deploy_stand)
            self.zero_velocity_button = QtWidgets.QPushButton("Zero velocities")
            self.zero_velocity_button.clicked.connect(self.zero_velocities)
            self.view_two_poses_button = QtWidgets.QPushButton("View 2 poses")
            self.view_two_poses_button.clicked.connect(lambda: self.open_pose_viewer(include_converted=False))
            self.view_three_poses_button = QtWidgets.QPushButton("View + converted")
            self.view_three_poses_button.clicked.connect(lambda: self.open_pose_viewer(include_converted=True))
            self.view_current_button = QtWidgets.QPushButton("View current")
            self.view_current_button.clicked.connect(self.open_current_pose_viewer)

            top = QtWidgets.QHBoxLayout()
            top.addWidget(self.backend_combo)
            top.addWidget(self.mode_combo)
            top.addWidget(self.convert_button)
            top.addWidget(self.urdf_default_button)
            top.addWidget(self.deploy_default_button)
            top.addWidget(self.deploy_stand_button)
            top.addWidget(self.zero_velocity_button)
            top.addWidget(self.view_two_poses_button)
            top.addWidget(self.view_three_poses_button)
            top.addWidget(self.view_current_button)
            top.addStretch(1)

            self.input_form = QtWidgets.QFormLayout()
            self.output_form = QtWidgets.QFormLayout()
            self.inputs = []
            self.outputs = []
            for _ in range(8):
                spin = QtWidgets.QDoubleSpinBox()
                spin.setRange(-20.0, 20.0)
                spin.setDecimals(6)
                spin.setSingleStep(0.01)
                spin.setKeyboardTracking(False)
                spin.valueChanged.connect(self.convert)
                self.inputs.append(spin)

                out = QtWidgets.QLineEdit()
                out.setReadOnly(True)
                out.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight if hasattr(QtCore.Qt, "AlignmentFlag") else QtCore.Qt.AlignRight)
                self.outputs.append(out)

            input_box = QtWidgets.QGroupBox("Input")
            input_box.setLayout(self.input_form)
            output_box = QtWidgets.QGroupBox("Output")
            output_box.setLayout(self.output_form)

            columns = QtWidgets.QHBoxLayout()
            columns.addWidget(input_box, 1)
            columns.addWidget(output_box, 1)

            self.status = QtWidgets.QLabel("")
            self.status.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse if hasattr(QtCore.Qt, "TextInteractionFlag") else QtCore.Qt.TextSelectableByMouse)

            layout = QtWidgets.QVBoxLayout(self)
            layout.addLayout(top)
            layout.addLayout(columns)
            layout.addWidget(self.status)

            self.render_names()
            self.set_urdf_default()

        def is_pr_to_ul(self) -> bool:
            return self.mode_combo.currentIndex() == 0

        def input_names(self):
            return PR_INPUT_NAMES if self.is_pr_to_ul() else UL_INPUT_NAMES

        def output_names(self):
            return UL_INPUT_NAMES if self.is_pr_to_ul() else PR_INPUT_NAMES

        def backend(self) -> str:
            return self.backend_combo.currentText()

        def render_names(self):
            self._clear_form(self.input_form)
            self._clear_form(self.output_form)
            for name, widget in zip(self.input_names(), self.inputs):
                self.input_form.addRow(name, widget)
            for name, widget in zip(self.output_names(), self.outputs):
                self.output_form.addRow(name, widget)
            self.convert()

        @staticmethod
        def _clear_form(form):
            while form.count():
                item = form.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)

        def values(self) -> list[float]:
            return [spin.value() for spin in self.inputs]

        def set_values(self, values: list[float]):
            for spin, value in zip(self.inputs, values):
                spin.blockSignals(True)
                spin.setValue(float(value))
                spin.blockSignals(False)
            self.convert()

        def convert(self):
            try:
                mode = "pr_to_ul" if self.is_pr_to_ul() else "ul_to_pr"
                names, values = converter.convert(self.backend(), mode, self.values())
                for out, value in zip(self.outputs, values):
                    out.setText(f"{value:.6f}")
                self.status.setText(f"Model: {converter.describe(self.backend())}")
            except Exception as exc:
                self.status.setText(f"ERROR: {exc}")

        def set_urdf_default(self):
            self.mode_combo.setCurrentIndex(0)
            self.set_values([-0.15, 0.0, -0.15, 0.0, 0.0, 0.0, 0.0, 0.0])

        def set_deploy_default(self):
            self.mode_combo.setCurrentIndex(1)
            self.set_values([0.29, 0.29, 0.29, 0.29, 0.0, 0.0, 0.0, 0.0])

        def set_deploy_stand(self):
            self.mode_combo.setCurrentIndex(1)
            self.set_values([0.26, 0.26, 0.26, 0.26, 0.0, 0.0, 0.0, 0.0])

        def zero_velocities(self):
            for spin in self.inputs[4:]:
                spin.blockSignals(True)
                spin.setValue(0.0)
                spin.blockSignals(False)
            self.convert()

        def open_pose_viewer(self, include_converted: bool):
            self.pose_viewers = [proc for proc in self.pose_viewers if proc.poll() is None]
            cmd = [
                pose_viewer_python,
                "-B",
                pose_viewer_script,
                "--convert_backend",
                self.backend(),
            ]
            if not include_converted:
                cmd.append("--no_converted_ul")
            try:
                proc = subprocess.Popen(cmd, cwd=str(SCRIPT_DIR.parents[1]), env=clean_python_env())
                self.pose_viewers.append(proc)
                mode = "with converted pose" if include_converted else "raw two poses only"
                self.status.setText(f"Started MuJoCo pose viewer ({mode}, backend={self.backend()}), pid={proc.pid}")
            except Exception as exc:
                self.status.setText(f"ERROR starting MuJoCo pose viewer: {exc}")

        def open_current_pose_viewer(self):
            try:
                mode = "pr_to_ul" if self.is_pr_to_ul() else "ul_to_pr"
                names, values = converter.convert(self.backend(), mode, self.values())
                pose = {}
                if self.is_pr_to_ul():
                    for name, value in zip(PR_INPUT_NAMES[:4], self.values()[:4]):
                        pose[PR_OUTPUT_TO_JOINT[name]] = float(value)
                    for name, value in zip(names[:4], values[:4]):
                        pose[UL_OUTPUT_TO_JOINT[name]] = float(value)
                else:
                    for name, value in zip(UL_INPUT_NAMES[:4], self.values()[:4]):
                        pose[UL_OUTPUT_TO_JOINT[name]] = float(value)
                    for name, value in zip(names[:4], values[:4]):
                        pose[PR_OUTPUT_TO_JOINT[name]] = float(value)
                cmd = [
                    pose_viewer_python,
                    "-B",
                    pose_viewer_script,
                    "--custom_pose_json",
                    json.dumps(pose),
                    "--custom_pose_label",
                    f"Current Qt input ({self.backend()}, {'PR -> UL' if self.is_pr_to_ul() else 'UL -> PR'})",
                ]
                proc = subprocess.Popen(cmd, cwd=str(SCRIPT_DIR.parents[1]), env=clean_python_env())
                self.pose_viewers.append(proc)
                self.status.setText(f"Started current-pose MuJoCo viewer, pid={proc.pid}")
            except Exception as exc:
                self.status.setText(f"ERROR starting current-pose viewer: {exc}")

    return MainWindow


def main():
    parser = argparse.ArgumentParser(description="Qt UI for Lens110 ankle PR/UL conversion.")
    parser.add_argument("--weighted_mlp_dir", default=str(DEFAULT_WEIGHTED_MLP_DIR))
    parser.add_argument("--poly_degree3_dir", default=str(DEFAULT_POLY_DEGREE3_DIR))
    parser.add_argument("--old_pkl_dir", default=str(DEFAULT_OLD_PKL_DIR))
    parser.add_argument("--weighted_mlp_python", default=str(DEFAULT_TORCH_PYTHON))
    parser.add_argument("--old_pkl_python", default=str(DEFAULT_SKLEARN_PYTHON))
    parser.add_argument("--pose_viewer_python", default=str(DEFAULT_POSE_VIEWER_PYTHON))
    parser.add_argument("--pose_viewer_script", default=str(DEFAULT_POSE_VIEWER_SCRIPT))
    args = parser.parse_args()

    QtCore, QtWidgets = import_qt()
    app = QtWidgets.QApplication(sys.argv)
    converter = ConversionBackends(
        args.weighted_mlp_dir,
        args.poly_degree3_dir,
        args.old_pkl_dir,
        args.weighted_mlp_python,
        args.old_pkl_python,
    )
    window_cls = build_main_window(
        QtCore,
        QtWidgets,
        converter,
        args.pose_viewer_python,
        args.pose_viewer_script,
    )
    window = window_cls()
    window.show()
    sys.exit(app.exec() if hasattr(app, "exec") else app.exec_())


if __name__ == "__main__":
    main()
