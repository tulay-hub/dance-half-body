import argparse
import os

import joblib
import numpy as np


MODEL_NAMES = ("pos_pr2ul", "vel_pr2ul", "pos_ul2pr", "vel_ul2pr")


def export_models(model_dir: str, output_name: str = "ankle_poly_models.npz") -> str:
    exported = {}
    for model_name in MODEL_NAMES:
        model = joblib.load(os.path.join(model_dir, f"{model_name}.pkl"))
        exported[f"{model_name}_powers"] = model.named_steps["poly"].powers_.astype(np.int64)
        exported[f"{model_name}_coef"] = model.named_steps["ridge"].coef_.astype(np.float32)
        exported[f"{model_name}_intercept"] = model.named_steps["ridge"].intercept_.astype(np.float32)

    output_path = os.path.join(model_dir, output_name)
    np.savez(output_path, **exported)
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--output-name", default="ankle_poly_models.npz")
    args = parser.parse_args()

    output_path = export_models(args.model_dir, args.output_name)
    print(f"Exported polynomial ankle models to {output_path}")


if __name__ == "__main__":
    main()

