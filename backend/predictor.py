from pathlib import Path
import json
import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "fit_model.pkl"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"


class FlowFitPredictor:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}\n"
                "Run scripts/train_model.py first."
            )

        self.pipeline = joblib.load(MODEL_PATH)

        self.metadata = {}
        if METADATA_PATH.exists():
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

    def predict(self, data: dict):
        df = pd.DataFrame([data])

        prediction = self.pipeline.predict(df)[0]

        probabilities = None
        confidence = None

        if hasattr(self.pipeline, "predict_proba"):
            probabilities = self.pipeline.predict_proba(df)[0]

            classes = self.pipeline.classes_

            confidence = float(max(probabilities))

            probability_map = {
                str(cls): round(float(prob), 4)
                for cls, prob in zip(classes, probabilities)
            }
        else:
            probability_map = {}

        fit_class = str(prediction).lower()

        return {
            "fit": fit_class,
            "confidence": round(confidence * 100, 2)
            if confidence is not None
            else None,
            "probabilities": probability_map,
        }


predictor = FlowFitPredictor()