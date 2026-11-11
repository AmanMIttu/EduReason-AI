"""Streamlit regression tests for EduReason AI's complete student form."""

import joblib
import pandas as pd
import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "app.py"
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "data.csv"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.joblib"


class StudentFormAppTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata = joblib.load(METADATA_PATH)
        cls.features = cls.metadata["feature_names"]
        cls.data = pd.read_csv(DATA_PATH, sep=";")

    def _new_app(self):
        app = AppTest.from_file(str(APP_PATH), default_timeout=90).run()
        self.assertEqual([], [error.message for error in app.exception])
        self.assertEqual(30, len(app.number_input) + len(app.selectbox))
        self.assertEqual(30, len(self.features))
        for index in range(30):
            self.assertIsNotNone(app.get_by_key(f"student_feature_{index}"))
        return app

    def _assert_prediction_succeeded(self, app):
        app.form_submit_button(key="predict_student").click().run(timeout=120)
        self.assertEqual([], [error.message for error in app.exception])
        self.assertTrue(any("All 30 required features supplied" in item.value for item in app.success))
        headers = [item.value for item in app.header]
        self.assertIn("Prediction Results", headers)
        self.assertIn("SHAP Explanation", headers)
        self.assertIn("Symbolic Reasoning", headers)
        self.assertIn("Prediction Confidence", headers)
        self.assertEqual(2, len(app.dataframe))
        shapes = [item.value.shape for item in app.dataframe]
        self.assertIn((10, 4), shapes)  # Local SHAP table
        self.assertIn((3, 2), shapes)   # Probabilities for the three classes

    def test_load_sample_student_and_predict_all_30_features(self):
        app = self._new_app()
        app.button(key="load_sample_student").click().run(timeout=90)
        sample = self.data.sample(n=1, random_state=42).iloc[0]
        for index, feature in enumerate(self.features):
            self.assertEqual(
                sample[feature],
                app.get_by_key(f"student_feature_{index}").value,
                msg=f"Sample value mismatch for {feature!r}",
            )
        self.assertNotIn("Target", [widget.label for widget in app.number_input])
        self.assertNotIn("Target", [widget.label for widget in app.selectbox])
        self._assert_prediction_succeeded(app)

    def test_manually_entered_complete_dataset_record_and_predict(self):
        app = self._new_app()
        manual_record = self.data.iloc[1]
        for index, feature in enumerate(self.features):
            app.get_by_key(f"student_feature_{index}").set_value(manual_record[feature])
        for index, feature in enumerate(self.features):
            self.assertEqual(
                manual_record[feature],
                app.get_by_key(f"student_feature_{index}").value,
                msg=f"Manual value mismatch for {feature!r}",
            )
        self._assert_prediction_succeeded(app)


if __name__ == "__main__":
    unittest.main()

