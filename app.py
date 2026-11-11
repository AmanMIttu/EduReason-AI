
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

try:
    import shap
except Exception as exc:
    shap = None
    SHAP_IMPORT_ERROR = exc
else:
    SHAP_IMPORT_ERROR = None

# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "first_semester_model.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.joblib"
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "data.csv"

INPUT_GROUPS = {
    "Demographics": [
        "Marital status", "Age at enrollment", "Gender", "Nacionality",
        "Displaced", "Educational special needs", "International",
        "Daytime/evening attendance\t",
    ],
    "Admission information": [
        "Application mode", "Application order", "Course",
        "Previous qualification", "Previous qualification (grade)",
        "Admission grade", "Unemployment rate", "Inflation rate", "GDP",
    ],
    "Family background": [
        "Mother's qualification", "Father's qualification",
        "Mother's occupation", "Father's occupation",
    ],
    "Financial information": [
        "Debtor", "Tuition fees up to date", "Scholarship holder",
    ],
    "First-semester academic performance": [
        "Curricular units 1st sem (credited)",
        "Curricular units 1st sem (enrolled)",
        "Curricular units 1st sem (evaluations)",
        "Curricular units 1st sem (approved)",
        "Curricular units 1st sem (grade)",
        "Curricular units 1st sem (without evaluations)",
    ],
}

BINARY_LABELS = {
    "Daytime/evening attendance\t": {0: "Evening", 1: "Daytime"},
    "Gender": {0: "Female", 1: "Male"},
}


@st.cache_data
def load_reference_data():
    """Load source records used to populate valid form values and sample input."""
    return pd.read_csv(DATA_PATH, sep=";")


@st.cache_resource
def load_metadata():
    return joblib.load(METADATA_PATH)


def _python_scalar(value):
    return value.item() if isinstance(value, np.generic) else value


def _numeric_default(reference, feature):
    value = reference[feature].median()
    if pd.api.types.is_integer_dtype(reference[feature].dtype):
        return int(round(value))
    return float(value)


def _feature_default(reference, feature, binary_features, categorical_features):
    if feature in binary_features or feature in categorical_features:
        return _python_scalar(reference[feature].mode(dropna=True).iloc[0])
    return _numeric_default(reference, feature)


def _format_code(value):
    return f"Code {value}"


def _feature_label(feature):
    return feature.rstrip()


def build_student_frame(student, feature_names):
    """Build an exact model input; reject missing or unexpected values."""
    missing = [feature for feature in feature_names if feature not in student]
    extra = [feature for feature in student if feature not in feature_names]
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing: {missing}")
        if extra:
            details.append(f"unexpected: {extra}")
        raise ValueError("Student input does not match model features (" + "; ".join(details) + ").")
    if any(pd.isna(student[feature]) for feature in feature_names):
        raise ValueError("Every model feature must have a value before prediction.")
    return pd.DataFrame([{feature: student[feature] for feature in feature_names}], columns=feature_names)

st.set_page_config(
    page_title="EduReason AI",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 EduReason AI")
st.subheader("Explainable Student Outcome Prediction")
st.write(
    "Predict student outcomes using machine learning, "
    "SHAP explanations, and transparent symbolic reasoning."
)

# --------------------------------------------------
# Load model
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_shap_explainer():
    """Cache a TreeExplainer for the fitted Random Forest in the saved pipeline."""
    if shap is None:
        raise RuntimeError(f"SHAP could not be imported: {SHAP_IMPORT_ERROR}")

    fitted_model = load_model()
    classifier = fitted_model.named_steps.get("classifier")
    if classifier is None or not hasattr(classifier, "feature_importances_"):
        raise TypeError("The saved pipeline does not contain a supported tree classifier.")

    return shap.TreeExplainer(classifier)


def _feature_index_groups(preprocessor):
    """Map transformed pipeline columns back to original input feature names."""
    groups = {}
    for name, transformer, columns in preprocessor.transformers_:
        if name == "remainder" or transformer == "drop":
            continue
        columns = list(columns) if not isinstance(columns, str) else [columns]
        output_slice = preprocessor.output_indices_[name]

        if name == "cat":
            encoder = preprocessor.named_transformers_[name]
            cursor = output_slice.start
            infrequent = getattr(encoder, "infrequent_categories_", None)
            dropped = getattr(encoder, "drop_idx_", None)
            for i, (feature, categories) in enumerate(zip(columns, encoder.categories_)):
                width = len(categories)
                if infrequent is not None and infrequent[i] is not None:
                    width -= len(infrequent[i]) - 1
                if dropped is not None and dropped[i] is not None:
                    width -= 1
                groups[feature] = np.arange(cursor, cursor + width, dtype=int)
                cursor += width
            if cursor != output_slice.stop:
                raise ValueError("Unable to map encoded categorical columns to their source features.")
        else:
            if output_slice.stop - output_slice.start != len(columns):
                raise ValueError(f"Unexpected transformed feature count for pipeline step `{name}`.")
            for offset, feature in enumerate(columns):
                groups[feature] = np.array([output_slice.start + offset], dtype=int)

    return groups


def explain_student_prediction(fitted_model, student_df, predicted_class):
    """Return signed SHAP contributions grouped by original student features."""
    if shap is None:
        raise RuntimeError(f"SHAP could not be imported: {SHAP_IMPORT_ERROR}")

    preprocessor = fitted_model.named_steps["preprocessor"]
    classifier = fitted_model.named_steps["classifier"]
    transformed = preprocessor.transform(student_df)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed = np.asarray(transformed, dtype=np.float64)

    explanation = load_shap_explainer()(transformed)
    values = np.asarray(explanation.values)
    class_index = np.flatnonzero(np.asarray(classifier.classes_) == predicted_class)
    if len(class_index) != 1:
        raise ValueError(f"Could not identify the predicted class `{predicted_class}` in the model outputs.")
    class_index = int(class_index[0])

    if values.ndim == 3:
        local_values = values[0, :, class_index]
    elif values.ndim == 2 and len(classifier.classes_) == 1:
        local_values = values[0]
    else:
        raise ValueError(f"Unexpected SHAP output shape: {values.shape}.")

    groups = _feature_index_groups(preprocessor)
    if len(local_values) != len(preprocessor.get_feature_names_out()):
        raise ValueError("SHAP values do not match the pipeline's transformed features.")

    contributions = []
    for feature in student_df.columns:
        indices = groups.get(feature)
        if indices is None:
            raise ValueError(f"No transformed-feature mapping found for `{feature}`.")
        contributions.append({
            "Feature": feature,
            "Student value": student_df.iloc[0][feature],
            "SHAP contribution": float(local_values[indices].sum()),
        })
    result = pd.DataFrame(contributions)
    result["Influence"] = np.where(
        result["SHAP contribution"] > 0,
        "Supports predicted outcome",
        np.where(result["SHAP contribution"] < 0, "Opposes predicted outcome", "No contribution"),
    )
    return result.reindex(result["SHAP contribution"].abs().sort_values(ascending=False).index).head(10)


def global_feature_importance(fitted_model, feature_names):
    """Aggregate fitted Random Forest impurity importance to source features."""
    preprocessor = fitted_model.named_steps["preprocessor"]
    classifier = fitted_model.named_steps["classifier"]
    importances = np.asarray(classifier.feature_importances_, dtype=float)
    groups = _feature_index_groups(preprocessor)
    values = []
    for feature in feature_names:
        indices = groups.get(feature)
        if indices is None:
            raise ValueError(f"No transformed-feature mapping found for `{feature}`.")
        values.append({"Feature": feature, "Importance": float(importances[indices].sum())})
    return pd.DataFrame(values).sort_values("Importance", ascending=False).head(10)

model = load_model()

# --------------------------------------------------
# Student input
# --------------------------------------------------

metadata = load_metadata()
reference = load_reference_data()
feature_names = list(metadata["feature_names"])
binary_features = list(metadata.get("binary_features", []))
categorical_features = list(metadata.get("categorical_features", []))
numerical_features = list(metadata.get("numerical_features", []))

if len(feature_names) != 30:
    st.error(f"Expected 30 model features, but metadata defines {len(feature_names)}.")
    st.stop()
if not set(feature_names).issubset(reference.columns) or "Target" not in reference.columns:
    st.error("The source dataset does not contain all model features and the Target column.")
    st.stop()
if set(binary_features) | set(categorical_features) | set(numerical_features) != set(feature_names):
    st.error("Model metadata does not classify every required feature exactly once.")
    st.stop()
if sum(len(features) for features in INPUT_GROUPS.values()) != len(feature_names) or set(
    feature for features in INPUT_GROUPS.values() for feature in features
) != set(feature_names):
    st.error("The input groups do not match the saved model's required features.")
    st.stop()

st.header("Student Information")
st.caption("All 30 model inputs are required. Defaults and choices come from the original training dataset.")

if st.button("Load Sample Student", key="load_sample_student"):
    sample = reference.sample(n=1, random_state=42).iloc[0]
    for index, feature in enumerate(feature_names):
        st.session_state[f"student_feature_{index}"] = _python_scalar(sample[feature])
    st.session_state["sample_student_loaded"] = True
if st.session_state.get("sample_student_loaded"):
    st.info("A real student record is loaded. Edit any field before submitting; the target outcome is not shown or used as an input.")

with st.form("student_form"):
    for group_name, group_features in INPUT_GROUPS.items():
        st.subheader(group_name)
        columns = st.columns(2)
        for offset, feature in enumerate(group_features):
            index = feature_names.index(feature)
            key = f"student_feature_{index}"
            widget = columns[offset % 2]
            default = _feature_default(reference, feature, binary_features, categorical_features)
            label = _feature_label(feature)

            with widget:
                if feature in binary_features:
                    values = sorted(_python_scalar(v) for v in reference[feature].dropna().unique())
                    if values != [0, 1]:
                        st.error(f"Expected binary dataset values 0 and 1 for {label}.")
                        st.stop()
                    labels = BINARY_LABELS.get(feature, {0: "No", 1: "Yes"})
                    select_kwargs = {} if key in st.session_state else {"index": values.index(default)}
                    st.selectbox(
                        label,
                        options=values,
                        format_func=lambda value, mapping=labels: mapping[int(value)],
                        key=key,
                        **select_kwargs,
                    )
                elif feature in categorical_features:
                    options = sorted(_python_scalar(v) for v in reference[feature].dropna().unique())
                    if not options:
                        st.error(f"No valid dataset categories are available for {label}.")
                        st.stop()
                    if key in st.session_state and st.session_state[key] not in options:
                        st.session_state[key] = default
                    select_kwargs = {} if key in st.session_state else {"index": options.index(default)}
                    st.selectbox(
                        label,
                        options=options,
                        format_func=_format_code,
                        key=key,
                        **select_kwargs,
                    )
                elif feature in numerical_features:
                    series = reference[feature]
                    is_integer = pd.api.types.is_integer_dtype(series.dtype)
                    minimum = int(series.min()) if is_integer else float(series.min())
                    maximum = int(series.max()) if is_integer else float(series.max())
                    if key in st.session_state:
                        current = st.session_state[key]
                        if current < minimum or current > maximum:
                            st.session_state[key] = _numeric_default(reference, feature)
                    number_kwargs = {} if key in st.session_state else {"value": default}
                    st.number_input(
                        label,
                        min_value=minimum,
                        max_value=maximum,
                        step=1 if is_integer else 0.01,
                        key=key,
                        help=f"Observed dataset range: {minimum} to {maximum}.",
                        **number_kwargs,
                    )
                else:
                    st.error(f"No input control is defined for {label}.")
                    st.stop()

    submitted = st.form_submit_button("Predict Student Outcome", type="primary", key="predict_student")

student = {
    feature: st.session_state.get(f"student_feature_{index}")
    for index, feature in enumerate(feature_names)
}

# --------------------------------------------------
# Symbolic reasoning
# --------------------------------------------------

def generate_reasoning(student):

    reasons = []

    approved = student["Curricular units 1st sem (approved)"]
    grade = student["Curricular units 1st sem (grade)"]
    enrolled = student["Curricular units 1st sem (enrolled)"]
    evaluations = student["Curricular units 1st sem (evaluations)"]

    if approved <= 2:
        reasons.append(
            "Very few first-semester units approved."
        )

    if grade < 8:
        reasons.append(
            "First-semester grade is below 8."
        )

    if enrolled > 0 and approved / enrolled < 0.5:
        reasons.append(
            "Fewer than half of enrolled units were approved."
        )

    if evaluations > 0 and approved == 0:
        reasons.append(
            "Evaluations occurred but no units were approved."
        )

    if student["Debtor"] == 1:
        reasons.append(
            "Student has an outstanding debt."
        )

    if student["Tuition fees up to date"] == 0:
        reasons.append(
            "Tuition fees are not up to date."
        )

    if not reasons:
        reasons.append(
            "No warning indicators triggered by current rules."
        )

    return reasons

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if submitted:
    try:
        student_df = build_student_frame(student, feature_names)
        prediction = model.predict(student_df)[0]
        probabilities = model.predict_proba(student_df)[0]
    except Exception as exc:
        st.error(f"Prediction could not be generated: {type(exc).__name__}: {exc}")
    else:
        st.success(f"All {student_df.shape[1]} required features supplied. Predicted outcome: {prediction}")

        probability_df = pd.DataFrame({
            "Outcome": model.classes_,
            "Probability": probabilities
        })

        st.header("Prediction Results")
        st.bar_chart(
            probability_df.set_index("Outcome"),
            alt="Predicted probability for each student outcome",
        )

        st.header("SHAP Explanation")
        try:
            local_importance = explain_student_prediction(model, student_df, prediction)
            st.caption(
                f"These are the top 10 original student features influencing the predicted class, **{prediction}**. "
                "Positive SHAP contributions support that outcome; negative contributions oppose it."
            )
            local_chart = local_importance.assign(
                **{
                    "Toward predicted outcome": local_importance["SHAP contribution"].clip(lower=0),
                    "Away from predicted outcome": local_importance["SHAP contribution"].clip(upper=0),
                }
            )
            st.bar_chart(
                local_chart,
                x="Feature",
                y=["Toward predicted outcome", "Away from predicted outcome"],
                horizontal=True,
                sort=False,
                stack=False,
                alt="Top positive and negative SHAP contributions for the predicted student outcome",
            )
            st.dataframe(
                local_importance,
                hide_index=True,
                width="stretch",
                alt="Top 10 student features and their signed SHAP contributions",
            )

            global_importance = global_feature_importance(model, feature_names)
            st.subheader("Global feature importance")
            st.caption(
                "This Random Forest importance summarizes impurity reduction across the fitted trees. "
                "It is a global model statistic, not a directional SHAP contribution."
            )
            st.bar_chart(
                global_importance,
                x="Feature",
                y="Importance",
                horizontal=True,
                sort=False,
                alt="Top 10 global Random Forest feature importances",
            )
        except Exception as exc:
            st.warning(
                f"SHAP explanation is unavailable for this prediction ({type(exc).__name__}: {exc}). "
                "The model prediction, probabilities, and symbolic reasoning are still available."
            )

        st.header("Symbolic Reasoning")
        for reason in generate_reasoning(student):
            st.write("•", reason)

        st.header("Prediction Confidence")
        st.dataframe(
            probability_df,
            width="stretch",
            alt="Model probabilities for Dropout, Enrolled, and Graduate outcomes",
        )

st.divider()

st.caption(
    "EduReason AI | Explainable Machine Learning "
    "and Symbolic Student Outcome Reasoning"
)


