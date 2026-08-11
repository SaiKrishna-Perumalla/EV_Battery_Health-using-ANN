import json
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow import keras


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="EV Battery Health Prediction",
    page_icon="🔋",
    layout="wide"
)


# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------
# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

# Folder where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# artifacts folder inside the project
ARTIFACT_DIR = os.path.join(
    BASE_DIR,
    "artifacts"
)

# Model file
MODEL_PATH = os.path.join(
    ARTIFACT_DIR,
    "battery_soh_ann.keras"
)

# Preprocessor file
PREPROCESSOR_PATH = os.path.join(
    ARTIFACT_DIR,
    "preprocessor.joblib"
)

# Metadata file
METADATA_PATH = os.path.join(
    ARTIFACT_DIR,
    "metadata.json"
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🔋 EV Battery Health Prediction using ANN")

st.write(
    "Predict battery State of Health (SOH %) "
    "from operating and environmental measurements."
)

st.info(
    "Health-status thresholds in this demo are illustrative "
    "and should not be treated as manufacturer-specific "
    "diagnostic limits."
)


# --------------------------------------------------
# CHECK MODEL FILES
# --------------------------------------------------

missing = [
    p
    for p in [
        MODEL_PATH,
        PREPROCESSOR_PATH,
        METADATA_PATH
    ]
    if not os.path.exists(p)
]

if missing:

    st.error("Model artifact files are missing.")

    st.write("Streamlit is looking for:")

    for path in missing:
        st.code(path)

    st.write("Your artifacts folder should contain:")

    st.code(
        """
artifacts/
    battery_soh_ann.keras
    preprocessor.joblib
    metadata.json
        """
    )

    st.stop()


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_assets():

    model = keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        metadata = json.load(f)

    return (
        model,
        preprocessor,
        metadata
    )


model, preprocessor, metadata = load_assets()


# --------------------------------------------------
# HEALTH STATUS FUNCTION
# --------------------------------------------------

def health_status(soh):

    if soh >= 90:
        return "Excellent"

    elif soh >= 80:
        return "Good"

    elif soh >= 70:
        return "Aging"

    else:
        return "Poor / Service Recommended"


# --------------------------------------------------
# PREDICTION FUNCTION
# --------------------------------------------------

def predict_soh(input_df):

    processed = preprocessor.transform(
        input_df
    )

    prediction = model.predict(
        processed,
        verbose=0
    ).ravel()

    return prediction


# --------------------------------------------------
# TABS
# --------------------------------------------------

tab1, tab2, tab3 = st.tabs(
    [
        "Single Prediction",
        "Batch CSV Prediction",
        "Model Information"
    ]
)


# ==================================================
# TAB 1
# ==================================================

with tab1:

    st.subheader(
        "Enter Battery Measurements"
    )

    batch_values = metadata.get(
        "batch_values",
        ["BatchA"]
    )

    col1, col2 = st.columns(2)


    # -----------------------------
    # LEFT SIDE
    # -----------------------------

    with col1:

        batch_id = st.selectbox(
            "Batch ID",
            batch_values
        )

        cycle = st.number_input(
            "Cycle",
            min_value=0,
            value=500,
            step=1
        )

        voltage = st.number_input(
            "Voltage",
            value=3.70,
            format="%.4f"
        )

        current = st.number_input(
            "Current",
            value=2.00,
            format="%.4f"
        )

        temperature = st.number_input(
            "Temperature",
            value=30.0,
            format="%.3f"
        )

        charge_time = st.number_input(
            "Charge Time",
            value=120.0,
            format="%.3f"
        )


    # -----------------------------
    # RIGHT SIDE
    # -----------------------------

    with col2:

        discharge_time = st.number_input(
            "Discharge Time",
            value=110.0,
            format="%.3f"
        )

        resistance = st.number_input(
            "Internal Resistance",
            value=0.050,
            format="%.6f"
        )

        capacity = st.number_input(
            "Capacity",
            value=2.50,
            format="%.4f"
        )

        humidity = st.number_input(
            "Ambient Humidity",
            value=50.0,
            format="%.3f"
        )

        c_rate = st.number_input(
            "C Rate",
            value=1.0,
            format="%.4f"
        )


    # --------------------------------------------------
    # CREATE INPUT DATAFRAME
    # --------------------------------------------------

    input_data = pd.DataFrame(
        [
            {
                "BatchID": batch_id,
                "Cycle": cycle,
                "Voltage": voltage,
                "Current": current,
                "Temperature": temperature,
                "ChargeTime": charge_time,
                "DischargeTime": discharge_time,
                "InternalResistance": resistance,
                "Capacity": capacity,
                "AmbientHumidity": humidity,
                "C_Rate": c_rate
            }
        ]
    )


    # --------------------------------------------------
    # PREDICT BUTTON
    # --------------------------------------------------

    if st.button(
        "Predict Battery Health",
        type="primary"
    ):

        prediction = predict_soh(
            input_data
        )

        pred = float(
            prediction[0]
        )

        # Keep result between 0 and 100
        pred = float(
            np.clip(
                pred,
                0,
                100
            )
        )

        status = health_status(
            pred
        )

        st.divider()

        st.metric(
            "Predicted SOH",
            f"{pred:.2f}%"
        )

        st.progress(
            int(round(pred))
        )

        if pred >= 90:

            st.success(
                f"Battery Health Status: {status}"
            )

        elif pred >= 80:

            st.info(
                f"Battery Health Status: {status}"
            )

        elif pred >= 70:

            st.warning(
                f"Battery Health Status: {status}"
            )

        else:

            st.error(
                f"Battery Health Status: {status}"
            )

        st.write(
            "Input used for prediction:"
        )

        st.dataframe(
            input_data,
            use_container_width=True
        )


# ==================================================
# TAB 2
# ==================================================

with tab2:

    st.subheader(
        "Upload Battery CSV"
    )

    st.write(
        "Upload a CSV containing the same input "
        "columns used during model training."
    )

    st.write(
        "`BatteryID` may be included. "
        "It will not be used for ANN prediction."
    )


    expected_features = (
        metadata["numeric_features"]
        +
        metadata["categorical_features"]
    )


    st.caption(
        "Required model columns: "
        +
        ", ".join(expected_features)
    )


    uploaded = st.file_uploader(
        "Choose CSV file",
        type=["csv"]
    )


    if uploaded is not None:

        try:

            batch_df = pd.read_csv(
                uploaded
            )


            missing_cols = [
                c
                for c in expected_features
                if c not in batch_df.columns
            ]


            if missing_cols:

                st.error(
                    "Missing columns: "
                    +
                    ", ".join(missing_cols)
                )


            else:

                model_input = batch_df[
                    expected_features
                ].copy()


                predictions = predict_soh(
                    model_input
                )


                predictions = np.clip(
                    predictions,
                    0,
                    100
                )


                output_df = batch_df.copy()


                output_df[
                    "Predicted_SOH"
                ] = predictions


                output_df[
                    "Health_Status"
                ] = output_df[
                    "Predicted_SOH"
                ].apply(
                    health_status
                )


                st.success(
                    f"Successfully predicted "
                    f"{len(output_df)} battery records."
                )


                st.dataframe(
                    output_df,
                    use_container_width=True
                )


                csv_bytes = (
                    output_df
                    .to_csv(index=False)
                    .encode("utf-8")
                )


                st.download_button(
                    label="Download Predictions",
                    data=csv_bytes,
                    file_name=(
                        "battery_soh_predictions.csv"
                    ),
                    mime="text/csv"
                )


        except Exception as e:

            st.error(
                f"Error reading or predicting CSV: {e}"
            )


# ==================================================
# TAB 3
# ==================================================

with tab3:

    st.subheader(
        "ANN Project Details"
    )

    st.write(
        "**Project:** "
        "EV Battery Health Prediction"
    )

    st.write(
        "**Target:** "
        "State of Health (SOH %)"
    )

    st.write(
        "**Model:** "
        "Feed-forward Artificial Neural Network"
    )

    st.write(
        "**Problem Type:** Regression"
    )


    # --------------------------------------------------
    # MODEL METRICS
    # --------------------------------------------------

    metrics = metadata.get(
        "metrics",
        {}
    )


    if metrics:

        c1, c2, c3 = st.columns(3)


        mae = metrics.get(
            "mae",
            float("nan")
        )

        rmse = metrics.get(
            "rmse",
            float("nan")
        )

        r2 = metrics.get(
            "r2",
            float("nan")
        )


        c1.metric(
            "Test MAE",
            f"{mae:.4f}"
        )

        c2.metric(
            "Test RMSE",
            f"{rmse:.4f}"
        )

        c3.metric(
            "Test R²",
            f"{r2:.4f}"
        )


    # --------------------------------------------------
    # FEATURES
    # --------------------------------------------------

    st.write(
        "**Numerical Features:**"
    )

    st.write(
        ", ".join(
            metadata[
                "numeric_features"
            ]
        )
    )


    st.write(
        "**Categorical Features:**"
    )

    st.write(
        ", ".join(
            metadata[
                "categorical_features"
            ]
        )
    )
