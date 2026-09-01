import joblib
import pandas as pd
import streamlit as st

st.title("Medical Device Failure Prediction")

model = joblib.load("medical_device_failure_prediction.pkl")

#TypeDescription -> valid Manufacturer options (from the training data)
type_to_manufacturers = {
    "Defibrillator": ["Koninklijke Philips N.V.", "Physio-Control Inc"],
    "Infusion Pump": ["B. Braun Melsungen AG", "Baxter Healthcare Corp", "Medtronic"],
    "Physiologic Monitoring System": ["GE Healthcare", "Koninklijke Philips N.V.", "Nihon Kohden Corp"],
    "Radiographic System": ["Canon Inc (Toshiba Medical)", "Koninklijke Philips N.V.", "Siemens Healthineers"],
    "Sphygmomanometers": ["Med-Vantage Sdn Bhd", "Omron Healthcare Co Ltd", "Welch Allyn Inc"],
    "Ventilator": ["Blease Medical Equipment Ltd", "Datex-Ohmeda Inc", "Drager Medical AG & Co KGaA"],
}
condition_options = ["Active / in use", "Unrepairable failure but still in use", "Approved for disposal"]
operations_options = ["12 hours / 6 days a week", "24 hours / 7 days a week"]

#TypeDescription
type_description = st.selectbox("Device type", list(type_to_manufacturers.keys()))

#Age
age_input = st.text_input("Device age in years (0-30)")

#AssetCondition
condition_choice = st.selectbox("Asset condition", condition_options)
asset_condition = condition_options.index(condition_choice)  # options are coded 0, 1, 2

#Manufacturer (depends on the TypeDescription chosen above)
manufacturer_options = type_to_manufacturers[type_description]
manufacturer = st.selectbox(f"Manufacturer (for {type_description})", manufacturer_options)

#Operations
operations_choice = st.selectbox("Operations schedule", operations_options)
operations = operations_options.index(operations_choice) + 1  # options are coded 1, 2

if st.button("Predict"):
    # Validate Age
    if not age_input.isdigit() or not (0 <= int(age_input) <= 30):
        st.error("Age is not appropriate. Please enter a whole number between 0 and 30.")
    else:
        age = int(age_input)

        X_new = pd.DataFrame([{
            "TypeDescription": type_description,
            "Age": age,
            "AssetCondition": asset_condition,
            "Manufacturer": manufacturer,
            "Operations": operations,
        }])

        prediction = model.predict(X_new)[0]

        st.success(f"Estimated time to failure: {prediction:.1f} months (~{prediction/12:.1f} years)")