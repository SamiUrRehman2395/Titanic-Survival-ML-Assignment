
import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢"
)

@st.cache_resource
def load_model():
    return joblib.load("titanic_svc_model.pkl")

model = load_model()

st.title("🚢 Titanic Passenger Survival Prediction")
st.write("Enter passenger details to get a model prediction.")

pclass = st.selectbox("Passenger Class", ["First", "Second", "Third"])
gender = st.selectbox("Gender", ["Female", "Male"])
sibling = st.selectbox("Siblings/Spouses Aboard", ["One", "Three", "Two", "Zero"])
embarked = st.selectbox(
    "Port of Embarkation",
    ["Cherbourg", "Queenstown", "Southampton"]
)

if st.button("Predict Survival"):
    features = pd.DataFrame([{
        "PClass": {"First": 0, "Second": 1, "Third": 2}[pclass],
        "Gender": {"Female": 0, "Male": 1}[gender],
        "Sibling": {"One": 0, "Three": 1, "Two": 2, "Zero": 3}[sibling],
        "Embarked": {
            "Cherbourg": 0,
            "Queenstown": 1,
            "Southampton": 2
        }[embarked]
    }])

    prediction = int(model.predict(features)[0])

    if prediction == 1:
        st.success("Prediction: Survived")
    else:
        st.error("Prediction: Did not survive")

    st.caption("This is a model prediction, not a certainty.")
