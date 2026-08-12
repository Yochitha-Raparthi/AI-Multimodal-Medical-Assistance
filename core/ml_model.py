

import pickle
import numpy as np

# -------------------------------
# Load Saved Files
# -------------------------------
model = pickle.load(open("core/model.pkl", "rb"))
label_encoder = pickle.load(open("core/label_encoder.pkl", "rb"))
symptom_list = pickle.load(open("core/symptom_list.pkl", "rb"))

# -------------------------------
# Prediction Function
# -------------------------------
def predict_disease(user_symptoms):
    """
    user_symptoms: list of symptom strings
    Example: ["itching", "skin_rash"]
    """

    # Normalize input
    if isinstance(user_symptoms, str):
        user_symptoms = user_symptoms.split(",")

    user_symptoms = [
        str(sym).strip().lower().replace(" ", "_")
        for sym in user_symptoms
        if str(sym).strip() != ""
    ]

    # Create zero vector
    input_vector = np.zeros(len(symptom_list))

    recognized = []

    for symptom in user_symptoms:
        if symptom in symptom_list:
            index = symptom_list.index(symptom)
            input_vector[index] = 1
            recognized.append(symptom)

    # If no symptom matched
    if np.sum(input_vector) == 0:
        return "Symptoms not recognized", 0, "Unknown"

    # Reshape for prediction
    input_vector = input_vector.reshape(1, -1)

    # Predict
    prediction_encoded = model.predict(input_vector)[0]
    probabilities = model.predict_proba(input_vector)[0]

    confidence = np.max(probabilities) * 100
    predicted_disease = label_encoder.inverse_transform([prediction_encoded])[0]

    # Risk logic
    if confidence > 80:
        risk = "High"
    elif confidence > 50:
        risk = "Medium"
    else:
        risk = "Low"

    return predicted_disease, round(confidence, 2), risk

