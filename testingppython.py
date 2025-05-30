import streamlit as st
import numpy as np
import joblib
from huggingface_hub import hf_hub_download
from transformers import pipeline
from duckduckgo_search import DDGS

@st.cache_resource
def load_model():
    return joblib.load(hf_hub_download("AWeirdDev/human-disease-prediction", "sklearn_model.joblib"))


@st.cache_resource
def load_generator():
    return pipeline("text2text-generation", model="google/flan-t5-base")

model = load_model()
generator = load_generator()


symptom_list = ['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills', 
                'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 
                'burning_micturition', 'spotting_ urination', 'fatigue', 'weight_gain', 'anxiety', 
                'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness', 'lethargy', 
                'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes', 
                'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish_skin', 
                'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation', 
                'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes', 
                'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise', 
                'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 
                'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 
                'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 
                'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels', 
                'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties', 
                'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech', 
                'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints', 
                'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness', 
                'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine', 
                'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)', 
                'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 
                'belly_pain', 'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes', 
                'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum', 
                'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion', 
                'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen', 
                'history_of_alcohol_consumption', 'fluid_overload.1', 'blood_in_sputum', 
                'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 
                'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 
                'blister', 'red_sore_around_nose', 'yellow_crust_ooze']

@st.cache_data(show_spinner=False)
def get_definition(disease):
    return generator(f"{disease} is ", max_length=50, num_return_sequences=1)[0]['generated_text']

@st.cache_data(show_spinner=False)
def get_treatment(disease):
    return generator(f"In studies, {disease} is treated by ", do_sample=True, num_return_sequences=1)[0]['generated_text']

@st.cache_data(show_spinner=False)
def get_urgency(disease):
    return generator(f"The urgency of {disease} must be treated is by ", do_sample=True, num_return_sequences=1)[0]['generated_text']

# Search for hospitals using DuckDuckGo
def search_hospitals(predicted_disease, user_location, max_results=5):
    query = f"{predicted_disease} hospital near {user_location} book appointment"
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append({
                'title': r['title'],
                'link': r['href'],
                'snippet': r['body']
            })
    return results

st.title("🧠 Early Disease Prediction")

age = st.slider("Select your age", 0, 100, 25)
gender = st.radio("Select your gender", ("Male", "Female"))
gender_val = 1 if gender == "Male" else 0

st.subheader("Select Symptoms:")
symptom_inputs = []
for symptom in symptom_list:
    checked = st.checkbox(symptom.replace("_", " ").capitalize())
    symptom_inputs.append(1 if checked else 0)

location = st.text_input("Enter your city or location for nearby hospital search")

if st.button("🩺 Detect Disease"):
    input_data = np.array([symptom_inputs])
    prediction = model.predict(input_data)
    predicted_disease = prediction[0]

    with st.spinner("🔍 Fetching..."):
        definition = get_definition(predicted_disease)
        treatment = get_treatment(predicted_disease)
        urgency = get_urgency(predicted_disease)
        hospitals = search_hospitals(predicted_disease, location) if location else []

    st.success(f" **Predicted Disease:** {predicted_disease}")
    st.markdown(f"**About {predicted_disease}:**\n\n{definition}\n\n**Treatment:**\n\n{treatment}\n\n**Urgency:**\n\n{urgency}")

    if hospitals:
        st.subheader("🏥 Nearby Hospitals with Appointment Links:")
        for hospital in hospitals:
            st.markdown(f"🔹 **{hospital['title']}**\n\n📄 {hospital['snippet']}\n\n🔗 [Book/Visit]({hospital['link']})\n")
