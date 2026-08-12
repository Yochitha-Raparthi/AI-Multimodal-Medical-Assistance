import gradio as gr
import os
import uuid
import base64
import speech_recognition as sr
from gtts import gTTS
from groq import Groq
from core.ml_model import predict_disease
from dotenv import load_dotenv
load_dotenv()
# ==============================
# GROQ CLIENT
# ==============================
print(os.getenv("GROQ_API_KEY"))
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


predicted_disease_global = {"disease": None}

# ==============================
# IMAGE FUNCTIONS
# ==============================

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def analyze_image_with_query(query, image_path):

    if image_path is None:
        return "⚠ Please upload an image."

    encoded_image = encode_image(image_path)

    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": query},
            {"type": "image_url", 
             "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
        ]}
    ]

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages,
            max_tokens=400
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Image Analysis Error: {str(e)}"

# ==============================
# SPEECH TO TEXT
# ==============================

def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()

    try:
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)

        return recognizer.recognize_google(audio)

    except:
        return "Could not understand audio."

# ==============================
# TEXT TO SPEECH
# ==============================

def text_to_speech(text):
    os.makedirs("outputs", exist_ok=True)
    filename = f"response_{uuid.uuid4().hex}.mp3"
    filepath = os.path.join("outputs", filename)
    gTTS(text).save(filepath)
    return filepath

# ==============================
# ML DIAGNOSIS
# ==============================

def diagnose(symptoms_text):
    disease, confidence, risk = predict_disease(symptoms_text)

    if disease == "Symptoms not recognized":
        return "⚠ Invalid symptoms."

    predicted_disease_global["disease"] = disease

    return f"""
🩺 Predicted Disease: {disease}
📊 Confidence: {confidence}%
⚠ Risk Level: {risk}
"""

# ==============================
# LLM TEXT + VOICE
# ==============================

def ask_llm(query_text, audio_file):

    if audio_file:
        query_text = transcribe_audio(audio_file)

    if not query_text:
        return "⚠ Enter or speak question.", None

    disease = predicted_disease_global["disease"]

    if not disease:
        return "⚠ Predict disease first.", None

    prompt = f"""
# Patient disease: {disease}

# Question: {query_text}
# Answer clearly and safely.
# """
    prompt = f"""
You are AROGYA AI, a professional and responsible medical assistant.

The patient has been predicted with: {disease}.

Follow these strict rules:
1. Do NOT provide dangerous medical prescriptions.
2. Always suggest consulting a doctor.
3. Use simple language.
4. Structure the response clearly.

Provide answer in this format:

🩺 Condition Overview:
Explain briefly.

📋 Possible Causes:
- Bullet points

💊 General Care Suggestions:
- Safe recommendations only

⚠ When to See a Doctor:
Clear emergency signs.

Now answer the patient question:

Question: {query_text}
If the question relates to the uploaded image,
correlate findings carefully.

Do not overdiagnose.
Be medically cautious.
"""

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400
        )

        answer = response.choices[0].message.content

    except Exception as e:
        return f"LLM Error: {str(e)}", None

    audio_output = text_to_speech(answer)

    return answer, audio_output
custom_css = """
body {
    background: linear-gradient(to right, #43cea2, #185a9d);
}

.gradio-container {
    background: white;
    border-radius: 25px;
    padding: 25px;
}

button {
    background-color: #185a9d !important;
    color: white !important;
    border-radius: 10px !important;
}

textarea {
    border-radius: 12px !important;
}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    # gr.Image("assets/medical_banner.png",
    #      show_label=False,
    #      interactive=False)
    gr.Markdown("""
# 🏥 AROGYA AI
### Intelligent Multimodal Medical Assistant
""")
    
    # gr.Image("assets/heartbeat.gif",
    #      show_label=False,
    #      interactive=False)

    # TAB 1
    # with gr.Tab("🩺 Disease Prediction"):
    #     symptoms_input = gr.Textbox(label="Enter Symptoms")
    #     predict_btn = gr.Button("Predict")
    #     ml_output = gr.Textbox()
    with gr.Tab("🩺 Disease Prediction"):

     with gr.Row():
        with gr.Column(scale=2):
            symptoms_input = gr.Textbox(label="Enter Symptoms")
            predict_btn = gr.Button("Predict Disease")

        # with gr.Column(scale=1):
        #     gr.Image("assets/heartbeat.gif",
        #              show_label=False,
        #              interactive=False)

            ml_output = gr.Markdown()
            predict_btn.click(diagnose, inputs=symptoms_input, outputs=ml_output)

    # TAB 2
    with gr.Tab("💬 AI Medical Consultation"):
        query_input = gr.Textbox(label="Type Question")
        voice_input = gr.Audio(sources=["microphone"], type="filepath")
        ask_btn = gr.Button("Ask AI")
        # llm_text = gr.Textbox(label="AI Response")
        llm_text = gr.Markdown(label="AI Medical Response")
        llm_audio = gr.Audio(label="Voice Response")
        ask_btn.click(ask_llm, inputs=[query_input, voice_input],
                      outputs=[llm_text, llm_audio])

    # TAB 3
    # with gr.Tab("🖼 Medical Image Analysis"):
    #     image_input = gr.Image(type="filepath", label="Upload Medical Image")
    #     image_query = gr.Textbox(label="Ask About This Image")
    #     image_btn = gr.Button("Analyze Image")
    #     image_output = gr.Textbox(label="Analysis Result")
    #     image_btn.click(analyze_image_with_query,
    #                     inputs=[image_query, image_input],
    #                     outputs=image_output)
    with gr.Tab("🖼 Medical Image Analysis"):

     gr.Markdown("### 📸 Upload Medical Image for AI Analysis")

     image_input = gr.Image(
        type="filepath",
        label="Upload Medical Image",
        height=300
     )

     image_query = gr.Textbox(
        label="Ask Question About This Image"
     )

     image_btn = gr.Button("🔍 Analyze Image")

     image_output = gr.Markdown(
        label="🧠 AI Analysis Result"
    )

    image_btn.click(
        analyze_image_with_query,
        inputs=[image_query, image_input],
        outputs=image_output
    )

demo.launch()

