# 🏥 AROGYA-AI — AI Medical Chatbot

### Intelligent Multimodal Medical Assistant using Machine Learning, Vision & Voice

AROGYA-AI is an AI-powered medical assistant that combines **Machine Learning, Large Language Models, computer vision, speech recognition, and text-to-speech** to provide an interactive healthcare assistance system.

The application allows users to:

* 🩺 Predict possible diseases from symptoms
* 🖼️ Analyze uploaded medical images using a vision-capable AI model
* 💬 Ask medical questions using text
* 🎤 Ask questions using voice input
* 🔊 Receive AI responses as audio
* 📊 View disease confidence and risk level
* 🖥️ Interact through a simple Gradio web interface

> ⚠️ **Medical Disclaimer:** AROGYA-AI is an educational and research project. It is not a replacement for a qualified medical professional. AI-generated results should not be considered a final diagnosis or medical prescription.

---

## ✨ Features

### 🩺 1. Disease Prediction

The system accepts symptoms entered by the user and uses a trained machine learning model to predict a possible disease.

The prediction provides:

* Predicted disease
* Confidence percentage
* Risk level

Example:

```text
🩺 Predicted Disease: Fungal Infection
📊 Confidence: 92.45%
⚠ Risk Level: High
```

The disease prediction model is implemented using a saved ML model and label encoder.

---

### 🖼️ 2. Medical Image Analysis

Users can upload a medical image and ask a question about it.

The application sends the image and question to a vision-capable AI model for analysis.

Example:

```text
Patient Question:
What could this skin condition be?
```

The system generates a structured response containing:

* 🩺 Observation
* 📋 Possible Findings
* 💊 General Care Suggestions
* ⚠️ Medical Disclaimer

---

### 💬 3. AI Medical Consultation

Users can ask medical questions through text.

The chatbot uses an LLM to generate a response based on the predicted disease and the patient's question.

The response is structured into:

```text
🩺 Condition Overview

📋 Possible Causes

💊 General Care Suggestions

⚠ When to See a Doctor
```

The chatbot is designed to provide general information and avoid dangerous medical prescriptions.

---

### 🎤 4. Voice Input

Users can record their question through the microphone.

The system converts speech into text using speech recognition and sends the converted text to the medical chatbot.

---

### 🔊 5. Voice Response

The AI-generated response is converted into an MP3 audio file using **gTTS (Google Text-to-Speech)**.

The generated audio is then displayed in the Gradio interface.

---

### 🖥️ 6. Gradio Interface

The application provides a web-based interface with three main sections:

```text
┌──────────────────────────────────────┐
│            🏥 AROGYA AI              │
│ Intelligent Multimodal Assistant     │
├──────────────────────────────────────┤
│ 🩺 Disease Prediction                │
├──────────────────────────────────────┤
│ 💬 AI Medical Consultation           │
├──────────────────────────────────────┤
│ 🖼️ Medical Image Analysis            │
└──────────────────────────────────────┘
```

---

# 🏗️ System Architecture

```text
                         ┌─────────────────┐
                         │      User       │
                         └────────┬────────┘
                                  │
                   ┌──────────────┼──────────────┐
                   │              │              │
                   ▼              ▼              ▼
              Symptoms         Voice          Image
                   │              │              │
                   ▼              ▼              ▼
              ML Model      Speech-to-Text   Vision AI
                   │              │              │
                   ▼              └──────┬───────┘
          Disease Prediction             │
                   │                     │
                   └──────────┬──────────┘
                              ▼
                       Medical LLM
                              │
                              ▼
                     AI Medical Response
                              │
                    ┌─────────┴─────────┐
                    │                   │
                    ▼                   ▼
                 Text Response      Text-to-Speech
                                        │
                                        ▼
                                  Audio Response
```

---

# 🛠️ Technologies Used

| Technology        | Purpose                         |
| ----------------- | ------------------------------- |
| Python            | Main programming language       |
| Gradio            | Web interface                   |
| NumPy             | Numerical processing            |
| Scikit-learn      | Machine learning                |
| Pickle            | Loading trained ML models       |
| Groq              | LLM-based medical consultation  |
| Gemini            | Medical image analysis          |
| SpeechRecognition | Speech-to-text                  |
| gTTS              | Text-to-speech                  |
| Pillow            | Image processing                |
| python-dotenv     | Environment variable management |

---

# 📁 Project Structure

```text
AROGYA-AI/
│
├── app.py
├── api_client.py
├── .env
├── requirements.txt
│
├── core/
│   ├── ml_model.py
│   ├── model.pkl
│   ├── label_encoder.pkl
│   └── symptom_list.pkl
│
├── outputs/
│   └── generated audio files
│
├── assets/
│   └── project assets
│
└── README.md
```

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone <your-github-repository-url>
```

Navigate into the project:

```bash
cd AROGYA-AI
```

---

## 2. Create a Virtual Environment

On Windows:

```bash
python -m venv ml_env
```

Activate it:

```bash
ml_env\Scripts\activate
```

On macOS/Linux:

```bash
source ml_env/bin/activate
```

---

## 3. Install Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, install the main dependencies:

```bash
pip install gradio
pip install numpy
pip install scikit-learn
pip install groq
pip install google-generativeai
pip install Pillow
pip install SpeechRecognition
pip install gTTS
pip install python-dotenv
pip install pydub
```

---

# 🔑 API Configuration

Create a file named:

```text
.env
```

in the project root directory.

Add your API keys:

```env
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_gemini_api_key
```

### Important

Never upload your `.env` file to GitHub.

Add this to `.gitignore`:

```text
.env
ml_env/
__pycache__/
outputs/
*.pyc
```

---

# 🤖 AI Models

## Machine Learning Model

The disease prediction component uses a trained machine learning model stored in:

```text
core/model.pkl
```

The corresponding label encoder is:

```text
core/label_encoder.pkl
```

The available symptom list is:

```text
core/symptom_list.pkl
```

---

## Groq LLM

The medical consultation component uses a supported Groq model.

Example:

```python
model="llama-3.3-70b-versatile"
```

The model receives the patient's question and predicted disease and generates a medically cautious response.

---

## Gemini Vision Model

The medical image analysis component uses a vision-capable Gemini model.

The image is loaded using Pillow and passed to the vision model together with the user's question.

---

# ▶️ Running the Application

Activate your virtual environment first:

```bash
ml_env\Scripts\activate
```

Then run:

```bash
python app.py
```

You should see something similar to:

```text
Running on local URL: http://127.0.0.1:7860
```

Open the displayed URL in your browser.

---

# 🧪 How to Use

## Step 1 — Disease Prediction

Open:

```text
🩺 Disease Prediction
```

Enter symptoms such as:

```text
itching, skin rash, redness
```

Click:

```text
Predict Disease
```

The application displays the predicted disease, confidence and risk level.

---

## Step 2 — Medical Consultation

Open:

```text
💬 AI Medical Consultation
```

Enter a question such as:

```text
What precautions should I take?
```

or record your question using the microphone.

Click:

```text
Ask AI
```

The chatbot generates a response and an audio response.

---

## Step 3 — Medical Image Analysis

Open:

```text
🖼️ Medical Image Analysis
```

Upload a suitable medical image.

Enter a question such as:

```text
What can you observe in this image?
```

Click:

```text
🔍 Analyze Image
```

The vision model analyzes the image and provides an AI-generated response.

---

# 🧠 Machine Learning Pipeline

The disease prediction process works as follows:

```text
User Symptoms
      │
      ▼
Normalize Symptoms
      │
      ▼
Match with Symptom List
      │
      ▼
Create Feature Vector
      │
      ▼
Trained ML Model
      │
      ▼
Disease Prediction
      │
      ▼
Calculate Confidence
      │
      ▼
Determine Risk Level
```

The input symptoms are converted into a binary feature vector.

For example:

```text
itching       → 1
skin_rash     → 1
headache      → 0
cough         → 0
fever         → 0
```

---

# 🔐 Security

API keys should never be directly written inside Python source code.

Use environment variables:

```python
import os

api_key = os.getenv("GROQ_API_KEY")
```

and:

```python
api_key = os.getenv("GOOGLE_API_KEY")
```

Never commit:

```text
.env
```

to GitHub.

If an API key is accidentally exposed, immediately revoke it and generate a new one.

---

# ⚠️ Known Limitations

* AI predictions are not guaranteed to be medically accurate.
* Disease prediction depends on the quality of the trained dataset.
* Image analysis may produce incorrect or incomplete observations.
* Speech recognition may fail because of background noise or unclear speech.
* API services require an internet connection.
* Free API tiers may have usage and rate limits.
* The application should not be used for emergency medical decisions.

---

# 🔮 Future Enhancements

Possible future improvements include:

* 📱 Mobile application
* 🗃️ Patient history storage
* 🔐 User authentication
* 📊 Medical history dashboard
* 🌐 Multi-language support
* 🎤 Improved speech recognition
* 🧠 Improved disease prediction model
* 🖼️ More advanced medical image analysis
* 📈 Patient health trend visualization
* 👨‍⚕️ Doctor consultation integration
* 🏥 Hospital/clinic integration
* ☁️ Cloud deployment

---

# 👩‍💻 Project Purpose

AROGYA-AI was developed as an educational AI/ML project to demonstrate the integration of:

```text
Machine Learning
       +
Large Language Models
       +
Computer Vision
       +
Speech Recognition
       +
Text-to-Speech
       +
Web Interface
```

The project demonstrates how multiple AI technologies can be combined into a single multimodal healthcare assistance application.

---

# 📜 License

This project is intended for educational and research purposes.

If you use or modify this project, follow the license requirements of the underlying models, libraries, datasets, and third-party APIs.

---

# ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.

---

## ⚠️ Medical Disclaimer

**AROGYA-AI is not a medical professional and does not provide a definitive medical diagnosis or prescription.**

The information generated by this application is intended only for educational and informational purposes.

Always consult a qualified healthcare professional for diagnosis, treatment, medication, or medical emergencies.
