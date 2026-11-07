<div align="center">

# 💊 MediGuard

### AI-Powered Medicine Expiry Tracking & Safe Disposal Assistant

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Gemini](https://img.shields.io/badge/Gemini-API-4285F4?style=for-the-badge&logo=googlegemini&logoColor=white)](https://aistudio.google.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)

<!-- After you deploy on Streamlit Community Cloud, add this line with your real link:
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-success?style=for-the-badge&logo=streamlit)](YOUR-APP-URL)
-->



</div>

> **Not medical advice.** MediGuard never says whether an expired medicine is safe to use, and it does not give doses or diagnoses.

---

## 📌 Overview

Expired medicines often get thrown in the bin or washed down the sink, which harms people and the environment. Most homes and small clinics also have no simple way to know what is about to expire.

**MediGuard** fixes both problems. Take a photo of a medicine strip, box or bottle and the app reads the name, batch number and expiry date. It then warns you before the medicine expires and tells you how to dispose of it safely, based on the type of medicine.

---

## 🌟 Key Features

### 1. 📷 Photo Scan with Gemini Vision
- Upload photos or use the camera. The Gemini API returns the **name, active ingredient, form, category, batch number, manufacture date, expiry date** and a confidence score.
- If the model is busy (503 or 429 errors), the app retries and then falls back to other Gemini models.

### 2. ✅ Validation Checks
A medicine is flagged for manual review when:
- the expiry date cannot be read,
- the year looks wrong (before 2015 or more than 10 years ahead),
- the manufacture date is after the expiry date, or
- the model's confidence is below 0.6.

### 3. 🧠 Disposal Classifier (scikit-learn)
- TF-IDF on character n-grams plus logistic regression sorts each medicine into **8 disposal classes**: antibiotic, controlled, sharps, liquid, tablet/capsule, topical, inhaler and other.
- **Safety-first ensemble:** if the classifier and Gemini disagree, the app keeps the higher-risk class and adds a "verify" flag.

### 4. 🗂️ Inventory, Alerts and Reminders
- Every medicine gets a status: **Expired, Expiring ≤30d, Expiring ≤90d, OK or Unknown**.
- Filter the inventory, edit mistakes in a table, export to **CSV**, or export to a **calendar file (.ics)** with a reminder on the expiry date and another 30 days before.

### 5. ♻️ Disposal Guide
- Step-by-step disposal advice for every expired medicine, with the highest-risk classes shown first.

### 6. 💬 Inventory-Aware Assistant
- Ask questions such as "What expires in the next 30 days?" The answers come from **your own inventory**. The assistant will not give doses or say an expired medicine is safe.

---

## 🔄 How It Works

```
Photo → Gemini vision extraction → Rule-based checks → TF-IDF + Logistic Regression
      → Safety-first ensemble → Disposal advice → Alerts, charts and calendar reminders
```

---

## 🖼️ Screenshots

| Dashboard | Scan | Disposal |
|:---:|:---:|:---:|
| <img width="1920" height="1080" alt="Screenshot (171)" src="https://github.com/user-attachments/assets/7ae3c232-7dd6-4a50-adda-543360f38fbb" />| <img width="1920" height="1080" alt="Screenshot (172)" src="https://github.com/user-attachments/assets/0062e92e-d9ee-46de-a8f5-66baafe60b51" />| <img width="1920" height="1080" alt="Screenshot (174)" src="https://github.com/user-attachments/assets/632f85e6-2a1a-4ef2-8dcc-73c99a92c70d" />
 

---

## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **App framework:** [Streamlit](https://streamlit.io/)
- **AI:** [Google Gemini API](https://ai.google.dev/) (`google-genai`) for image reading and the chat assistant
- **Machine learning:** [scikit-learn](https://scikit-learn.org/) (TF-IDF, logistic regression, stratified cross-validation)
- **Data and charts:** Pandas, NumPy, [Plotly](https://plotly.com/python/)
- **Storage:** local `inventory.csv`

---

## 📁 Project Structure

```
MediGuard/
├── app.py            # UI, Gemini calls, validation, classifier, disposal logic
├── inventory.csv     # Saved inventory (created by the app)
├── banner.png        # Header image
├── requirements.txt  # Python dependencies
├── .gitignore
└── docs/             # Screenshots used in this README
```

---

## 💻 Local Development

You need Python 3.10 or newer and a free API key from [Google AI Studio](https://aistudio.google.com/).

### 1. Clone the repository
```bash
git clone https://github.com/Roastyyy31/MediGuard.git
cd MediGuard
```

### 2. Create a virtual environment and install dependencies
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Add your Gemini API key
Either paste it into the **Gemini API key** box in the app's sidebar, or set it before starting:
```bash
export GEMINI_API_KEY="your-key"     # Windows: set GEMINI_API_KEY=your-key
```

### 4. Run the app
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501). No photos handy? Click **Load demo data** in the sidebar to fill the app with 8 sample medicines.

---

## ⚠️ Limitations

- The classifier trains on only **64 hand-labelled examples**, so its accuracy figure is a rough guide, not a benchmark.
- Photo reading depends on the Gemini model. Blurry or curved packaging can give wrong dates, which is why the app flags doubtful results and lets you edit them.
- The training names and disposal advice lean towards medicines sold in India. Check your local pharmacy or hospital take-back rules before disposing of anything.

## 🔭 Future Work

- Grow the training set and report a proper evaluation.
- Barcode scanning for faster entry.
- Directory of nearby pharmacy take-back and recycling points.

---

## 👤 Author

**Rugved Badgujar**, final-year B.E. Computer Science (AI & ML), Mumbai University
[GitHub](https://github.com/Roastyyy31) · [LinkedIn](www.linkedin.com/in/rugved-badgujar-b8a4a925a)
