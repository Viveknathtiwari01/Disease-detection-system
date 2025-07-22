import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
import streamlit as st # type: ignore
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from wordcloud import WordCloud # type: ignore
import matplotlib.pyplot as plt
import plotly.express as px # type: ignore

st.set_page_config(
    page_title="Disease Diagnosis System", 
    page_icon = "🩺" )

# --- Disease tips dictionary (expand as needed) ---
disease_tips = {
    "(vertigo) Paroymsal  Positional Vertigo": [
        "Consult a specialist for accurate diagnosis and management.",
        "Follow lifestyle modifications to manage symptoms effectively.",
        "Schedule regular follow-ups to monitor your condition."
    ],
    "AIDS": [
        "Consult a specialist for accurate diagnosis and management.",
        "Follow lifestyle modifications to manage symptoms effectively.",
        "Schedule regular follow-ups to monitor your condition."
    ],
    "Acne": [
        "Consult a specialist for accurate diagnosis and management.",
        "Follow lifestyle modifications to manage symptoms effectively.",
        "Schedule regular follow-ups to monitor your condition."
    ],
    "Alcoholic hepatitis": [
        "Consult a specialist for accurate diagnosis and management.",
        "Follow lifestyle modifications to manage symptoms effectively.",
        "Schedule regular follow-ups to monitor your condition."
    ],
    "Allergy": [
        "Consult a specialist for accurate diagnosis and management.",
        "Follow lifestyle modifications to manage symptoms effectively.",
        "Schedule regular follow-ups to monitor your condition."
    ],
    "Arthritis": [
        "Consult a specialist for accurate diagnosis and management.",
        "Follow lifestyle modifications to manage symptoms effectively.",
        "Schedule regular follow-ups to monitor your condition."
    ],
    "Bronchial Asthma": [
        "Consult a specialist for accurate diagnosis and management.",
        "Follow lifestyle modifications to manage symptoms effectively.",
        "Schedule regular follow-ups to monitor your condition."
    ],
    "Cervical spondylosis": [
        "Consult a specialist for accurate diagnosis and management.",
        "Follow lifestyle modifications to manage symptoms effectively.",
        "Schedule regular follow-ups to monitor your condition."
    ],
    "Chicken pox": [
        "Consult a specialist for accurate diagnosis and management.",
        "Follow lifestyle modifications to manage symptoms effectively.",
        "Schedule regular follow-ups to monitor your condition."
    ],
    "Common Cold": [
        "Consult a specialist for accurate diagnosis and management.",
        "Follow lifestyle modifications to manage symptoms effectively.",
        "Schedule regular follow-ups to monitor your condition."
    ],
    "Dengue": [
        "Consult a specialist for accurate diagnosis and management.",
        "Follow lifestyle modifications to manage symptoms effectively.",
        "Schedule regular follow-ups to monitor your condition."
    ],
    "Diabetes": [
        "Consult a specialist for accurate diagnosis and management.",
        "Follow lifestyle modifications to manage symptoms effectively.",
        "Schedule regular follow-ups to monitor your condition."
    ],
    "GERD": [
        "Consult a specialist for accurate diagnosis and management.",
        "Follow lifestyle modifications to manage symptoms effectively.",
        "Schedule regular follow-ups to monitor your condition."
    ],
    "Heart attack": [
        "Consult a specialist for accurate diagnosis and management.",
        "Follow lifestyle modifications to manage symptoms effectively.",
        "Schedule regular follow-ups to monitor your condition."
    ],
    "Hepatitis B": [
        "Consult a specialist for accurate diagnosis and management.",
        "Follow lifestyle modifications to manage symptoms effectively.",
        "Schedule regular follow-ups to monitor your condition."
    ],
    "Hepatitis C": [
        "Consult a specialist for accurate diagnosis and management.",
        "Follow lifestyle modifications to manage symptoms effectively.",
        "Schedule regular follow-ups to monitor your condition."
    ],
    "hepatitis A": [
        "Consult a specialist for accurate diagnosis and management.",
        "Follow lifestyle modifications to manage symptoms effectively.",
        "Schedule regular follow-ups to monitor your condition."
    ],
    "Hypertension": [
        "Consult a specialist for accurate diagnosis and management.",
        "Follow lifestyle modifications to manage symptoms effectively.",
        "Schedule regular follow-ups to monitor your condition."
    ],
    "Hyperthyroidism": [
        "Consult a specialist for accurate diagnosis and management.",
        "Follow lifestyle modifications to manage symptoms effectively.",
        "Schedule regular follow-ups to monitor your condition."
    ],
    "Hypothyroidism": [
        "Consult a specialist for accurate diagnosis and management.",
        "Follow lifestyle modifications to manage symptoms effectively.",
        "Schedule regular follow-ups to monitor your condition."
    ],
    "Impetigo": [
        "Consult a specialist for accurate diagnosis and management.",
        "Follow lifestyle modifications to manage symptoms effectively.",
        "Schedule regular follow-ups to monitor your condition."
    ],
    "Jaundice": [
        "Consult a specialist for accurate diagnosis and management.",
        "Follow lifestyle modifications to manage symptoms effectively.",
        "Schedule regular follow-ups to monitor your condition."
    ],
    "Malaria": [
        "Consult a specialist for accurate diagnosis and management.",
        "Follow lifestyle modifications to manage symptoms effectively.",
        "Schedule regular follow-ups to monitor your condition."
    ],
    "Migraine": [
        "Consult a specialist for accurate diagnosis and management.",
        "Follow lifestyle modifications to manage symptoms effectively.",
        "Schedule regular follow-ups to monitor your condition."
    ],
    "Osteoarthristis": [
        "Consult a specialist for accurate diagnosis and management.",
        "Follow lifestyle modifications to manage symptoms effectively.",
        "Schedule regular follow-ups to monitor your condition."
    ],
    "Paralysis (brain hemorrhage)": [
        "Consult a specialist for accurate diagnosis and management.",
        "Follow lifestyle modifications to manage symptoms effectively.",
        "Schedule regular follow-ups to monitor your condition."
    ],
    "Peptic ulcer diseae": [
        "Consult a specialist for accurate diagnosis and management.",
        "Follow lifestyle modifications to manage symptoms effectively.",
        "Schedule regular follow-ups to monitor your condition."
    ],
    "Pneumonia": [
        "Consult a specialist for accurate diagnosis and management.",
        "Follow lifestyle modifications to manage symptoms effectively.",
        "Schedule regular follow-ups to monitor your condition."
    ],
    "Psoriasis": [
        "Consult a specialist for accurate diagnosis and management.",
        "Follow lifestyle modifications to manage symptoms effectively.",
        "Schedule regular follow-ups to monitor your condition."
    ],
    "Tuberculosis": [
        "Consult a specialist for accurate diagnosis and management.",
        "Follow lifestyle modifications to manage symptoms effectively.",
        "Schedule regular follow-ups to monitor your condition."
    ],
    "Typhoid": [
        "Consult a specialist for accurate diagnosis and management.",
        "Follow lifestyle modifications to manage symptoms effectively.",
        "Schedule regular follow-ups to monitor your condition."
    ],
    "Urinary tract infection": [
        "Consult a specialist for accurate diagnosis and management.",
        "Follow lifestyle modifications to manage symptoms effectively.",
        "Schedule regular follow-ups to monitor your condition."
    ],
    "Varicose veins": [
        "Consult a specialist for accurate diagnosis and management.",
        "Follow lifestyle modifications to manage symptoms effectively.",
        "Schedule regular follow-ups to monitor your condition."
    ]
}


# --- Localization dictionary (expand as needed) ---
translations = {
    "en": {
        "Select your symptoms": "Select your symptoms",
        "Predict": "Predict",
        "Tips": "Tips",
        "Confidence Chart": "Confidence Chart",
        "Top Disease Predictions": "Top Disease Predictions",
        "Word Cloud": "Word Cloud",
        "Language": "Language",
        "Stay healthy! For persistent or severe symptoms, consult a healthcare professional.": "Stay healthy! For persistent or severe symptoms, consult a healthcare professional.",
        "Please select at least one symptom.": "Please select at least one symptom."
    },
    "hi": {
        "Select your symptoms": "अपने लक्षण चुनें",
        "Predict": "पूर्वानुमान",
        "Tips": "सलाह",
        "Confidence Chart": "विश्वास चार्ट",
        "Top Disease Predictions": "शीर्ष रोग पूर्वानुमान",
        "Word Cloud": "शब्द बादल",
        "Language": "भाषा",
        "Stay healthy! For persistent or severe symptoms, consult a healthcare professional.": "स्वस्थ रहें! लगातार या गंभीर लक्षणों के लिए, एक स्वास्थ्य विशेषज्ञ से परामर्श करें।",
        "Please select at least one symptom.": "कृपया कम से कम एक लक्षण चुनें।"
    }
}

# --- Preprocessing, Vectorizer, Classifier (unchanged) ---
class SymptomPreprocessor:
    def __init__(self):
        pass
    def preprocess_symptoms(self, symptoms):
        # Normalize and clean symptoms
        if isinstance(symptoms, str):
            symptoms = symptoms.lower().strip()
            symptoms = [s.strip() for s in symptoms.split(',') if s.strip()]
        elif isinstance(symptoms, list):
            symptoms = [s.lower().strip() for s in symptoms if s and isinstance(s, str)]
        return list(set(symptoms))  # Remove duplicates

    def row_to_symptom_list(self, row):
        # Combine all symptom columns into a list
        symptoms = [str(s).strip().lower() for s in row if pd.notnull(s) and str(s).strip()]
        return list(set(symptoms))

# 2. Vectorization
class SymptomVectorizer:
    def __init__(self):
        self.vectorizer = CountVectorizer(tokenizer=lambda x: x, lowercase=False, binary=True)
    def fit(self, symptom_lists):
        texts = [' '.join(symptoms) for symptoms in symptom_lists]
        self.vectorizer.fit(texts)
    def transform(self, symptom_lists):
        texts = [' '.join(symptoms) for symptoms in symptom_lists]
        return self.vectorizer.transform(texts)
    def fit_transform(self, symptom_lists):
        texts = [' '.join(symptoms) for symptoms in symptom_lists]
        return self.vectorizer.fit_transform(texts)

# 3. Multi-label Classification
class DiseaseClassifier:
    def __init__(self):
        self.model = OneVsRestClassifier(LogisticRegression(solver='liblinear'))
        self.mlb = MultiLabelBinarizer()
        self.vectorizer = SymptomVectorizer()
    def fit(self, X_symptoms, y_diseases):
        # Fit label binarizer
        y = self.mlb.fit_transform(y_diseases)
        # Fit vectorizer
        X = self.vectorizer.fit_transform(X_symptoms)
        # Fit classifier
        self.model.fit(X, y)
    def predict(self, symptoms):
        X = self.vectorizer.transform([symptoms])
        y_pred_prob = self.model.predict_proba(X)[0]
        classes = self.mlb.classes_
        return dict(zip(classes, y_pred_prob))
    def predict_top(self, symptoms, top_n=5):
        scores = self.predict(symptoms)
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:top_n]
    def evaluate(self, X_symptoms, y_diseases):
        X = self.vectorizer.transform(X_symptoms)
        y_true = self.mlb.transform(y_diseases)
        y_pred = self.model.predict(X)
        return classification_report(y_true, y_pred, target_names=self.mlb.classes_, output_dict=True)

# --- Visualization functions ---
def plot_confidence_bar(predictions, lang):
    diseases, scores = zip(*predictions)
    fig = px.bar(
        x=diseases, y=scores,
        labels={'x':translations[lang]["Top Disease Predictions"], 'y':'Confidence'},
        title=translations[lang]["Confidence Chart"],
        color=scores, color_continuous_scale='Blues'
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_wordcloud(symptom_list, lang):
    if not symptom_list:
        st.info("No symptoms selected for word cloud.")
        return
    text = ' '.join(symptom_list)
    wordcloud = WordCloud(width=400, height=200, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig, use_container_width=True)

def plot_pie_chart(predictions, lang):
    diseases, scores = zip(*predictions)
    fig = px.pie(
        names=diseases,
        values=scores,
        title=translations[lang]["Confidence Chart"] + " (Pie)",
        color_discrete_sequence=px.colors.sequential.Blues
    )
    st.plotly_chart(fig, use_container_width=True)

def show_tips(disease, lang):
    tips = disease_tips.get(disease, ["Consult a physician for more information."])
    st.markdown(f"💡 **{translations[lang]['Tips']}:**")
    for tip in tips:
        st.write(f"- {tip}")

# --- Main Streamlit App ---
def run_app():
    st.markdown("""
        <h1 style='text-align: center; color: #4B8BBE;'>🩺 Disease Diagnosis System</h1>
        <div style='text-align:center;'>
        <span style='font-size:18px;'>A modern, interactive, and user-friendly disease prediction tool</span>
        </div>
        <hr style='border:1px solid #4B8BBE;'>
    """, unsafe_allow_html=True)
    lang = st.selectbox("Language / भाषा", ["en", "hi"], index=0)
    st.markdown(f"#### 🤒 {translations[lang]['Select your symptoms']}")

    # Load data and model
    X, y = load_data("dataset.csv")
    preprocessor = SymptomPreprocessor()
    classifier = DiseaseClassifier()
    classifier.fit(X, y)
    all_symptoms = sorted({symptom for sublist in X for symptom in sublist})

    selected_symptoms = st.multiselect(
        translations[lang]["Select your symptoms"],
        options=all_symptoms,
        help="Start typing to search and select symptoms."
    )
    st.markdown("---")
    if st.button(f"🔍 {translations[lang]['Predict']}", use_container_width=True):
        if not selected_symptoms:
            st.warning(translations[lang]["Please select at least one symptom."])
            return
        predictions = classifier.predict_top(selected_symptoms, top_n=5)
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader(f"🔬 {translations[lang]['Top Disease Predictions']}")
            for disease, score in predictions:
                st.markdown(f"<div style='font-size:18px'><b>{disease}</b>: <span style='color:#1f77b4'>{score*100:.2f}%</span></div>", unsafe_allow_html=True)
                show_tips(disease, lang)
        with col2:
            st.subheader(f"📊 {translations[lang]['Confidence Chart']}")
            plot_confidence_bar(predictions, lang)
            st.subheader("🟦 Confidence Pie Chart")
            plot_pie_chart(predictions, lang)
        with st.expander(f"🌐 {translations[lang]['Word Cloud']}"):
            plot_wordcloud(selected_symptoms, lang)
        st.markdown("---")
        st.success(translations[lang]["Stay healthy! For persistent or severe symptoms, consult a healthcare professional."])
    st.markdown("""
        <hr>
        <div style='text-align:center; color: #888;'>
        <small>Made with ❤️ using Streamlit | <a href='https://github.com/your-repo' target='_blank'>GitHub</a></small>
        </div>
    """, unsafe_allow_html=True)

# --- Data loading function (unchanged) ---
def load_data(path):
    df = pd.read_csv(path)
    preprocessor = SymptomPreprocessor()
    symptom_cols = [col for col in df.columns if col.startswith("Symptom")]
    df['symptom_list'] = df[symptom_cols].apply(lambda row: preprocessor.row_to_symptom_list(row), axis=1)
    df['Disease'] = df['Disease'].str.strip()
    X = df['symptom_list'].tolist()
    y = [[d] for d in df['Disease']]
    return X, y

if __name__ == "__main__":
    run_app()
