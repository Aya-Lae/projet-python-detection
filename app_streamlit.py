import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow import keras
import os
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# Configuration de la page
st.set_page_config(
    page_title="IA Médicale - Détection de Maladies",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #667eea;
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem;
        border-radius: 10px;
    }
    .success-box {
        background: #black;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
    }
    .warning-box {
        background: #black;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
    }
    .danger-box {
        background: #red;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# Classes
BRAIN_TUMOR_CLASSES = ['Glioma', 'Méningiome', 'Pas de Tumeur', 'Pituitaire']
MNIST_CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Descriptions médicales
TUMOR_DESCRIPTIONS = {
    'Glioma': 'Tumeur maligne des cellules gliales. Traitement urgent recommandé.',
    'Méningiome': 'Tumeur généralement bénigne des méninges. Surveillance nécessaire.',
    'Pituitaire': 'Tumeur de la glande pituitaire. Consultation endocrinologue recommandée.',
    'Pas de Tumeur': 'Aucune anomalie détectée. Cerveau sain.'
}


@st.cache_resource
def load_models():
    """Charge les modèles (avec cache Streamlit)"""
    brain_model = None
    mnist_model = None

    # Chemins possibles
    brain_paths = [
        'web_application/models/brain_tumor_model.keras',
        'partie2_medical/brain_tumor_model.keras',
        'brain_tumor_model.keras'
    ]

    mnist_paths = [
        'web_application/models/mnist_model.keras',
        'partie1_mnist/mnist_model.keras',
        'mnist_model.keras'
    ]

    # Charger modèle tumeurs
    for path in brain_paths:
        if os.path.exists(path):
            try:
                brain_model = keras.models.load_model(path)
                break
            except:
                continue

    # Charger modèle MNIST
    for path in mnist_paths:
        if os.path.exists(path):
            try:
                mnist_model = keras.models.load_model(path)
                break
            except:
                continue

    return brain_model, mnist_model


def preprocess_brain_image(image):
    """Prétraite une image IRM"""
    img = cv2.resize(np.array(image), (150, 150))
    if len(img.shape) == 2:  # Grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=0)


def preprocess_mnist_image(image):
    """Prétraite une image MNIST"""
    img = cv2.resize(np.array(image.convert('L')), (28, 28))
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=(0, -1))


def create_probability_chart(predictions, classes):
    """Crée un graphique de probabilités"""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#667eea' if i == np.argmax(predictions) else '#95a5a6'
              for i in range(len(predictions))]

    bars = ax.barh(classes, predictions * 100, color=colors)
    ax.set_xlabel('Probabilité (%)', fontsize=12, fontweight='bold')
    ax.set_title('📊 Probabilités par Classe', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 100)

    # Ajouter les valeurs
    for i, (bar, prob) in enumerate(zip(bars, predictions)):
        width = bar.get_width()
        ax.text(width + 2, bar.get_y() + bar.get_height() / 2,
                f'{prob * 100:.2f}%',
                va='center', fontweight='bold')

    plt.tight_layout()
    return fig


def main():
    """Application principale Streamlit"""

    # Header
    st.markdown('<h1 class="main-header">🧠 IA MÉDICALE - DÉTECTION DE MALADIES</h1>',
                unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #7f8c8d;">'
                'Intelligence Artificielle pour l\'Analyse d\'Images Médicales | </p>',
                unsafe_allow_html=True)

    st.markdown("---")

    # Charger les modèles
    brain_model, mnist_model = load_models()

    # Sidebar

    page = st.sidebar.radio(
        "Choisir une analyse :",
        ["🧠 Tumeurs Cérébrales", "🔢 Chiffres MNIST",
         "📊 Résultats", ],
        index=0
    )


    # Pages
    if page == "🏠 Accueil":
        show_home_page()

    elif page == "🧠 Tumeurs Cérébrales":
        show_brain_tumor_page(brain_model)

    elif page == "🔢 Chiffres MNIST":
        show_mnist_page(mnist_model)

    elif page == "📊 Résultats":
        show_results_page()

    elif page == "ℹ️ À Propos":
        show_about_page()


def show_home_page():
    """Page d'accueil"""

    # Métriques
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <h2 style="margin: 0;">2</h2>
            <p style="margin: 0;">Modèles IA</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <h2 style="margin: 0;">98%</h2>
            <p style="margin: 0;">Précision MNIST</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <h2 style="margin: 0;">95%</h2>
            <p style="margin: 0;">Précision Tumeurs</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-card">
            <h2 style="margin: 0;">&lt;3s</h2>
            <p style="margin: 0;">Temps d'Analyse</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("##")

    # Fonctionnalités
    st.header("✨ Fonctionnalités")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="success-box">
            <h3>🧠 Détection de Tumeurs Cérébrales</h3>
            <p>Analyse d'images IRM pour identifier 4 types de tumeurs : Glioma, 
            Méningiome, Pituitaire, ou absence de tumeur.</p>
            <ul>
                <li>Précision : ~95%</li>
                <li>Dataset : 4,000 images</li>
                <li>Temps : 1-3 secondes</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="success-box">
            <h3>🔢 Reconnaissance de Chiffres MNIST</h3>
            <p>Identification de chiffres manuscrits de 0 à 9 avec une précision 
            exceptionnelle.</p>
            <ul>
                <li>Précision : ~98-99%</li>
                <li>Dataset : 60,000 images</li>
                <li>Temps : <1 seconde</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("##")

    # Comment ça marche
    st.header("🔍 Comment Ça Marche ?")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 20px; background: #f8f9fa; border-radius: 10px;">
            <h2 style="color: #667eea;">1️⃣</h2>
            <h4>Uploader une Image</h4>
            <p>Sélectionnez une image IRM ou un chiffre manuscrit</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 20px; background: #f8f9fa; border-radius: 10px;">
            <h2 style="color: #667eea;">2️⃣</h2>
            <h4>Analyse par IA</h4>
            <p>Le CNN analyse l'image en temps réel</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 20px; background: #f8f9fa; border-radius: 10px;">
            <h2 style="color: #667eea;">3️⃣</h2>
            <h4>Résultats Instantanés</h4>
            <p>Diagnostic avec niveau de confiance</p>
        </div>
        """, unsafe_allow_html=True)


def show_brain_tumor_page(model):
    """Page de détection de tumeurs"""

    st.header("🧠 Détection de Tumeurs Cérébrales")

    if model is None:
        st.error("❌ Le modèle de détection de tumeurs n'est pas disponible. "
                 "Assurez-vous d'avoir entraîné le modèle et que le fichier .keras existe.")
        return

    st.markdown("""
    <div class="warning-box">
        <strong> Information :</strong> Uploadez une image IRM de cerveau pour détecter 
        la présence et le type de tumeur.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("##")

    # Upload
    uploaded_file = st.file_uploader(
        "Choisir une image IRM",
        type=['png', 'jpg', 'jpeg'],
        help="Formats acceptés : PNG, JPG, JPEG"
    )

    if uploaded_file is not None:
        # Afficher l'image
        col1, col2 = st.columns([1, 1])

        with col1:

            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)

        # Analyser
        if st.button(" Analyser l'IRM", type="primary"):
            with st.spinner(" Analyse en cours..."):
                # Prétraiter et prédire
                img_array = preprocess_brain_image(image)
                predictions = model.predict(img_array, verbose=0)[0]
                predicted_class = int(np.argmax(predictions))
                confidence = float(predictions[predicted_class]) * 100

                with col2:
                    st.subheader("Diagnostic")

                    # Diagnostic
                    if predicted_class == 2:  # Pas de tumeur
                        st.markdown(f"""
                        <div class="success-box">
                            <h2 style="margin: 0; color: #28a745;">{BRAIN_TUMOR_CLASSES[predicted_class]}</h2>
                            <h3 style="margin: 10px 0;">Confiance : {confidence:.2f}%</h3>
                            <p><strong>Interprétation :</strong><br>{TUMOR_DESCRIPTIONS[BRAIN_TUMOR_CLASSES[predicted_class]]}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="danger-box">
                            <h2 style="margin: 0; color: #dc3545;">{BRAIN_TUMOR_CLASSES[predicted_class]}</h2>
                            <h3 style="margin: 10px 0;">Confiance : {confidence:.2f}%</h3>
                            <p><strong>Interprétation :</strong><br>{TUMOR_DESCRIPTIONS[BRAIN_TUMOR_CLASSES[predicted_class]]}</p>
                        </div>
                        """, unsafe_allow_html=True)

                # Graphique
                st.markdown("##")
                st.subheader("📊 Probabilités Détaillées")
                fig = create_probability_chart(predictions, BRAIN_TUMOR_CLASSES)
                st.pyplot(fig)

                # Avertissement
                st.markdown("##")
                st.markdown("""
                <div class="danger-box">
                    <strong>⚠️ Avertissement Médical :</strong> Ce diagnostic est généré par IA 
                    à des fins éducatives uniquement. Il ne remplace pas l'avis d'un professionnel 
                    de santé. Consultez toujours un médecin pour un diagnostic officiel.
                </div>
                """, unsafe_allow_html=True)


def show_mnist_page(model):
    """Page MNIST"""

    st.header("🔢 Reconnaissance de Chiffres MNIST")

    if model is None:
        st.error("❌ Le modèle MNIST n'est pas disponible. "
                 "Assurez-vous d'avoir entraîné le modèle avec mnist_classifier.py")
        return

    st.info("️ Uploadez une image d'un chiffre manuscrit (0-9) pour le reconnaître.")

    st.markdown("##")

    # Upload
    uploaded_file = st.file_uploader(
        "📤 Choisir une image de chiffre",
        type=['png', 'jpg', 'jpeg'],
        help="Image d'un chiffre manuscrit"
    )

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader(" Image Uploadée")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)

        if st.button("Reconnaître le Chiffre", type="primary"):
            with st.spinner("🔄 Analyse en cours..."):
                img_array = preprocess_mnist_image(image)
                predictions = model.predict(img_array, verbose=0)[0]
                predicted_class = int(np.argmax(predictions))
                confidence = float(predictions[predicted_class]) * 100

                with col2:
                    st.subheader("Résultat")

                    st.markdown(f"""
                    <div class="success-box" style="text-align: center;">
                        <h1 style="font-size: 5rem; margin: 0; color: #28a745;">{predicted_class}</h1>
                        <h3 style="margin: 10px 0;">Confiance : {confidence:.2f}%</h3>
                        <p>✅ Le modèle a reconnu le chiffre : <strong>{predicted_class}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("##")
                st.subheader("📊 Probabilités pour Chaque Chiffre")
                fig = create_probability_chart(predictions, MNIST_CLASSES)
                st.pyplot(fig)


def show_results_page():
    """Page des résultats d'entraînement"""

    st.header("📊 Résultats d'Entraînement des Modèles")

    st.info("Cette page affiche les graphiques générés lors de l'entraînement des modèles.")

    # Chercher les graphiques
    image_dirs = ['web_application/static/images', 'partie1_mnist', 'partie2_medical', '.']
    found_images = []

    for directory in image_dirs:
        if os.path.exists(directory):
            for file in os.listdir(directory):
                if file.endswith('.png'):
                    full_path = os.path.join(directory, file)
                    if full_path not in [img[1] for img in found_images]:
                        found_images.append((file, full_path))

    if found_images:
        st.success(f"✅ {len(found_images)} graphiques trouvés")

        # Afficher les graphiques
        for filename, filepath in found_images:
            st.markdown(f"### 📈 {filename.replace('.png', '').replace('_', ' ').title()}")
            try:
                image = Image.open(filepath)
                st.image(image, use_container_width=True)
            except:
                st.error(f"Impossible de charger : {filename}")
            st.markdown("---")
    else:
        st.warning("""
        ⚠️ **Aucun graphique trouvé**

        Pour générer les graphiques, entraînez d'abord les modèles :

        1. **MNIST** : `python partie1_mnist/mnist_classifier.py`
        2. **Tumeurs** : `python partie2_medical/run_model.py`

        Les graphiques seront automatiquement générés après l'entraînement.
        """)


def show_about_page():
    """Page à propos"""

    st.header("ℹ️ À Propos du Projet")

    st.markdown("""
    ## 🎯 Objectif

    Ce projet vise à développer des modèles d'intelligence artificielle capables de détecter 
    des maladies à partir d'images médicales. L'application est développée **entièrement en Python** 
    avec Streamlit.

    ## 📚 Structure du Projet
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="success-box">
            <h3>🔢 Partie 1 : MNIST</h3>
            <p>Classification de chiffres manuscrits (0-9)</p>
            <ul>
                <li>Dataset : 60,000 images</li>
                <li>Architecture : CNN 3 couches</li>
                <li>Précision : ~98-99%</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="success-box">
            <h3>🧠 Partie 2 : Tumeurs</h3>
            <p>Détection de tumeurs cérébrales</p>
            <ul>
                <li>Dataset : 4,000 images IRM</li>
                <li>Architecture : CNN 4 couches</li>
                <li>Précision : ~95%</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("##")



if __name__ == "__main__":
    main()