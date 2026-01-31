# 🧠 Brain Tumor Detection using AI

## 📌 Description
Application d'intelligence artificielle pour la détection de tumeurs cérébrales à partir d'images médicales IRM, utilisant le deep learning avec TensorFlow et une interface web interactive développée avec Streamlit.

## 🎯 Objectif du Projet
Ce projet académique vise à développer un modèle de machine learning capable de classifier des images IRM cérébrales et détecter la présence de tumeurs avec une haute précision, tout en offrant une interface utilisateur intuitive pour les professionnels de santé.

## 🛠️ Stack Technique

### Backend & Machine Learning
- **Python 3.x**
- **TensorFlow / Keras** - Deep learning framework
- **NumPy** - Calculs numériques
- **Pandas** - Manipulation de données
- **OpenCV / Pillow** - Traitement d'images
- **Scikit-learn** - Preprocessing et métriques

### Frontend
- **Streamlit** - Interface web interactive
- **Matplotlib / Seaborn** - Visualisation des résultats

## ✨ Fonctionnalités

- ✅ **Upload d'images IRM** - Interface drag & drop
- ✅ **Détection en temps réel** - Prédiction instantanée
- ✅ **Classification binaire** - Tumeur détectée / Non détectée
- ✅ **Visualisation des résultats** - Affichage avec niveau de confiance
- ✅ **Preprocessing automatique** - Normalisation et redimensionnement
- ✅ **Rapport de prédiction** - Statistiques et probabilités

## 🧪 Architecture du Modèle

Le modèle utilise un réseau de neurones convolutionnel (CNN) avec:
- Couches de convolution pour l'extraction de features
- Pooling layers pour la réduction dimensionnelle
- Couches fully connected pour la classification
- Fonction d'activation ReLU et Softmax
- Optimiseur Adam avec fonction de perte categorical crossentropy

**Métriques de performance:**
- Précision (Accuracy): [Ajoute ton score si tu l'as]
- Recall / Sensitivity
- F1-Score
- Matrice de confusion

## 📁 Structure du Projet
```
projet-python-detection/
│
├── data/
│   ├── train/          # Images d'entraînement
│   └── test/           # Images de test
│
├── models/
│   └── tumor_detector.h5   # Modèle entraîné
│
├── src/
│   ├── preprocessing.py     # Preprocessing des images
│   ├── model.py            # Architecture du modèle
│   ├── train.py            # Script d'entraînement
│   └── predict.py          # Script de prédiction
│
├── app.py              # Application Streamlit
├── requirements.txt    # Dépendances Python
└── README.md          # Documentation
```

## 🚀 Installation & Utilisation

### Prérequis
- Python 3.7+
- pip

### Installation
```bash
# Clone le repository
git clone https://github.com/Aya-Lae/projet-python-detection.git
cd projet-python-detection

# Créer un environnement virtuel (optionnel mais recommandé)
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

### Lancer l'application
```bash
# Démarrer l'interface Streamlit
streamlit run app.py
```

L'application sera accessible sur `http://localhost:8501`

### Utilisation

1. Ouvrir l'application dans le navigateur
2. Uploader une image IRM cérébrale (formats: JPG, PNG)
3. Cliquer sur "Analyser"
4. Consulter les résultats de détection avec le niveau de confiance

## 📊 Dataset

Le modèle a été entraîné sur un dataset d'images IRM cérébrales comprenant:
- Images avec tumeurs cérébrales
- Images saines (contrôle)
- [Précise la source du dataset si publique]

**Preprocessing appliqué:**
- Redimensionnement à [dimension] x [dimension] pixels
- Normalisation des valeurs de pixels (0-1)
- Augmentation de données (rotation, flip, zoom)

## 🎓 Compétences Développées

- Deep Learning et Computer Vision
- Preprocessing et augmentation de données
- Architecture de réseaux de neurones convolutionnels (CNN)
- Déploiement de modèles ML avec interface web
- Traitement d'images médicales
- Framework TensorFlow/Keras
- Développement d'applications avec Streamlit

## 🔮 Améliorations Futures

- [ ] Classification multi-classe (types de tumeurs)
- [ ] Segmentation précise de la zone tumorale
- [ ] Intégration d'autres architectures (ResNet, VGG16, etc.)
- [ ] API REST pour intégration dans d'autres systèmes
- [ ] Export des rapports de diagnostic en PDF
- [ ] Historique des analyses

## 📚 Contexte Académique

Projet réalisé à l'**ENSA Berrechid** dans le cadre du cycle ingénieur en Génie Informatique, pour approfondir les connaissances en intelligence artificielle et machine learning appliqués au domaine médical.

## 👨‍💻 Auteur

**Aya Laaouine**  
Étudiante Ingénieur en Informatique - ENSA Berrechid  
- GitHub: [@Aya-Lae](https://github.com/Aya-Lae)
- LinkedIn: [Aya Laaouine](https://linkedin.com/in/aya-laaouine830222360)
- Email: ayalaaouine2@gmail.com

## 📄 License

Ce projet est développé à des fins éducatives et académiques.

## ⚠️ Disclaimer

Ce projet est un outil éducatif de démonstration. Il ne doit pas être utilisé pour des diagnostics médicaux réels sans validation clinique appropriée.

---

*Projet développé avec passion pour l'IA et la santé numérique* 🚀
```
