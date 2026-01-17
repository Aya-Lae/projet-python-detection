"""
PARTIE 1 : Classification de Chiffres Manuscrits (MNIST)
VERSION CORRIGÉE - Téléchargement automatique du dataset
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from sklearn.metrics import classification_report, confusion_matrix
import os


try:
    # Keras télécharge automatiquement MNIST
    (X_train, y_train), (X_test, y_test) = mnist.load_data()


except Exception as e:
    print(f" ERREUR lors du téléchargement : {e}")
    exit(1)

#=========== Prétraitement=================
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

#===============VISUALISATION===============
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
fig.suptitle(' Exemples du Dataset MNIST', fontsize=16, fontweight='bold')

for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i].reshape(28, 28), cmap='gray')
    ax.set_title(f'Chiffre: {y_train[i]}', fontsize=12, fontweight='bold')
    ax.axis('off')

plt.tight_layout()
plt.savefig('mnist_examples.png', dpi=150, bbox_inches='tight')
plt.close()

# ==================== CRÉATION DU MODÈLE ====================
model = models.Sequential([
    # Bloc 1 : Convolution
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),

    # Bloc 2 : Convolution
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Bloc 3 : Convolution
    layers.Conv2D(64, (3, 3), activation='relu'),

    # Classification
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')  # 10 classes (0-9)
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
print()

# ==================== ENTRAÎNEMENT ====================
print(f"   Temps estimé: 5-10 minutes (sans GPU)")

history = model.fit(
    X_train, y_train,
    batch_size=128,
    epochs=10,
    validation_split=0.1,
    verbose=1
)

# ==================== ÉVALUATION ====================
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Précision sur le test : {test_accuracy * 100:.2f}%")
print(f"Perte sur le test     : {test_loss:.4f}\n")

# Prédictions
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

# Rapport détaillé
print(classification_report(y_test, y_pred,
                            target_names=[str(i) for i in range(10)]))

# ==================== VISUALISATIONS ====================

# 1. Historique d'entraînement
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].plot(history.history['accuracy'], 'b-', label='Train', linewidth=2)
axes[0].plot(history.history['val_accuracy'], 'r-', label='Validation', linewidth=2)
axes[0].set_title('📈 Précision du Modèle MNIST', fontsize=16, fontweight='bold')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].legend(fontsize=12)
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['loss'], 'b-', label='Train', linewidth=2)
axes[1].plot(history.history['val_loss'], 'r-', label='Validation', linewidth=2)
axes[1].set_title('📉 Perte du Modèle MNIST', fontsize=16, fontweight='bold')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Loss', fontsize=12)
axes[1].legend(fontsize=12)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mnist_training_history.png', dpi=150, bbox_inches='tight')
print(" Historique sauvegardé : mnist_training_history.png")
plt.close()

# 2. Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(10), yticklabels=range(10),
            cbar_kws={'label': 'Nombre de prédictions'})
plt.title(' Matrice de Confusion MNIST', fontsize=18, fontweight='bold', pad=20)
plt.ylabel('Vrai Chiffre', fontsize=14)
plt.xlabel('Chiffre Prédit', fontsize=14)
plt.tight_layout()
plt.savefig('mnist_confusion_matrix.png', dpi=150, bbox_inches='tight')
print("Matrice sauvegardée : mnist_confusion_matrix.png")
plt.close()

# 3. Exemples de prédictions
fig, axes = plt.subplots(3, 5, figsize=(15, 9))
fig.suptitle('🔍 Exemples de Prédictions MNIST', fontsize=16, fontweight='bold')

indices = np.random.choice(len(X_test), 15, replace=False)

for i, (ax, idx) in enumerate(zip(axes.flat, indices)):
    ax.imshow(X_test[idx].reshape(28, 28), cmap='gray')

    true_label = y_test[idx]
    pred_label = y_pred[idx]

    color = 'green' if true_label == pred_label else 'red'
    ax.set_title(f'Vrai: {true_label} | Prédit: {pred_label}',
                 color=color, fontsize=11, fontweight='bold')
    ax.axis('off')

plt.tight_layout()
plt.savefig('mnist_predictions.png', dpi=150, bbox_inches='tight')
plt.close()

# ==================== SAUVEGARDE DU MODÈLE ====================
model_path = 'mnist_model.keras'
model.save(model_path)
print(f"\n Modèle sauvegardé : {model_path}")

# ==============Vérifier que le fichier existe====================
if os.path.exists(model_path):
    file_size = os.path.getsize(model_path) / (1024 * 1024)  # En MB
    print(f" Fichier créé avec succès ({file_size:.2f} MB)")
else:
    print(" ERREUR : Le fichier n'a pas été créé")

