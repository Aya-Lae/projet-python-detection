import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ==================== CONFIGURATION ====================
IMG_SIZE = 150  # Réduit pour accélérer l'entraînement
BATCH_SIZE = 32
EPOCHS = 30  # Réduit à 30 pour commencer
DATA_DIR = '../data/Training'

# Classes de tumeurs
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# ==================== CHARGEMENT DES DONNÉES ====================
def load_data(data_dir):

    images = []
    labels = []

    if not os.path.exists(data_dir):
        print(f"\n❌ ERREUR : Le dossier '{data_dir}' n'existe pas!")
        return None, None

    total_images = 0
    for class_idx, class_name in enumerate(CLASSES):
        class_path = os.path.join(data_dir, class_name)

        if not os.path.exists(class_path):
            print(f"⚠️  ATTENTION : Dossier '{class_name}' introuvable dans {data_dir}")
            continue

        img_files = [f for f in os.listdir(class_path)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]


        loaded = 0
        for img_name in img_files[:1000]:  # Limite à 1000 images par classe
            img_path = os.path.join(class_path, img_name)

            try:
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = img.astype('float32') / 255.0

                    images.append(img)
                    labels.append(class_idx)
                    loaded += 1
            except Exception as e:
                continue

        print(f"   Images chargées : {loaded}")
        total_images += loaded

    if total_images == 0:
        print("\n ERREUR : Aucune image chargée!")
        return None, None

    return np.array(images), np.array(labels)


# ==================== CRÉATION DU MODÈLE ====================
def create_model(input_shape=(150, 150, 3), num_classes=4):

    model = models.Sequential([
        # Bloc 1
        layers.Conv2D(32, (3, 3), activation='relu',
                      input_shape=input_shape, padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Bloc 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Bloc 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Bloc 4
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),

        # Classification
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("Modèle créé avec succès!\n")
    model.summary()
    print()

    return model


# ==================== VISUALISATIONS ====================
def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Précision
    axes[0].plot(history.history['accuracy'], 'b-', label='Train', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], 'r-', label='Validation', linewidth=2)
    axes[0].set_title('📈 Précision du Modèle', fontsize=16, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # Perte
    axes[1].plot(history.history['loss'], 'b-', label='Train', linewidth=2)
    axes[1].plot(history.history['val_loss'], 'r-', label='Validation', linewidth=2)
    axes[1].set_title('📉 Perte du Modèle', fontsize=16, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(fontsize=12)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    print("\n💾 Graphique sauvegardé : training_history.png")
    plt.show()


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES,
                cbar_kws={'label': 'Nombre de prédictions'},
                annot_kws={'size': 14})

    plt.title(' Matrice de Confusion', fontsize=18, fontweight='bold', pad=20)
    plt.ylabel('Vraie Classe', fontsize=14)
    plt.xlabel('Classe Prédite', fontsize=14)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("Matrice sauvegardée : confusion_matrix.png")
    plt.show()


def plot_sample_predictions(X_test, y_test, y_pred, num_samples=12):
    """Affiche des exemples de prédictions"""
    plt.figure(figsize=(15, 10))

    # Sélectionner des échantillons aléatoires
    indices = np.random.choice(len(X_test), num_samples, replace=False)

    for i, idx in enumerate(indices):
        plt.subplot(3, 4, i + 1)
        plt.imshow(X_test[idx])

        true_label = CLASSES[y_test[idx]]
        pred_label = CLASSES[y_pred[idx]]

        color = 'green' if y_test[idx] == y_pred[idx] else 'red'
        plt.title(f'Vrai: {true_label}\nPrédit: {pred_label}',
                  color=color, fontsize=10, fontweight='bold')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('sample_predictions.png', dpi=150, bbox_inches='tight')
    print("💾 Exemples sauvegardés : sample_predictions.png")
    plt.show()


# ==================== FONCTION PRINCIPALE ====================
def main():
    # Étape 1 : Chargement des données
    X, y = load_data(DATA_DIR)

    if X is None or len(X) == 0:

        return

    # Étape 2 : Division des données

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    # Distribution des classes
    print("\n   📊 Distribution (Train) :")
    for i, class_name in enumerate(CLASSES):
        count = np.sum(y_train == i)
        pct = (count / len(y_train)) * 100
        print(f"      {class_name:12s} : {count:4d} ({pct:5.1f}%)")

    print()

    # Étape 3 : Création du modèle
    model = create_model(input_shape=(IMG_SIZE, IMG_SIZE, 3))

    # Étape 4 : Entraînement
    print(f"   Temps estimé: 20-30 minutes (sans GPU)")


    # Callbacks
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )

    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # Étape 5 : Évaluation


    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n   🎯 Test Accuracy : {test_accuracy * 100:.2f}%")
    print(f"   📉 Test Loss     : {test_loss:.4f}\n")

    # Prédictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Rapport détaillé

    print(classification_report(y_test, y_pred, target_names=CLASSES))

    # Étape 6 : Visualisations
    print("=" * 70)
    print("ÉTAPE 6/6 : GÉNÉRATION DES VISUALISATIONS")
    print("=" * 70 + "\n")

    plot_training_history(history)
    plot_confusion_matrix(y_test, y_pred)
    plot_sample_predictions(X_test, y_test, y_pred)

    # Sauvegarde du modèle
    model_path = 'brain_tumor_model.keras'
    model.save(model_path)
    print(f"\n Modèle sauvegardé : {model_path}")

    # Résumé final
    print("\n" + "=" * 70)
    print("✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
    print("=" * 70)
    print("\n📁 FICHIERS GÉNÉRÉS :")
    print("   1. brain_tumor_model.keras       (Modèle entraîné)")
    print("   2. training_history.png          (Graphiques d'entraînement)")
    print("   3. confusion_matrix.png          (Matrice de confusion)")
    print("   4. sample_predictions.png        (Exemples de prédictions)")

    print("\n📋 PROCHAINES ÉTAPES :")
    print("   1. Consultez les graphiques générés")
    print("   2. Testez le modèle : python predict.py")
    print("   3. Partagez vos résultats!")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Entraînement interrompu par l'utilisateur.")
    except Exception as e:
        print(f"\n\n❌ ERREUR : {e}")
        import traceback

        traceback.print_exc()