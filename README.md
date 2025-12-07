# � Bbox Stories

*Glisse ta vidéo. On la remixe en boîtes, formes et couleurs.*

Bbox Stories est une petite web app Streamlit pensée pour jouer avec tes vidéos perso (stories Instagram, reels, clips, etc.) et les transformer en versions "augmentées" : une vidéo d'origine, mais recouverte de bounding boxes, formes et labels colorés.

Sous le capot, l'app utilise des modèles de détection type YOLO (via `ultralytics`) et la librairie `supervision` pour gérer les détections et le rendu visuel.

## Idée du projet

L'objectif n'est pas seulement la détection d'objets classique, mais plutôt :

- **Remixer visuellement des stories** avec des bboxes et des formes.
- **Explorer différents styles** (rectangles, coins, cercles, ellipses, traces, points).
- **Jouer avec les couleurs** grâce à des palettes inspirées de matplotlib.
- **Contrôler la densité visuelle** (seuil de confiance, IOU, labels ou scores, taille du texte, épaisseur des traits).

En résumé : un outil fun pour générer des vidéos "alternatives" à partir de ton propre contenu.

## Fonctionnalités principales

- **Upload par drag & drop** d'une vidéo (`mp4`, `avi`, `mov`, `mkv`).
- **Choix du modèle** : modèles pré-entraînés YOLO11 ou modèle `.pt` custom.
- **Aperçu multi-frames** : première, milieu et dernière frame annotées avant de lancer le traitement complet.
- **Contrôle des paramètres de détection** : seuil de confiance et IOU via listes déroulantes.
- **Styles de bbox** : rectangle, arrondi, coins, cercle, point, ellipse, trace.
- **Palettes de couleurs** : sélection parmi plusieurs palettes (viridis, inferno, plasma, magma, etc.) avec aperçu via emoji.
- **Labels configurables** : aucun, label seul, score seul, label + score, avec contrôle de la taille du texte.
- **Vidéo de sortie compatible web** : ré-encodage pour lecture fluide dans le navigateur + bouton de téléchargement.

## Installation

1. **Pré-requis**
   - Avoir `pyenv` installé avec Python 3.12.12 (`pyenv install 3.12.12`)
   - Avoir `ffmpeg` installé sur votre système.

2. **Configurer l'environnement :**

   ```bash
   # Définir la version de python locale
   pyenv local 3.12.12

   # Création de l'environnement virtuel
   python -m venv venv

   # Activation (Linux/Mac)
   source venv/bin/activate
   # Activation (Windows)
   # .\venv\Scripts\activate

   # Installation des paquets
   pip install -r requirements.txt
   ```

## Utilisation

Une fois l'environnement activé et les dépendances installées :

```bash
streamlit run app.py
```

Quelques étapes dans l'interface :

1. Choisir un modèle (pré-entraîné ou `.pt` custom).
2. Glisser / déposer une vidéo.
3. Ajuster les paramètres dans la sidebar :
   - Seuil de confiance / IOU.
   - Style des bbox + épaisseur.
   - Palette de couleurs.
   - Affichage des labels (ou non) + taille du texte.
4. Regarder l'**aperçu** (3 frames annotées : début / milieu / fin).
5. Lancer le traitement complet et télécharger la vidéo annotée.

L'application s'ouvrira automatiquement dans votre navigateur (généralement à l'adresse `http://localhost:8501`).


----

yolo world
