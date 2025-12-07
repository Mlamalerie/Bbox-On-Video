import streamlit as st
import cv2
import tempfile
import os
import time
from ultralytics import YOLO
import supervision as sv


@st.cache_resource
def load_model(model_path: str):
    """Charge et met en cache un modèle YOLO pour un chemin donné."""
    return YOLO(model_path)


MAX_DIRECT_SIDE = 1280


def create_box_annotator(style: str, thickness: int, palette: sv.ColorPalette | None):
    """Retourne un annotateur Supervision selon le style demandé et la palette donnée."""

    color_kwargs = {"color": palette} if palette is not None else {}

    if style == "Rectangle":
        return sv.BoxAnnotator(thickness=thickness, **color_kwargs)
    if style == "Arrondi":
        return sv.RoundBoxAnnotator(thickness=thickness, **color_kwargs)
    if style == "Coins":
        return sv.BoxCornerAnnotator(thickness=thickness, **color_kwargs)
    if style == "Cercle":
        return sv.CircleAnnotator(thickness=thickness, **color_kwargs)
    if style == "Point":
        # DotAnnotator utilise un radius plutôt qu'une épaisseur
        return sv.DotAnnotator(radius=max(1, thickness * 2), **color_kwargs)
    if style == "Ellipse":
        return sv.EllipseAnnotator(thickness=thickness, **color_kwargs)
    if style == "Trace":
        return sv.TraceAnnotator(thickness=thickness, **color_kwargs)
    # Fallback
    return sv.BoxAnnotator(thickness=thickness, **color_kwargs)


def run_detections(
    frame,
    model,
    conf_threshold: float,
    iou_threshold: float,
    use_sahi: bool,
):
    """Calcule les détections sur une frame.

    - Si use_sahi est False ou que l'image est "petite", on appelle YOLO directement.
    - Sinon, on utilise sv.InferenceSlicer pour faire un découpage type SAHI.
    """

    height, width = frame.shape[:2]

    if (not use_sahi) or max(height, width) <= MAX_DIRECT_SIDE:
        results = model(frame, conf=conf_threshold, iou=iou_threshold, verbose=False)[0]
        return sv.Detections.from_ultralytics(results)

    def callback(image_slice):
        results = model(
            image_slice,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False,
        )[0]
        return sv.Detections.from_ultralytics(results)

    slicer = sv.InferenceSlicer(
        callback=callback,
        slice_wh=(768, 768),
        overlap_ratio_wh=(0.2, 0.2),
    )

    return slicer(image=frame)


st.set_page_config(page_title="Bbox Stories", layout="wide")

st.title("� Bbox Stories")
st.markdown(
    "*Glisse ta vidéo. On la remixe en boîtes, formes et couleurs.*"
)

# Sidebar pour la configuration
st.sidebar.header("Configuration")

# Sélection du modèle
model_type = st.sidebar.radio(
    "Source du modèle",
    ("Modèles pré-entraînés (YOLO11)", "Charger un modèle personnalisé (.pt)")
)

model_path = None

if model_type == "Modèles pré-entraînés (YOLO11)":
    selected_model = st.sidebar.selectbox(
        "Choisir un modèle YOLO11",
        ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"],
    )
    model_path = selected_model
else:
    uploaded_model = st.sidebar.file_uploader("Charger votre fichier .pt", type=["pt"])
    if uploaded_model is not None:
        # Save uploaded model to a temp file so ultralytics can load it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_model:
            tmp_model.write(uploaded_model.read())
            model_path = tmp_model.name

# Paramètres de détection
st.sidebar.subheader("Paramètres de détection")

conf_options = [0.2, 0.3, 0.4, 0.5, 0.6]
iou_options = [0.3, 0.4, 0.5, 0.6, 0.7]

conf_threshold = st.sidebar.selectbox(
    "Seuil de confiance",
    options=conf_options,
    index=2,  # 0.4 par défaut
)

iou_threshold = st.sidebar.selectbox(
    "Seuil IOU",
    options=iou_options,
    index=2,  # 0.5 par défaut
)

# Style visuel des bbox
st.sidebar.subheader("Style des bbox")

bbox_thickness = st.sidebar.selectbox(
    "Épaisseur des bbox",
    options=[1, 2, 4, 8, 16],
    index=1,  # 2 par défaut
)

bbox_style_map = {
    "⬛ Rectangle": "Rectangle",
    "🟦 Arrondi": "Arrondi",
    "📐 Coins": "Coins",
    "⚪ Cercle": "Cercle",
    "🔹 Point": "Point",
    "💠 Ellipse": "Ellipse",
    "🌀 Trace": "Trace",
}

selected_bbox_style_label = st.sidebar.selectbox(
    "Style de bbox",
    options=list(bbox_style_map.keys()),
    index=0,
)

bbox_style = bbox_style_map[selected_bbox_style_label]

palette_labels = [
    "Aucune",
    "🌈 viridis",
    "🔥 inferno",
    "💜 plasma",
    "🌋 magma",
    "🌊 cividis",
    "🍃 Greens",
    "🟦 cool",
    "🌅 autumn",
    "💛 Wistia",
    "🔮 Purples",
]

palette_map = {
    "Aucune": None,
    "🌈 viridis": "viridis",
    "🔥 inferno": "inferno",
    "💜 plasma": "plasma",
    "🌋 magma": "magma",
    "🌊 cividis": "cividis",
    "🍃 Greens": "Greens",
    "🟦 cool": "cool",
    "🌅 autumn": "autumn",
    "💛 Wistia": "Wistia",
    "🔮 Purples": "Purples",
}

palette_choice = st.sidebar.selectbox(
    "Palette de couleurs",
    options=palette_labels,
    index=0,
)

bbox_palette = None
mpl_palette_name = palette_map.get(palette_choice)
if mpl_palette_name is not None:
    # N = 16 couleurs distinctes par défaut
    bbox_palette = sv.ColorPalette.from_matplotlib(mpl_palette_name, 16)

# Labels
st.sidebar.subheader("Labels")

label_mode = st.sidebar.selectbox(
    "Affichage des labels",
    options=[
        "Aucun",
        "Label seulement",
        "Score seulement",
        "Label + score",
    ],
    index=3,
)

# "Puissances de 2" pour l'échelle de texte (labels)
if label_mode != "Aucun":
    label_scale = st.sidebar.selectbox(
        "Taille des labels",
        options=[0.25, 0.5, 1.0, 2.0],  # ~2^-2, 2^-1, 2^0, 2^1
        index=1,  # 0.5 par défaut
    )
else:
    # valeur par défaut utilisée mais non affichée
    label_scale = 0.5

# SAHI / grandes vidéos
st.sidebar.subheader("SAHI / grandes vidéos")

use_sahi = st.sidebar.checkbox(
    "Activer le découpage (SAHI) pour les grandes images",
    value=True,
    help=(
        "Si l'image est plus grande qu'une certaine taille, elle est découpée en tuiles "
        "pour la détection, ce qui peut améliorer la détection des petits objets."
    ),
)

# Upload vidéo
uploaded_video = st.file_uploader("Choisissez une vidéo", type=["mp4", "avi", "mov", "mkv"])

if uploaded_video and model_path:
    # Sauvegarde temporaire de la vidéo uploadée
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    video_path = tfile.name

    # Nom de base dérivé de la vidéo d'entrée
    original_name = uploaded_video.name or "video"
    base_name, _ = os.path.splitext(original_name)

    st.video(video_path)

    # Chargement du modèle (partagé entre aperçu et traitement complet)
    st.write("Chargement du modèle...")
    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle: {e}")
        st.stop()

    # Aperçu sur plusieurs frames (début, milieu, fin)
    try:
        video_info = sv.VideoInfo.from_video_path(video_path)
        total_frames = video_info.total_frames or 0

        if total_frames <= 0:
            raise ValueError("Nombre de frames invalide pour l'aperçu")

        indices = [0]
        if total_frames > 2:
            indices.append(total_frames // 2)
        if total_frames > 1:
            indices.append(total_frames - 1)

        cap = cv2.VideoCapture(video_path)
        preview_images = []  # list of (title, annotated_frame)

        def annotate_frame(frame):
            detections = run_detections(
                frame,
                model,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                use_sahi=use_sahi,
            )

            box_annot = create_box_annotator(bbox_style, bbox_thickness, bbox_palette)
            if bbox_palette is not None:
                label_annot = sv.LabelAnnotator(color=bbox_palette, text_scale=label_scale)
            else:
                label_annot = sv.LabelAnnotator(text_scale=label_scale)

            labels = []
            if detections.class_id is not None and detections.confidence is not None:
                for class_id, confidence in zip(detections.class_id, detections.confidence):
                    class_name = model.model.names[class_id]
                    if label_mode == "Label seulement":
                        labels.append(f"{class_name}")
                    elif label_mode == "Score seulement":
                        labels.append(f"{confidence:.2f}")
                    elif label_mode == "Label + score":
                        labels.append(f"{class_name} {confidence:.2f}")

            annotated = box_annot.annotate(scene=frame.copy(), detections=detections)
            if label_mode != "Aucun" and len(labels) > 0:
                annotated = label_annot.annotate(scene=annotated, detections=detections, labels=labels)

            return annotated

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            annotated = annotate_frame(frame)

            if idx == 0:
                title = "Première frame annotée"
            elif idx == indices[-1]:
                title = "Dernière frame annotée"
            else:
                title = "Frame du milieu annotée"

            preview_images.append((title, annotated))

        cap.release()

        if preview_images:
            st.subheader("Aperçu")
            cols = st.columns(len(preview_images))
            for col, (title, img) in zip(cols, preview_images):
                h, w = img.shape[:2]
                display_width = min(640, w)
                with col:
                    st.caption(title)
                    st.image(img[:, :, ::-1], width=display_width)
        else:
            st.warning("Impossible de générer l'aperçu : aucune frame lisible.")

    except Exception as e:
        st.warning(f"Impossible de générer l'aperçu : {e}")

    if st.button("Lancer la détection"):
        st.write("Traitement en cours....")

        box_annotator = create_box_annotator(bbox_style, bbox_thickness, bbox_palette)
        if bbox_palette is not None:
            label_annotator = sv.LabelAnnotator(color=bbox_palette, text_scale=label_scale)
        else:
            label_annotator = sv.LabelAnnotator(text_scale=label_scale)

        video_info = sv.VideoInfo.from_video_path(video_path)
        frame_generator = sv.get_video_frames_generator(source_path=video_path)

        output_path = os.path.join(tempfile.gettempdir(), f"{base_name}_annotated.mp4")

        progress_bar = st.progress(0)
        status_text = st.empty()

        start_time = time.time()

        try:
            with sv.VideoSink(target_path=output_path, video_info=video_info) as sink:
                for frame_index, frame in enumerate(frame_generator):
                    if video_info.total_frames:
                        progress = min((frame_index + 1) / video_info.total_frames, 1.0)
                        progress_bar.progress(progress)

                    detections = run_detections(
                        frame,
                        model,
                        conf_threshold=conf_threshold,
                        iou_threshold=iou_threshold,
                        use_sahi=use_sahi,
                    )

                    annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)

                    labels = []
                    if detections.class_id is not None and detections.confidence is not None:
                        for class_id, confidence in zip(detections.class_id, detections.confidence):
                            class_name = model.model.names[class_id]
                            if label_mode == "Label seulement":
                                labels.append(f"{class_name}")
                            elif label_mode == "Score seulement":
                                labels.append(f"{confidence:.2f}")
                            elif label_mode == "Label + score":
                                labels.append(f"{class_name} {confidence:.2f}")

                    if label_mode != "Aucun" and len(labels) > 0:
                        annotated_frame = label_annotator.annotate(
                            scene=annotated_frame,
                            detections=detections,
                            labels=labels,
                        )

                    sink.write_frame(frame=annotated_frame)

            # Re-encodage pour un MP4 plus compatible navigateur
            web_output_path = os.path.join(tempfile.gettempdir(), f"{base_name}_annotated_web.mp4")
            cap = cv2.VideoCapture(output_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(web_output_path, fourcc, fps, (width, height))

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                writer.write(frame)

            cap.release()
            writer.release()

            elapsed = time.time() - start_time
            progress_bar.progress(1.0)
            status_text.success(f"Traitement terminé en {elapsed:.1f} secondes !")

            # Notification éphémère
            st.toast(f"Vidéo traitée en {elapsed:.1f} secondes", icon="✅")

            st.subheader("Résultat")
            st.video(web_output_path)

            with open(web_output_path, "rb") as file:
                st.download_button(
                    label="Télécharger la vidéo annotée",
                    data=file,
                    file_name=f"{base_name}_annotated.mp4",
                    mime="video/mp4",
                )

        except Exception as e:
            st.error(f"Une erreur est survenue lors du traitement : {e}")
            
elif not model_path and uploaded_video:
    st.info("Veuillez sélectionner ou charger un modèle YOLO.")
elif model_path and not uploaded_video:
    st.info("Veuillez charger une vidéo.")

