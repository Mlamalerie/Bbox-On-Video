# Constantes globales pour l'application Bbox Stories

# Taille maximale (en pixels) du grand côté d'une image pour utiliser YOLO directement
# Au-delà, on bascule sur SAHI (découpage en tuiles).
MAX_DIRECT_SIDE: int = 1280

# Paramètres SAHI par défaut
# Taille des tuiles (width, height)
SAHI_SLICE_WH: tuple[int, int] = (512, 512)

# Chevauchement des tuiles (ratio width, ratio height)
SAHI_OVERLAP_WH: tuple[float, float] = (0.2, 0.2)

# Nombre de workers/threads pour le découpage SAHI
SAHI_WORKERS: int = 4
