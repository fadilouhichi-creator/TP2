import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image as RLImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[1]
DOCS_DIR = BASE_DIR / "docs"
ASSETS_DIR = DOCS_DIR / "assets"
REPORT_MD = DOCS_DIR / "Compte_rendu_TP2.md"
REPORT_PDF = DOCS_DIR / "Compte_rendu_TP2.pdf"
REPORT_AUTHOR = "Fedi Louhichi"
REPORT_SECTION = "S4"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from PyQt5 import QtWidgets

import main


def ensure_dirs():
    DOCS_DIR.mkdir(exist_ok=True)
    ASSETS_DIR.mkdir(exist_ok=True)


def create_demo_image():
    rng = np.random.default_rng(42)
    height, width = 420, 420

    gradient = np.tile(np.linspace(25, 210, width, dtype=np.float32), (height, 1))
    radial = np.zeros((height, width), dtype=np.float32)
    center = (width // 2, height // 2)
    for y in range(height):
        for x in range(width):
            distance = np.hypot(x - center[0], y - center[1])
            radial[y, x] = max(0, 60 - distance / 4)

    image = gradient + radial
    image = np.clip(image, 0, 255).astype(np.uint8)

    cv2.circle(image, (105, 110), 65, 240, -1)
    cv2.rectangle(image, (255, 70), (385, 180), 80, -1)
    cv2.line(image, (30, 350), (390, 250), 185, 6)
    cv2.putText(image, "CV1", (80, 330), cv2.FONT_HERSHEY_SIMPLEX, 2.1, 145, 5, cv2.LINE_AA)

    noise = rng.normal(0, 14, size=image.shape)
    image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    salt_mask = rng.random(image.shape) < 0.012
    pepper_mask = rng.random(image.shape) < 0.012
    image[salt_mask] = 255
    image[pepper_mask] = 0

    output_path = ASSETS_DIR / "image_demo.png"
    cv2.imwrite(str(output_path), image)
    return image, output_path


def save_histogram(image, output_path, title):
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256]).ravel()
    plt.figure(figsize=(5.2, 3.4))
    plt.plot(histogram, color="black", linewidth=1.8)
    plt.title(title)
    plt.xlabel("Niveaux de gris")
    plt.ylabel("Nombre de pixels")
    plt.xlim([0, 255])
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def save_gray_image(array, output_path):
    cv2.imwrite(str(output_path), array)
    return output_path


def array_to_pil(array):
    if array.ndim == 2:
        return Image.fromarray(array, mode="L").convert("RGB")
    rgb = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def fit_inside(image, max_width, max_height):
    width, height = image.size
    ratio = min(max_width / width, max_height / height)
    new_size = (max(1, int(width * ratio)), max(1, int(height * ratio)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def create_panel(items, output_path, columns=2, title=None):
    card_width = 520
    card_height = 340
    header_height = 56 if title else 0
    caption_height = 44
    rows = (len(items) + columns - 1) // columns
    width = columns * card_width + (columns + 1) * 20
    height = rows * card_height + (rows + 1) * 20 + header_height

    canvas = Image.new("RGB", (width, height), (248, 249, 251))
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.load_default()
    caption_font = ImageFont.load_default()

    if title:
        draw.rounded_rectangle((20, 20, width - 20, 20 + header_height - 10), radius=16, fill=(34, 40, 49))
        draw.text((38, 32), title, fill=(255, 255, 255), font=title_font)

    y_origin = header_height
    for index, (caption, image_path) in enumerate(items):
        row = index // columns
        col = index % columns
        x0 = 20 + col * (card_width + 20)
        y0 = y_origin + 20 + row * (card_height + 20)
        x1 = x0 + card_width
        y1 = y0 + card_height

        draw.rounded_rectangle((x0, y0, x1, y1), radius=18, fill=(255, 255, 255), outline=(210, 214, 220), width=2)
        draw.text((x0 + 16, y0 + 14), caption, fill=(30, 30, 30), font=caption_font)

        image = array_to_pil(cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED))
        preview = fit_inside(image, card_width - 32, card_height - caption_height - 30)
        px = x0 + (card_width - preview.size[0]) // 2
        py = y0 + caption_height + (card_height - caption_height - preview.size[1]) // 2
        canvas.paste(preview, (px, py))

    canvas.save(output_path)
    return output_path


def generate_results(image):
    original_hist = save_histogram(
        image,
        ASSETS_DIR / "hist_original.png",
        "Histogramme de l'image originale",
    )

    equalized = cv2.equalizeHist(image)
    equalized_path = save_gray_image(equalized, ASSETS_DIR / "equalized.png")
    equalized_hist = save_histogram(
        equalized,
        ASSETS_DIR / "hist_equalized.png",
        "Histogramme de l'image egalisee",
    )

    _, binary = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY)
    _, otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_path = save_gray_image(binary, ASSETS_DIR / "threshold_binary.png")
    otsu_path = save_gray_image(otsu, ASSETS_DIR / "threshold_otsu.png")

    mean_filtered = cv2.blur(image, (11, 11))
    gaussian_filtered = cv2.GaussianBlur(image, (15, 15), 10)
    median_filtered = cv2.medianBlur(image, 13)
    mean_path = save_gray_image(mean_filtered, ASSETS_DIR / "filter_mean.png")
    gaussian_path = save_gray_image(gaussian_filtered, ASSETS_DIR / "filter_gaussian.png")
    median_path = save_gray_image(median_filtered, ASSETS_DIR / "filter_median.png")

    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
    cropped = image[: height // 2, : width // 2]
    random.seed(42)
    scale = random.uniform(1.5, 4.0)
    zoomed = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    start_y = (zoomed.shape[0] - height) // 2
    start_x = (zoomed.shape[1] - width) // 2
    zoom_crop = zoomed[start_y:start_y + height, start_x:start_x + width]
    rotated_path = save_gray_image(rotated, ASSETS_DIR / "geometry_rotation.png")
    cropped_path = save_gray_image(cropped, ASSETS_DIR / "geometry_crop.png")
    zoom_path = save_gray_image(zoom_crop, ASSETS_DIR / "geometry_zoom.png")

    create_panel(
        [
            ("Image originale", ASSETS_DIR / "image_demo.png"),
            ("Histogramme original", original_hist),
            ("Image egalisee", equalized_path),
            ("Histogramme egalise", equalized_hist),
        ],
        ASSETS_DIR / "panel_histograms.png",
        columns=2,
        title="Resultats - Histogrammes et egalisation",
    )

    create_panel(
        [
            ("Image originale", ASSETS_DIR / "image_demo.png"),
            ("Seuillage binaire", binary_path),
            ("Seuillage d'Otsu", otsu_path),
        ],
        ASSETS_DIR / "panel_thresholds.png",
        columns=2,
        title="Resultats - Seuillage",
    )

    create_panel(
        [
            ("Image originale", ASSETS_DIR / "image_demo.png"),
            ("Filtre moyenneur", mean_path),
            ("Filtre gaussien", gaussian_path),
            ("Filtre median", median_path),
        ],
        ASSETS_DIR / "panel_filters.png",
        columns=2,
        title="Resultats - Filtrage",
    )

    create_panel(
        [
            ("Image originale", ASSETS_DIR / "image_demo.png"),
            ("Rotation 45 degres", rotated_path),
            ("Extraction quart haut-gauche", cropped_path),
            ("Zoom numerique", zoom_path),
        ],
        ASSETS_DIR / "panel_geometry.png",
        columns=2,
        title="Resultats - Operations geometriques",
    )

    return {
        "equalized": equalized,
        "binary": binary,
        "otsu": otsu,
        "mean": mean_filtered,
        "gaussian": gaussian_filtered,
        "median": median_filtered,
        "rotated": rotated,
        "cropped": cropped,
        "zoom": zoom_crop,
    }


def generate_interface_capture(image, image_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = main.DesignWindow()
    window.resize(1500, 980)
    window.image_path = image_path
    window.gray_image = image
    window.makeFigure(window.OriginalImg, image)
    window.show_HistOriginal()
    window.show_ImgHistEqualized()

    window.OtsuRadio.setChecked(True)
    window.show_ImgThresholding()

    window.MedianRadio.setChecked(True)
    window.show_ImgFiltered()

    random.seed(42)
    window.ZoomRadio.setChecked(True)
    window.show_ImgAugmented()

    window.show()
    app.processEvents()
    capture_path = ASSETS_DIR / "interface_capture.png"
    window.grab().save(str(capture_path))
    window.close()

    generated_files = [
        "Original_Histogram.png",
        "Equalized_Image.png",
        "Equalized_Histogram.png",
        "Thresholding_Image.png",
        "Filtered_Image.png",
        "Augmented_Image.png",
    ]
    for filename in generated_files:
        source = BASE_DIR / filename
        if source.exists():
            shutil.move(str(source), str(ASSETS_DIR / filename))

    return capture_path


def get_github_url():
    try:
        remote_url = subprocess.check_output(
            ["git", "remote", "get-url", "origin"],
            cwd=BASE_DIR,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if remote_url:
            return remote_url
    except Exception:
        pass
    return "Lien GitHub a ajouter apres publication du depot"


def write_markdown(github_url):
    content = f"""# Compte rendu detaille - TP2 Amelioration d'images

**Nom et prenom :** {REPORT_AUTHOR}  
**Section :** {REPORT_SECTION}  
**Matiere :** Computer Vision 1  
**Sujet :** Histogrammes, seuillage, filtrage et transformations geometriques

## 1. Objectif du travail

L'objectif de ce TP est de developper une application de traitement d'images sous **PyQt5** et **OpenCV**. L'application permet a l'utilisateur de charger une image en niveaux de gris, d'observer son histogramme, puis d'appliquer plusieurs operations d'amelioration et de transformation. Le projet demandait une interface graphique complete et une logique de traitement liee aux boutons, radios et zones d'affichage.

## 2. Environnement logiciel utilise

- Python 3.13
- PyQt5 pour l'interface graphique
- OpenCV pour le traitement d'images
- NumPy pour la manipulation des tableaux
- Matplotlib pour le trace des histogrammes

## 3. Structure du projet

Le dossier du projet contient les fichiers principaux suivants :

- `design.ui` : interface creee sous Qt Designer
- `design.py` : code Python genere a partir de l'interface
- `main.py` : logique complete de l'application
- `requirements.txt` : dependances du projet
- `docs/Compte_rendu_TP2.pdf` : version PDF du compte rendu

## 4. Description de l'interface graphique

L'interface est organisee en quatre grandes parties :

1. **Importation et histogrammes**
   Elle contient le bouton `Parcourir`, le bouton `Appliquer`, l'image originale, l'histogramme original, l'image egalisee et l'histogramme egalise.
2. **Seuillage**
   Cette partie permet de choisir entre le seuillage binaire simple et le seuillage d'Otsu.
3. **Filtrage**
   L'utilisateur peut choisir entre le filtre moyenneur, gaussien et median.
4. **Operations geometriques**
   Trois operations sont disponibles : rotation, extraction et agrandissement.

## 5. Implementation de la logique

### 5.1 Fonction `makeFigure()`

Cette fonction utilitaire a pour role d'afficher dynamiquement une image dans un widget de l'interface. Elle convertit soit un tableau NumPy, soit un chemin d'image, en `QPixmap`, puis adapte l'affichage a la taille disponible tout en conservant les proportions.

### 5.2 Fonction `get_image()`

Cette fonction ouvre une boite de dialogue pour selectionner une image de type `.jpg`, `.jpeg` ou `.png`. L'image choisie est chargee en niveaux de gris avec `cv2.imread(..., cv2.IMREAD_GRAYSCALE)`. Ensuite, elle est affichee dans le widget `OriginalImg`, puis son histogramme est calcule et affiche dans `OriginalHist`.

### 5.3 Fonction `show_HistOriginal()`

Cette fonction calcule l'histogramme de l'image originale grace a `cv2.calcHist`. Le resultat est dessine avec Matplotlib puis enregistre sous le nom `Original_Histogram.png`.

### 5.4 Fonction `show_ImgHistEqualized()`

L'egalisation d'histogramme est realisee avec `cv2.equalizeHist`. L'image produite est sauvegardee sous le nom `Equalized_Image.png`. Ensuite, un nouvel histogramme est calcule et sauvegarde sous `Equalized_Histogram.png`. Les deux resultats sont affiches dans l'interface.

### 5.5 Fonction `show_ImgThresholding()`

Cette fonction applique l'une des deux methodes suivantes :

- **Seuillage binaire** avec un seuil fixe `T = 120`
- **Seuillage d'Otsu** avec calcul automatique du seuil optimal

Le resultat est enregistre sous `Thresholding_Image.png` puis affiche dans le widget `ThresholdingImg`.

### 5.6 Fonction `show_ImgFiltered()`

Cette fonction applique le filtre selectionne :

- Filtre moyenneur avec noyau `11 x 11`
- Filtre gaussien avec noyau `15 x 15` et `sigma = 10`
- Filtre median avec taille `13`

L'image filtree est sauvegardee sous `Filtered_Image.png`.

### 5.7 Fonction `show_ImgAugmented()`

Trois transformations geometriques sont implementees :

- **Rotation** de `45` degres autour du centre de l'image
- **Extraction** du quart superieur gauche de l'image
- **Zoom numerique** avec facteur aleatoire entre `1.5` et `4.0`, suivi d'un recadrage central

L'image finale est enregistree sous `Augmented_Image.png`.

## 6. Captures des resultats obtenus

### 6.1 Interface apres execution

![Interface](assets/interface_capture.png)

### 6.2 Histogrammes et egalisation

![Histogrammes](assets/panel_histograms.png)

### 6.3 Resultats du seuillage

![Seuillage](assets/panel_thresholds.png)

### 6.4 Resultats du filtrage

![Filtrage](assets/panel_filters.png)

### 6.5 Resultats des operations geometriques

![Geometrie](assets/panel_geometry.png)

## 7. Verification et tests

Le projet a ete verifie de plusieurs manieres :

- verification de la syntaxe Python
- test de chargement de la fenetre PyQt5
- generation automatique des images de sortie
- production du present compte rendu avec les captures correspondantes

## 8. Conclusion

Ce TP a permis de mettre en pratique plusieurs notions fondamentales de l'amelioration d'images : histogrammes, equalisation, seuillage, filtrage spatial et transformations geometriques. L'utilisation combinee de PyQt5 et OpenCV a permis d'obtenir une application interactive, claire et directement exploitable dans PyCharm.

## Lien GitHub

{github_url}
"""
    REPORT_MD.write_text(content, encoding="utf-8")


def pdf_image(path, width_cm):
    image = RLImage(str(path))
    image.drawWidth = width_cm * cm
    image.drawHeight = image.drawWidth * 0.58
    return image


def build_pdf(github_url):
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="SectionTitle",
            parent=styles["Heading2"],
            fontSize=14,
            textColor=colors.HexColor("#1f2937"),
            spaceAfter=8,
            spaceBefore=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyTextCustom",
            parent=styles["BodyText"],
            fontSize=10.5,
            leading=14,
            spaceAfter=6,
        )
    )

    doc = SimpleDocTemplate(
        str(REPORT_PDF),
        pagesize=A4,
        leftMargin=1.6 * cm,
        rightMargin=1.6 * cm,
        topMargin=1.4 * cm,
        bottomMargin=1.4 * cm,
    )

    story = []
    story.append(Paragraph("Compte rendu detaille - TP2 Amelioration d'images", styles["Title"]))
    story.append(Spacer(1, 0.2 * cm))

    identity = Table(
        [
            ["Nom et prenom", REPORT_AUTHOR],
            ["Section", REPORT_SECTION],
            ["Matiere", "Computer Vision 1"],
            ["Technologies", "Python, PyQt5, OpenCV, NumPy, Matplotlib"],
        ],
        colWidths=[4.2 * cm, 11.6 * cm],
    )
    identity.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2ff")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8fafc")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(identity)
    story.append(Spacer(1, 0.35 * cm))

    sections = [
        (
            "1. Objectif du travail",
            "Ce TP consiste a concevoir une application graphique d'amelioration d'images. "
            "L'utilisateur doit pouvoir importer une image, analyser son histogramme, appliquer une "
            "egalisation, realiser un seuillage, filtrer l'image et lancer plusieurs operations geometriques.",
        ),
        (
            "2. Interface graphique",
            "L'interface a ete realisee avec Qt Designer dans le fichier design.ui. "
            "Elle regroupe les commandes de chargement, les radios de selection des traitements et "
            "les widgets de visualisation des images et histogrammes.",
        ),
        (
            "3. Logique de traitement",
            "Le fichier main.py contient une classe DesignWindow qui connecte chaque bouton a une "
            "fonction specialisee. Les fonctions principales sont makeFigure, get_image, show_HistOriginal, "
            "show_ImgHistEqualized, show_ImgThresholding, show_ImgFiltered et show_ImgAugmented.",
        ),
        (
            "4. Details des traitements",
            "L'egalisation repose sur cv2.equalizeHist. Le seuillage propose un mode binaire fixe et "
            "un mode Otsu. Le filtrage couvre les filtres moyenneur, gaussien et median. Enfin, les "
            "operations geometriques incluent une rotation a 45 degres, une extraction du quart superieur "
            "gauche et un zoom numerique avec recadrage central.",
        ),
    ]

    for title, body in sections:
        story.append(Paragraph(title, styles["SectionTitle"]))
        story.append(Paragraph(body, styles["BodyTextCustom"]))

    story.append(Paragraph("5. Capture de l'interface apres execution", styles["SectionTitle"]))
    story.append(pdf_image(ASSETS_DIR / "interface_capture.png", 17.2))
    story.append(PageBreak())

    story.append(Paragraph("6. Resultats experimentaux", styles["SectionTitle"]))
    story.append(Paragraph("Les figures suivantes illustrent les resultats obtenus sur une image de demonstration generee automatiquement.", styles["BodyTextCustom"]))
    story.append(pdf_image(ASSETS_DIR / "panel_histograms.png", 17.2))
    story.append(Spacer(1, 0.2 * cm))
    story.append(pdf_image(ASSETS_DIR / "panel_thresholds.png", 17.2))
    story.append(PageBreak())
    story.append(pdf_image(ASSETS_DIR / "panel_filters.png", 17.2))
    story.append(Spacer(1, 0.2 * cm))
    story.append(pdf_image(ASSETS_DIR / "panel_geometry.png", 17.2))

    story.append(Spacer(1, 0.25 * cm))
    story.append(Paragraph("7. Verification", styles["SectionTitle"]))
    story.append(
        Paragraph(
            "Le projet a ete valide par compilation Python, creation reelle des sorties image et "
            "demarrage correct de la fenetre PyQt5 dans un environnement de test hors ecran.",
            styles["BodyTextCustom"],
        )
    )

    story.append(Paragraph("8. Conclusion", styles["SectionTitle"]))
    story.append(
        Paragraph(
            "Cette application repond aux fonctionnalites demandees dans l'enonce. Elle offre une base "
            "solide pour illustrer les principales methodes d'amelioration d'images en vision par ordinateur.",
            styles["BodyTextCustom"],
        )
    )

    story.append(Paragraph("Lien GitHub", styles["SectionTitle"]))
    story.append(Paragraph(github_url.replace("&", "&amp;"), styles["BodyTextCustom"]))

    doc.build(story)


def main_generate():
    ensure_dirs()
    image, image_path = create_demo_image()
    generate_results(image)
    generate_interface_capture(image, image_path)
    github_url = get_github_url()
    write_markdown(github_url)
    build_pdf(github_url)


if __name__ == "__main__":
    main_generate()
