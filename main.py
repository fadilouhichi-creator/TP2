from pathlib import Path
import random
import sys

import cv2
import matplotlib
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog, QLabel

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
UI_FILE = BASE_DIR / "design.ui"
Ui_MainWindow, QtBaseClass = uic.loadUiType(str(UI_FILE))


class DesignWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.image_path = None
        self.gray_image = None
        self.display_labels = {}

        self._initialize_display_widgets()
        self._connect_signals()
        self.statusbar.showMessage("Chargez une image JPG, JPEG ou PNG pour commencer.")

    def _connect_signals(self):
        self.Browse.clicked.connect(self.get_image)
        self.Apply.clicked.connect(self.show_ImgHistEqualized)
        self.Validate_1.clicked.connect(self.show_ImgThresholding)
        self.Validate_2.clicked.connect(self.show_ImgFiltered)
        self.Validate_3.clicked.connect(self.show_ImgAugmented)

    def _initialize_display_widgets(self):
        placeholders = {
            self.OriginalImg: "Image originale",
            self.OriginalHist: "Histogramme original",
            self.EqualizedImg: "Image égalisée",
            self.EqualizedHist: "Histogramme égalisé",
            self.ThresholdingImg: "Image seuillée",
            self.FilteredImg: "Image filtrée",
            self.AugmentedImg: "Image transformée",
        }

        for widget, text in placeholders.items():
            layout = QtWidgets.QVBoxLayout(widget)
            layout.setContentsMargins(8, 8, 8, 8)
            layout.setAlignment(QtCore.Qt.AlignCenter)

            label = QLabel(text)
            label.setAlignment(QtCore.Qt.AlignCenter)
            label.setWordWrap(True)
            label.setStyleSheet("color: rgb(110, 110, 110); border: none;")
            layout.addWidget(label)
            self.display_labels[widget.objectName()] = label

    def _require_image(self):
        if self.gray_image is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Image manquante",
                "Veuillez d'abord sélectionner une image.",
            )
            return False
        return True

    def _clear_layout(self, widget):
        layout = widget.layout()
        if layout is None:
            return

        while layout.count():
            item = layout.takeAt(0)
            child_widget = item.widget()
            if child_widget is not None:
                child_widget.deleteLater()

    def _show_placeholder(self, widget, text):
        if widget.layout() is None:
            layout = QtWidgets.QVBoxLayout(widget)
            layout.setContentsMargins(8, 8, 8, 8)
            layout.setAlignment(QtCore.Qt.AlignCenter)

        self._clear_layout(widget)

        label = QLabel(text)
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.setWordWrap(True)
        label.setStyleSheet("color: rgb(110, 110, 110); border: none;")
        widget.layout().addWidget(label)
        self.display_labels[widget.objectName()] = label

    def _to_pixmap(self, image_source):
        if isinstance(image_source, (str, Path)):
            pixmap = QtGui.QPixmap(str(image_source))
            if pixmap.isNull():
                raise ValueError(f"Impossible de charger l'image: {image_source}")
            return pixmap

        if isinstance(image_source, np.ndarray):
            if image_source.ndim == 2:
                rgb_image = cv2.cvtColor(image_source, cv2.COLOR_GRAY2RGB)
            else:
                rgb_image = cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB)

            height, width, channel = rgb_image.shape
            bytes_per_line = channel * width
            qimage = QtGui.QImage(
                rgb_image.data,
                width,
                height,
                bytes_per_line,
                QtGui.QImage.Format_RGB888,
            ).copy()
            return QtGui.QPixmap.fromImage(qimage)

        raise TypeError("Type d'image non pris en charge.")

    def makeFigure(self, widget, image_source):
        if widget.layout() is None:
            layout = QtWidgets.QVBoxLayout(widget)
            layout.setContentsMargins(8, 8, 8, 8)
            layout.setAlignment(QtCore.Qt.AlignCenter)

        self._clear_layout(widget)

        pixmap = self._to_pixmap(image_source)
        target_size = widget.size()
        scaled = pixmap.scaled(
            max(1, target_size.width() - 16),
            max(1, target_size.height() - 16),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )

        label = QLabel()
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.setStyleSheet("border: none;")
        label.setPixmap(scaled)
        widget.layout().addWidget(label)
        self.display_labels[widget.objectName()] = label

    def _save_histogram(self, image, output_name, title):
        histogram = cv2.calcHist([image], [0], None, [256], [0, 256]).ravel()
        output_path = BASE_DIR / output_name

        plt.figure(figsize=(5, 3.6))
        plt.plot(histogram, color="black", linewidth=1.8)
        plt.title(title)
        plt.xlabel("Niveaux de gris")
        plt.ylabel("Nombre de pixels")
        plt.xlim([0, 255])
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

        return output_path

    def show_HistOriginal(self):
        if not self._require_image():
            return

        hist_path = self._save_histogram(
            self.gray_image,
            "Original_Histogram.png",
            "Histogramme de l'image originale",
        )
        self.makeFigure(self.OriginalHist, hist_path)

    def get_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Sélectionner une image",
            str(BASE_DIR),
            "Images (*.jpg *.jpeg *.png)",
        )

        if not file_path:
            return

        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            QtWidgets.QMessageBox.critical(
                self,
                "Erreur",
                "Le fichier sélectionné ne peut pas être chargé en tant qu'image.",
            )
            return

        self.image_path = Path(file_path)
        self.gray_image = image
        self.makeFigure(self.OriginalImg, self.gray_image)
        self.show_HistOriginal()

        self._show_placeholder(self.EqualizedImg, "Cliquez sur Appliquer pour l'égalisation.")
        self._show_placeholder(self.EqualizedHist, "Histogramme égalisé")
        self._show_placeholder(self.ThresholdingImg, "Choisissez une méthode de seuillage.")
        self._show_placeholder(self.FilteredImg, "Choisissez un filtre à appliquer.")
        self._show_placeholder(self.AugmentedImg, "Choisissez une opération géométrique.")

        self.statusbar.showMessage(f"Image chargée: {self.image_path.name}")

    def show_ImgHistEqualized(self):
        if not self._require_image():
            return

        equalized = cv2.equalizeHist(self.gray_image)
        equalized_image_path = BASE_DIR / "Equalized_Image.png"
        cv2.imwrite(str(equalized_image_path), equalized)
        self.makeFigure(self.EqualizedImg, equalized)

        equalized_hist_path = self._save_histogram(
            equalized,
            "Equalized_Histogram.png",
            "Histogramme de l'image égalisée",
        )
        self.makeFigure(self.EqualizedHist, equalized_hist_path)

        self.statusbar.showMessage("Égalisation d'histogramme appliquée avec succès.")

    def show_ImgThresholding(self):
        if not self._require_image():
            return

        if self.OtsuRadio.isChecked():
            _, result = cv2.threshold(
                self.gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        else:
            _, result = cv2.threshold(self.gray_image, 120, 255, cv2.THRESH_BINARY)

        threshold_path = BASE_DIR / "Thresholding_Image.png"
        cv2.imwrite(str(threshold_path), result)
        self.makeFigure(self.ThresholdingImg, result)

        self.statusbar.showMessage("Seuillage appliqué avec succès.")

    def show_ImgFiltered(self):
        if not self._require_image():
            return

        if self.MeanRadio.isChecked():
            result = cv2.blur(self.gray_image, (11, 11))
        elif self.MedianRadio.isChecked():
            result = cv2.medianBlur(self.gray_image, 13)
        else:
            result = cv2.GaussianBlur(self.gray_image, (15, 15), 10)

        filtered_path = BASE_DIR / "Filtered_Image.png"
        cv2.imwrite(str(filtered_path), result)
        self.makeFigure(self.FilteredImg, result)

        self.statusbar.showMessage("Filtrage appliqué avec succès.")

    def _zoom_and_crop_center(self, image):
        height, width = image.shape[:2]
        scale = random.uniform(1.5, 4.0)
        zoomed = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )

        zoomed_height, zoomed_width = zoomed.shape[:2]
        start_y = max(0, (zoomed_height - height) // 2)
        start_x = max(0, (zoomed_width - width) // 2)
        end_y = start_y + height
        end_x = start_x + width
        return zoomed[start_y:end_y, start_x:end_x]

    def show_ImgAugmented(self):
        if not self._require_image():
            return

        height, width = self.gray_image.shape[:2]

        if self.CroppingRadio.isChecked():
            result = self.gray_image[: height // 2, : width // 2]
        elif self.ZoomRadio.isChecked():
            result = self._zoom_and_crop_center(self.gray_image)
        else:
            center = (width // 2, height // 2)
            matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
            result = cv2.warpAffine(self.gray_image, matrix, (width, height))

        augmented_path = BASE_DIR / "Augmented_Image.png"
        cv2.imwrite(str(augmented_path), result)
        self.makeFigure(self.AugmentedImg, result)

        self.statusbar.showMessage("Opération géométrique appliquée avec succès.")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = DesignWindow()
    window.show()
    sys.exit(app.exec_())
