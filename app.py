import sys
import os
import numpy as np
from PIL import Image, ImageFilter

import torch
import torch.nn as nn
import torch.nn.functional as F

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QLabel,
    QFileDialog, QHBoxLayout, QVBoxLayout, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QSlider, QMessageBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = r"D:\GitHub\RemoveBG\model\unet_preprocessing.pth"

INPUT_SIZE = 256
RAW_MEAN = [0.485, 0.456, 0.406]
RAW_STD  = [0.229, 0.224, 0.225]

import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.net(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.conv = DoubleConv(in_ch, out_ch)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, base_c=32, bilinear=True):
        super().__init__()
        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)

        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)

        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear=bilinear)
        self.up2 = Up(base_c * 8,  base_c * 4 // factor, bilinear=bilinear)
        self.up3 = Up(base_c * 4,  base_c * 2 // factor, bilinear=bilinear)
        self.up4 = Up(base_c * 2,  base_c, bilinear=bilinear)

        self.out_conv = nn.Conv2d(base_c, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x,  x3)
        x = self.up3(x,  x2)
        x = self.up4(x,  x1)
        return self.out_conv(x)

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Không thấy model: {MODEL_PATH}")

    model = UNet(in_channels=3, num_classes=1, base_c=32, bilinear=True)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE).eval()
    return model



def preprocess_pil(img: Image.Image) -> torch.Tensor:
    img = img.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    mean = torch.tensor(RAW_MEAN).view(1, 3, 1, 1)
    std  = torch.tensor(RAW_STD).view(1, 3, 1, 1)
    t = (t - mean) / std
    return t.to(DEVICE)


def remove_bg_unet(model: nn.Module, img: Image.Image) -> Image.Image:
    with torch.no_grad():
        logits = model(preprocess_pil(img))
        prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

    gamma = 0.8
    prob = np.clip(prob, 0.0, 1.0) ** gamma

    alpha = Image.fromarray((prob * 255).astype(np.uint8)).resize(img.size, Image.BILINEAR)
    alpha = alpha.filter(ImageFilter.GaussianBlur(radius=1.2))

    a = np.array(alpha).astype(np.float32) / 255.0
    t_low, t_high = 0.12, 0.85
    a = np.clip((a - t_low) / (t_high - t_low + 1e-6), 0.0, 1.0)

    a_pad = np.pad(a, ((1, 1), (1, 1)), mode="edge")
    a = np.minimum.reduce([
        a_pad[0:-2,0:-2], a_pad[0:-2,1:-1], a_pad[0:-2,2:],
        a_pad[1:-1,0:-2], a_pad[1:-1,1:-1], a_pad[1:-1,2:],
        a_pad[2:  ,0:-2], a_pad[2:  ,1:-1], a_pad[2:  ,2:],
    ])

    alpha_final = (a * 255).astype(np.uint8)

    rgba = np.array(img.convert("RGBA"))
    rgba[..., 3] = alpha_final
    return Image.fromarray(rgba)



def pil_to_qpixmap(pil_img: Image.Image) -> QPixmap:
    pil_img = pil_img.convert("RGBA")
    arr = np.array(pil_img)
    h, w, _ = arr.shape
    qimg = QImage(arr.data, w, h, 4 * w, QImage.Format_RGBA8888)
    return QPixmap.fromImage(qimg.copy())


def qimage_to_pil(qimg: QImage) -> Image.Image:
    qimg = qimg.convertToFormat(QImage.Format_RGBA8888)
    w, h = qimg.width(), qimg.height()
    ptr = qimg.bits()
    ptr.setsize(h * w * 4)
    arr = np.frombuffer(ptr, np.uint8).reshape((h, w, 4))
    return Image.fromarray(arr, mode="RGBA")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RemoveBG U-Net - Drag & Drop Composer")

        self.model = None

        self.person_pil = None
        self.person_rgba = None
        self.bg_pil = None
        self.person_item = None
        self.bg_item = None

        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(self.view.renderHints())

        self.btn_load_person = QPushButton("Load ảnh người")
        self.btn_cut = QPushButton("Tách nền (U-Net)")
        self.btn_load_bg = QPushButton("Load ảnh nền")
        self.btn_export = QPushButton("Export PNG")

        self.btn_cut.setEnabled(False)
        self.btn_load_bg.setEnabled(False)
        self.btn_export.setEnabled(False)

        self.scale_label = QLabel("Scale: 100%")
        self.scale_slider = QSlider(Qt.Horizontal)
        self.scale_slider.setMinimum(25)
        self.scale_slider.setMaximum(200)
        self.scale_slider.setValue(100)
        self.scale_slider.setEnabled(False)

        controls = QHBoxLayout()
        controls.addWidget(self.btn_load_person)
        controls.addWidget(self.btn_cut)
        controls.addWidget(self.btn_load_bg)
        controls.addWidget(self.btn_export)

        scale_bar = QHBoxLayout()
        scale_bar.addWidget(self.scale_label)
        scale_bar.addWidget(self.scale_slider)

        root = QVBoxLayout()
        root.addLayout(controls)
        root.addLayout(scale_bar)
        root.addWidget(self.view)

        w = QWidget()
        w.setLayout(root)
        self.setCentralWidget(w)

        self.btn_load_person.clicked.connect(self.load_person)
        self.btn_cut.clicked.connect(self.cut_bg)
        self.btn_load_bg.clicked.connect(self.load_bg)
        self.btn_export.clicked.connect(self.export_png)
        self.scale_slider.valueChanged.connect(self.on_scale_change)

        try:
            self.model = load_model()
        except Exception as e:
            QMessageBox.critical(self, "Lỗi load model", str(e))

    def load_person(self):
        path, _ = QFileDialog.getOpenFileName(self, "Chọn ảnh người", "", "Images (*.png *.jpg *.jpeg *.webp)")
        if not path:
            return
        self.person_pil = Image.open(path).convert("RGB")

        self.scene.clear()
        self.bg_item = None
        self.person_item = QGraphicsPixmapItem(pil_to_qpixmap(self.person_pil))
        self.scene.addItem(self.person_item)
        self.person_item.setPos(0, 0)

        self.btn_cut.setEnabled(self.model is not None)
        self.btn_load_bg.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.scale_slider.setEnabled(False)
        self.scale_slider.setValue(100)
        self.scale_label.setText("Scale: 100%")

    def cut_bg(self):
        if self.model is None or self.person_pil is None:
            return

        self.setEnabled(False)
        try:
            self.person_rgba = remove_bg_unet(self.model, self.person_pil)
        finally:
            self.setEnabled(True)

        self.scene.clear()
        self.bg_item = None

        pm = pil_to_qpixmap(self.person_rgba)
        self.person_item = QGraphicsPixmapItem(pm)
        self.person_item.setFlags(
            QGraphicsPixmapItem.ItemIsMovable |
            QGraphicsPixmapItem.ItemIsSelectable |
            QGraphicsPixmapItem.ItemSendsGeometryChanges
        )
        self.scene.addItem(self.person_item)
        self.person_item.setPos(0, 0)

        self.btn_load_bg.setEnabled(True)
        self.btn_export.setEnabled(False)
        self.scale_slider.setEnabled(True)

    def load_bg(self):
        path, _ = QFileDialog.getOpenFileName(self, "Chọn ảnh nền", "", "Images (*.png *.jpg *.jpeg *.webp)")
        if not path:
            return
        self.bg_pil = Image.open(path).convert("RGB")

        self.scene.clear()

        bg_pm = pil_to_qpixmap(self.bg_pil)
        self.bg_item = QGraphicsPixmapItem(bg_pm)
        self.bg_item.setPos(0, 0)
        self.bg_item.setZValue(0)
        self.scene.addItem(self.bg_item)

        if self.person_rgba is None:
            QMessageBox.information(self, "Chưa tách nền", "Bạn cần tách nền trước.")
            return

        fg_pm = pil_to_qpixmap(self.person_rgba)
        self.person_item = QGraphicsPixmapItem(fg_pm)
        self.person_item.setFlags(
            QGraphicsPixmapItem.ItemIsMovable |
            QGraphicsPixmapItem.ItemIsSelectable |
            QGraphicsPixmapItem.ItemSendsGeometryChanges
        )
        self.person_item.setZValue(10)
        self.scene.addItem(self.person_item)

        bx, by = self.bg_pil.size
        fx, fy = self.person_rgba.size
        self.person_item.setPos((bx - fx) // 2, (by - fy) // 2)

        self.btn_export.setEnabled(True)
        self.scale_slider.setEnabled(True)
        self.scale_slider.setValue(100)

    def on_scale_change(self, v):
        self.scale_label.setText(f"Scale: {v}%")
        if self.person_item is None:
            return
        s = v / 100.0
        self.person_item.setScale(s)

    def export_png(self):
        if self.bg_pil is None or self.person_rgba is None or self.bg_item is None or self.person_item is None:
            QMessageBox.information(self, "Thiếu dữ liệu", "Cần có nền + foreground đã tách.")
            return

        bg = self.bg_pil.convert("RGBA")
        fg = self.person_rgba.convert("RGBA")

        pos = self.person_item.pos()
        scale = self.person_item.scale()

        if abs(scale - 1.0) > 1e-6:
            nw = max(1, int(fg.width * scale))
            nh = max(1, int(fg.height * scale))
            fg = fg.resize((nw, nh), Image.BILINEAR)

        x = int(round(pos.x()))
        y = int(round(pos.y()))

        out = bg.copy()
        out.alpha_composite(fg, (x, y))

        save_path, _ = QFileDialog.getSaveFileName(self, "Lưu ảnh PNG", "output.png", "PNG (*.png)")
        if not save_path:
            return
        if not save_path.lower().endswith(".png"):
            save_path += ".png"

        out.save(save_path)
        QMessageBox.information(self, "Xong", f"Đã lưu: {save_path}")


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(1100, 800)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
