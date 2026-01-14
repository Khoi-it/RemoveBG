import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter
import numpy as np


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = r"D:\GitHub\RemoveBG\model\unet_preprocessing.pth"

INPUT_SIZE = 256

RAW_MEAN = [0.485, 0.456, 0.406]
RAW_STD  = [0.229, 0.224, 0.225]



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


model = UNet(in_channels=3, num_classes=1, base_c=32, bilinear=True)
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.to(DEVICE).eval()
print("UNet model loaded")

def preprocess(img: Image.Image) -> torch.Tensor:
    img = img.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)

    mean = torch.tensor(RAW_MEAN).view(1, 3, 1, 1)
    std  = torch.tensor(RAW_STD).view(1, 3, 1, 1)
    t = (t - mean) / std
    return t.to(DEVICE)


def api_remove_bg(img):
    if img is None:
        return None, None, None, None

    with torch.no_grad():
        x = preprocess(img)
        logits = model(x)
        prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

    gamma = 0.8
    prob = np.clip(prob, 0.0, 1.0) ** gamma

    alpha = Image.fromarray((prob * 255).astype(np.uint8)).resize(img.size, Image.BILINEAR)

    feather_radius = 1.2
    alpha = alpha.filter(ImageFilter.GaussianBlur(radius=feather_radius))

    a = np.array(alpha).astype(np.float32) / 255.0
    t_low, t_high = 0.12, 0.85
    a = np.clip((a - t_low) / (t_high - t_low + 1e-6), 0.0, 1.0)


    shrink_px = 1
    for _ in range(shrink_px):
        a_pad = np.pad(a, ((1,1),(1,1)), mode="edge")
        a = np.minimum.reduce([
            a_pad[0:-2,0:-2], a_pad[0:-2,1:-1], a_pad[0:-2,2:],
            a_pad[1:-1,0:-2], a_pad[1:-1,1:-1], a_pad[1:-1,2:],
            a_pad[2:  ,0:-2], a_pad[2:  ,1:-1], a_pad[2:  ,2:],
        ])

    alpha_final = (a * 255).astype(np.uint8)

    img_rgba = img.convert("RGBA")
    out = np.array(img_rgba)
    out[..., 3] = alpha_final
    fg_clean = Image.fromarray(out)

    return fg_clean.copy(), fg_clean.copy(), fg_clean.copy(), fg_clean.copy()


def api_merge_bg(fg_state, bg):
    if fg_state is None or bg is None:
        return None
    return Image.alpha_composite(
        bg.convert("RGBA").resize(fg_state.size),
        fg_state.convert("RGBA")
    ).copy()


with gr.Blocks(title="RemoveBG - UNet") as demo:
    gr.HTML("<h2 style='text-align:center'>Ứng dụng Xóa Background (U-Net)</h2>")

    fg_state = gr.State()

    with gr.Tab("Tách nền"):
        input_img = gr.Image(type="pil", label="Ảnh đầu vào")
        output_img = gr.Image(type="pil", label="Ảnh đã tách")
        btn_process = gr.Button("Tách nền", variant="primary")

    with gr.Tab("Ghép nền"):
        img_fg = gr.Image(type="pil", label="Foreground")
        img_bg = gr.Image(type="pil", label="Background")
        output_merge = gr.Image(type="pil", label="Ảnh ghép")
        btn_merge = gr.Button("Ghép ảnh")

    btn_process.click(api_remove_bg, input_img, [output_img, img_fg, img_fg, fg_state])
    btn_merge.click(api_merge_bg, [fg_state, img_bg], output_merge)

if __name__ == "__main__":
    demo.launch()
