import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = r"D:\GitHub\RemoveBG\model\deeplabv3_mbv3_preprocessing.pth"
INPUT_SIZE = 512

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv6 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6)
        self.conv12 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12)
        self.conv18 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18)

        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.ReLU(inplace=True)
        )

        self.out_conv = nn.Conv2d(out_channels * 5, out_channels, 1)

    def forward(self, x):
        h, w = x.shape[2:]
        y1 = self.conv1(x)
        y2 = self.conv6(x)
        y3 = self.conv12(x)
        y4 = self.conv18(x)
        y5 = self.global_pool(x)
        y5 = F.interpolate(y5, size=(h, w), mode="bilinear", align_corners=False)
        y = torch.cat([y1, y2, y3, y4, y5], dim=1)
        return self.out_conv(y)

class DeepLabV3_MobileNetV3Large(nn.Module):
    def __init__(self):
        super().__init__()
        base = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        self.backbone = base.features
        self.aspp = ASPP(960, 256)
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.aspp(x)
        x = self.decoder(x)
        return F.interpolate(x, size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear", align_corners=False)

model = DeepLabV3_MobileNetV3Large()
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.to(DEVICE).eval()
print("Model load")

def preprocess(img):
    img = img.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0)
    mean = torch.tensor(IMAGENET_MEAN).view(1,3,1,1)
    std  = torch.tensor(IMAGENET_STD).view(1,3,1,1)
    return ((t - mean) / std).to(DEVICE)

def api_remove_bg(img):
    if img is None:
        return None, None, None, None

    with torch.no_grad():
        mask = (torch.sigmoid(model(preprocess(img))) > 0.5).float()

    mask_np = mask[0,0].cpu().numpy()
    mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
    mask_pil = mask_pil.resize(img.size, Image.NEAREST)

    img_rgba = img.convert("RGBA")
    img_np = np.array(img_rgba)
    alpha = np.where(np.array(mask_pil) > 128, 255, 0).astype(np.uint8)
    img_np[..., 3] = alpha

    fg_clean = Image.fromarray(img_np)

    return fg_clean.copy(), fg_clean.copy(), fg_clean.copy(), fg_clean.copy()


def api_merge_bg(fg_state, bg):
    if fg_state is None or bg is None:
        return None
    return Image.alpha_composite(bg.convert("RGBA").resize(fg_state.size),
                                 fg_state.convert("RGBA")).copy()


with gr.Blocks(title="RemoveBG - DeepLabV3") as demo:
    gr.HTML("<h2 style='text-align:center'>Ứng dụng Xóa Background</h2>")

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
