import gradio as gr
import torch
import torch.nn as nn
from PIL import Image, ImageFilter
import numpy as np
import segmentation_models_pytorch as smp

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = r"D:\GitHub\RemoveBG\model\unet_preprocessing1.pth"
INPUT_SIZE = 256

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation=None
).to(DEVICE)

ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"], strict=True)
model.eval()
print("smp.Unet(resnet18) loaded")

def preprocess(img: Image.Image) -> torch.Tensor:
    img = img.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    t = (t - mean) / std
    return t.to(DEVICE)

def api_remove_bg(img):
    if img is None:
        return None, None, None, None

    with torch.no_grad():
        x = preprocess(img)
        logits = model(x)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

    gamma = 0.8
    prob = np.clip(prob, 0.0, 1.0) ** gamma

    alpha = Image.fromarray((prob * 255).astype(np.uint8)).resize(img.size, Image.BILINEAR)
    alpha = alpha.filter(ImageFilter.GaussianBlur(radius=1.2))

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
