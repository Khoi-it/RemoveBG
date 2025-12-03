import gradio as gr
from PIL import Image
import numpy as np

# --- Phan xu ly Logic (Mockup cho BE) ---

def api_remove_bg(img):
    # Ham nay gia lap viec goi API xoa nen
    if img is None:
        return None
    
    # In ra log de debug
    print("Call API: Remove BG")
    
    # TODO: Sau nay them code request.post o day
    return img

def api_merge_bg(fg, bg):
    if fg is None:
        return None
    
    print("Call API: Merge Background")
    
    # Neu chua co bg thi tra ve anh goc
    if bg is None:
        return fg
        
    return bg

def api_edit_image(fg, bg, x, y, scale):
    print(f"Call API Edit: x={x}, y={y}, scale={scale}")
    # Tra ve bg de test giao dien
    return bg


# --- Giao dien Gradio ---

with gr.Blocks(title="Do an RemoveBG") as demo:
    
    gr.HTML("<h1 style='text-align: center; width: 100%;'>Demo Ung dung Tach nen</h1>")
    gr.Markdown("Giao dien Front-end ket noi voi Server.")

    # Tab 1: Chuc nang tach nen
    with gr.Tab("Tach nen"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("Input")
                # Cho phep upload hoac chup anh
                input_img = gr.Image(sources=["upload", "webcam"], label="Anh dau vao", type="pil")
                btn_process = gr.Button("Thuc hien", variant="primary")
            
            with gr.Column():
                gr.Markdown("Output")
                output_img = gr.Image(label="Ket qua", type="pil", format="png", interactive=False)

    # Tab 2: Chuc nang ghep nen
    with gr.Tab("Ghep nen"):
        with gr.Row():
            with gr.Column():
                # Input lay tu Tab 1
                img_fg = gr.Image(label="Anh da tach", type="pil")
                img_bg = gr.Image(label="Anh nen moi", type="pil")
                btn_merge = gr.Button("Ghep anh")
            
            with gr.Column():
                output_merge = gr.Image(label="Anh sau khi ghep", type="pil")

    # Tab 3: Chinh sua nang cao
    with gr.Tab("Chinh sua (Edit)"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("Dieu chinh vi tri")
                
                # Load lai anh de thao tac
                edit_fg = gr.Image(label="Vat the", type="pil", height=200)
                edit_bg = gr.Image(label="Nen", type="pil", height=200)
                
                slider_scale = gr.Slider(0.1, 2.0, value=1.0, step=0.1, label="Scale")
                slider_x = gr.Number(value=0, label="Toa do X")
                slider_y = gr.Number(value=0, label="Toa do Y")
                
                btn_apply = gr.Button("Cap nhat")
            
            with gr.Column():
                output_final = gr.Image(label="Ket qua hien thi", type="pil")

    # --- Xu ly su kien ---
    
    # 1. Click nut tach nen
    btn_process.click(
        fn=api_remove_bg, 
        inputs=input_img, 
        outputs=[output_img, img_fg, edit_fg] 
    )

    # 2. Click nut ghep nen
    btn_merge.click(
        fn=api_merge_bg,
        inputs=[img_fg, img_bg],
        outputs=output_merge
    )

    # 3. Click nut edit
    btn_apply.click(
        fn=api_edit_image,
        inputs=[edit_fg, edit_bg, slider_x, slider_y, slider_scale],
        outputs=output_final
    )

if __name__ == "__main__":
    demo.launch()