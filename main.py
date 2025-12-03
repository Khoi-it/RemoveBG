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

with gr.Blocks(title="Đồ án DataMining RemoveBG") as demo:
    
    gr.HTML("<h1 style='text-align: center; width: 100%;'>Demo Ứng Dụng Tách Nền RemoveBG</h1>")
    gr.Markdown("_(Giao diện Front-end kết nối với Server)_")

    # Tab 1: Chuc nang tach nen
    with gr.Tab("Tách nền"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("Input")
                # Cho phep upload hoac chup anh
                input_img = gr.Image(sources=["upload", "webcam"], label="Ảnh đầu vào", type="pil")
                btn_process = gr.Button("Thực hiện", variant="primary")
            
            with gr.Column():
                gr.Markdown("Output")
                output_img = gr.Image(label="Kết quả", type="pil", format="png", interactive=False)

    # Tab 2: Chuc nang ghep nen
    with gr.Tab("Ghép nền"):
        with gr.Row():
            with gr.Column():
                # Input lay tu Tab 1
                img_fg = gr.Image(label="Ảnh đã tách", type="pil")
                img_bg = gr.Image(label="Ảnh nền mới", type="pil")
                btn_merge = gr.Button("Ghép ảnh")
            
            with gr.Column():
                output_merge = gr.Image(label="Ảnh sau khi ghép", type="pil")

    # Tab 3: Chinh sua nang cao
    with gr.Tab("Chỉnh sửa (Edit)"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("Điều chỉnh vị trí")
                
                # Load lai anh de thao tac
                edit_fg = gr.Image(label="Vật thể", type="pil", height=200)
                edit_bg = gr.Image(label="Nền", type="pil", height=200)
                
                slider_scale = gr.Slider(0.1, 2.0, value=1.0, step=0.1, label="Scale")
                slider_x = gr.Number(value=0, label="Toa do X")
                slider_y = gr.Number(value=0, label="Toa do Y")
                
                btn_apply = gr.Button("Cập nhật ảnh")
            
            with gr.Column():
                output_final = gr.Image(label="Kết quả hiển thị", type="pil")

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