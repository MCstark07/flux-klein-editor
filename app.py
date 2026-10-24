import os
import time
import json
import uuid
import subprocess
import urllib.request
import webbrowser
import websocket
import requests
import numpy as np
from PIL import Image
import gradio as gr

COMFY_HOST = "127.0.0.1:8188"

# workflow_api.json node IDs (Update these if your JSON uses different numbers)
NODE_LOAD_IMAGE = "10"
NODE_PROMPT = "6"
NODE_KSAMPLER = "3"
NODE_SAVE_IMAGE = "9"

comfy_proc = None

def boot_comfy():
    global comfy_proc
    try:
        urllib.request.urlopen(f"http://{COMFY_HOST}/system_stats", timeout=1)
        return
    except Exception:
        pass

    print("Starting ComfyUI backend...")
    py_bin = os.path.join("python_embeded", "python.exe")
    main_script = os.path.join("ComfyUI", "main.py")

    comfy_proc = subprocess.Popen(
        [py_bin, "-s", main_script, "--windows-standalone-build", "--disable-auto-launch"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )

    while True:
        try:
            urllib.request.urlopen(f"http://{COMFY_HOST}/system_stats", timeout=1)
            print("Backend ready.")
            break
        except Exception:
            time.sleep(1)

def build_mask_image(editor_data):
    if not editor_data or editor_data.get("background") is None:
        raise gr.Error("No image uploaded.")

    base = Image.fromarray(editor_data["background"]).convert("RGBA")
    layers = editor_data.get("layers", [])
    if not layers:
        raise gr.Error("Draw a mask over the area you want to edit first.")

    mask_alpha = np.array(layers[0])[:, :, 3] > 10
    rgba = np.array(base)
    rgba[:, :, 3] = np.where(mask_alpha, 0, 255)

    out_path = "input_masked.png"
    Image.fromarray(rgba).save(out_path)
    return out_path

def generate(editor_data, prompt, denoise, progress=gr.Progress()):
    progress(0.1, desc="Preparing mask...")
    img_path = build_mask_image(editor_data)

    with open(img_path, "rb") as f:
        resp = requests.post(
            f"http://{COMFY_HOST}/upload/image",
            files={"image": f},
            data={"overwrite": "true"},
        ).json()

    with open("workflow_api.json", "r", encoding="utf-8") as f:
        wf = json.load(f)

    wf[NODE_LOAD_IMAGE]["inputs"]["image"] = resp["name"]
    wf[NODE_PROMPT]["inputs"]["text"] = prompt

    sampler_in = wf[NODE_KSAMPLER]["inputs"]
    if "denoise" in sampler_in:
        sampler_in["denoise"] = float(denoise)

    seed_key = "noise_seed" if "noise_seed" in sampler_in else "seed"
    if seed_key in sampler_in:
        sampler_in[seed_key] = int.from_bytes(os.urandom(4), "big")

    client_id = str(uuid.uuid4())
    ws = websocket.WebSocket()
    ws.connect(f"ws://{COMFY_HOST}/ws?clientId={client_id}")

    prompt_id = requests.post(
        f"http://{COMFY_HOST}/prompt",
        json={"prompt": wf, "client_id": client_id},
    ).json()["prompt_id"]

    progress(0.2, desc="Sampling...")
    while True:
        msg = json.loads(ws.recv())
        if msg["type"] == "progress":
            val, total = msg["data"]["value"], msg["data"]["max"]
            progress(0.2 + 0.75 * (val / total), desc=f"Step {val}/{total}")
        elif msg["type"] == "executing":
            if msg["data"]["node"] is None and msg["data"]["prompt_id"] == prompt_id:
                break
    ws.close()

    history = requests.get(f"http://{COMFY_HOST}/history/{prompt_id}").json()[prompt_id]
    img_meta = history["outputs"][NODE_SAVE_IMAGE]["images"][0]

    query = f"filename={img_meta['filename']}&subfolder={img_meta['subfolder']}&type={img_meta['type']}"
    raw_img = urllib.request.urlopen(f"http://{COMFY_HOST}/view?{query}").read()

    with open("output.png", "wb") as f:
        f.write(raw_img)

    return Image.open("output.png")

with gr.Blocks(title="Flux Klein Inpaint") as app:
    gr.Markdown("### Flux.2 Klein 4B Inpaint Studio")
    with gr.Row():
        with gr.Column():
            editor = gr.ImageEditor(
                label="Input Image & Mask",
                type="numpy",
                brush=gr.Brush(colors=["#FFFFFF"], color_mode="fixed", default_size=40),
                eraser=gr.Eraser(default_size=40),
                height=500,
            )
            prompt_in = gr.Textbox(label="Prompt", placeholder="Describe the edit...")
            denoise_in = gr.Slider(0.1, 1.0, value=0.85, step=0.05, label="Denoise Strength")
            submit_btn = gr.Button("Generate", variant="primary")

        with gr.Column():
            output_img = gr.Image(label="Output", height=500)

    submit_btn.click(generate, [editor, prompt_in, denoise_in], [output_img])

if __name__ == "__main__":
    boot_comfy()
    _, url, _ = app.launch(prevent_thread_lock=True, quiet=True)
    try:
        import webview
        webview.create_window("Flux Klein Inpaint", url, width=1240, height=840)
        webview.start()
    except Exception:
        webbrowser.open(url)
        while True:
            time.sleep(1)
    finally:
        if comfy_proc:
            comfy_proc.terminate()