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

# Exact Node IDs mapped from your Flux.2 Klein 4B workflow_api.json
NODE_LOAD_IMAGE = "20"
NODE_PROMPT = "19"
NODE_SEED = "7"
NODE_SCHEDULER = "11"
NODE_SAVE_IMAGE = "1"
LORA_NODES = ["16", "17", "18"]

comfy_proc = None

def get_available_loras():
    lora_dir = os.path.join("ComfyUI", "models", "loras")
    if not os.path.exists(lora_dir):
        return ["None"]
    files = []
    for root, _, filenames in os.walk(lora_dir):
        for f in filenames:
            if f.endswith((".safetensors", ".pt", ".ckpt")):
                rel = os.path.relpath(os.path.join(root, f), lora_dir)
                files.append(rel)
    return ["None"] + sorted(files)

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
        raise gr.Error("Please upload an image first.")

    base = Image.fromarray(editor_data["background"]).convert("RGBA")
    layers = editor_data.get("layers", [])
    if not layers:
        raise gr.Error("Please paint a mask over the area you want to edit.")

    mask_alpha = np.array(layers[0])[:, :, 3] > 10
    if not np.any(mask_alpha):
        raise gr.Error("Please paint a mask over the area you want to edit.")

    rgba = np.array(base)
    rgba[:, :, 3] = np.where(mask_alpha, 0, 255)

    out_path = "input_masked.png"
    Image.fromarray(rgba).save(out_path)
    return out_path

def apply_optional_loras(wf, user_loras):
    # user_loras is a list of (lora_name, strength) tuples
    all_loras = [x for x in get_available_loras() if x != "None"]
    fallback_lora = all_loras[0] if all_loras else None

    for idx, node_id in enumerate(LORA_NODES):
        if node_id not in wf:
            continue
        inputs = wf[node_id]["inputs"]
        chosen_name, chosen_str = user_loras[idx] if idx < len(user_loras) else ("None", 0.0)

        if chosen_name != "None" and float(chosen_str) != 0.0:
            inputs["lora_name"] = chosen_name
            inputs["strength_model"] = float(chosen_str)
            inputs["strength_clip"] = float(chosen_str)
        else:
            # Zero out strength so ComfyUI skips applying the LoRA
            inputs["strength_model"] = 0.0
            inputs["strength_clip"] = 0.0
            if fallback_lora and not os.path.exists(os.path.join("ComfyUI", "models", "loras", inputs.get("lora_name", ""))):
                inputs["lora_name"] = fallback_lora

    # Remove unconnected dead nodes if present so missing files never error out
    if "14" in wf and "model" not in wf["14"].get("inputs", {}):
        del wf["14"]

def generate(editor_data, prompt, denoise, steps, l1_name, l1_str, l2_name, l2_str, progress=gr.Progress()):
    progress(0.05, desc="Preparing image & mask...")
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
    wf[NODE_SCHEDULER]["inputs"]["denoise"] = float(denoise)
    wf[NODE_SCHEDULER]["inputs"]["steps"] = int(steps)
    wf[NODE_SEED]["inputs"]["noise_seed"] = int.from_bytes(os.urandom(4), "big")

    apply_optional_loras(wf, [(l1_name, l1_str), (l2_name, l2_str), ("None", 0.0)])

    client_id = str(uuid.uuid4())
    ws = websocket.WebSocket()
    ws.connect(f"ws://{COMFY_HOST}/ws?clientId={client_id}")

    post_resp = requests.post(
        f"http://{COMFY_HOST}/prompt",
        json={"prompt": wf, "client_id": client_id},
    ).json()

    if "error" in post_resp or "node_errors" in post_resp and post_resp["node_errors"]:
        ws.close()
        raise gr.Error(f"ComfyUI Workflow Error: {json.dumps(post_resp)}")

    prompt_id = post_resp["prompt_id"]

    progress(0.15, desc="Running Flux.2 Klein 4B...")
    while True:
        msg = json.loads(ws.recv())
        if msg["type"] == "progress":
            val, total = msg["data"]["value"], msg["data"]["max"]
            progress(0.15 + 0.80 * (val / total), desc=f"Sampling step {val}/{total}")
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

lora_choices = get_available_loras()

with gr.Blocks(title="Flux.2 Klein 4B Inpaint Studio") as app:
    gr.Markdown("## Flux.2 Klein 4B Inpaint Studio")
    with gr.Row():
        with gr.Column(scale=1):
            editor = gr.ImageEditor(
                label="1. Upload Image & Paint Mask",
                type="numpy",
                brush=gr.Brush(colors=["#FFFFFF"], color_mode="fixed", default_size=45),
                eraser=gr.Eraser(default_size=45),
                height=520,
            )
            prompt_in = gr.Textbox(
                label="2. Edit Prompt",
                placeholder="Describe what should appear in the masked area...",
            )
            with gr.Row():
                denoise_in = gr.Slider(0.1, 1.0, value=0.85, step=0.05, label="Denoise Strength")
                steps_in = gr.Slider(4, 30, value=8, step=1, label="Steps")

            with gr.Accordion("Advanced: Optional LoRAs (Click to expand)", open=False):
                gr.Markdown("*Leave set to **None** for standard generation.*")
                with gr.Row():
                    lora1_name = gr.Dropdown(choices=lora_choices, value="None", label="LoRA Slot 1")
                    lora1_str = gr.Slider(-2.0, 2.0, value=1.0, step=0.05, label="Slot 1 Strength")
                with gr.Row():
                    lora2_name = gr.Dropdown(choices=lora_choices, value="None", label="LoRA Slot 2")
                    lora2_str = gr.Slider(-2.0, 2.0, value=1.0, step=0.05, label="Slot 2 Strength")

            submit_btn = gr.Button("Generate Edit", variant="primary", size="lg")

        with gr.Column(scale=1):
            output_img = gr.Image(label="Edited Result", height=520)

    submit_btn.click(
        generate,
        [editor, prompt_in, denoise_in, steps_in, lora1_name, lora1_str, lora2_name, lora2_str],
        [output_img],
    )

if __name__ == "__main__":
    boot_comfy()
    _, url, _ = app.launch(prevent_thread_lock=True, quiet=True)
    try:
        import webview
        webview.create_window("Flux.2 Klein 4B Inpaint Studio", url, width=1280, height=860)
        webview.start()
    except Exception:
        webbrowser.open(url)
        while True:
            time.sleep(1)
    finally:
        if comfy_proc:
            comfy_proc.terminate()