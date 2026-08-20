"""Gradio MVP: upload a thermal `.mat`, see the defect mask, table and channels.

Run as a script (the directory name has a hyphen, so it is not an importable
package): `THERMAL_SEG_CKPT=... python web-frontend/app.py`.
"""
from __future__ import annotations

import gradio as gr
from runner import run

with gr.Blocks(title="Thermal Control") as demo:
    gr.Markdown("## Thermal Control — сегментация дефектов по ИК-видео")
    with gr.Row():
        with gr.Column():
            mat = gr.File(label=".mat видео", file_types=[".mat"])
            key = gr.Textbox(label="mat_key", placeholder="авто (один 3-D массив)")
            thr = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="порог")
            go = gr.Button("Предсказать", variant="primary")
        with gr.Column():
            overlay = gr.Image(label="маска на кадре")
            table = gr.Dataframe(headers=["region", "x", "y", "depth_mm"], label="дефекты")
    channels = gr.Gallery(label="каналы / prob / кропы", columns=4, height="auto")
    files = gr.File(label="mask.npy / depth.txt / meta.json")

    go.click(run, [mat, key, thr], [overlay, table, channels, files])

if __name__ == "__main__":
    demo.launch()
