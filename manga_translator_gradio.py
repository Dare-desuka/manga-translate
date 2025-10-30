"""
Simple Manga Translator untuk GitHub Codespaces (Gradio App)

Fungsi: Upload manga (gambar), otomatis OCR Jepang dan translate ke Indonesia atau Inggris, hasil langsung menempel di gambar.
Bisa dijalankan di GitHub Codespaces atau lokal.

Langkah singkat di Codespaces:
1. Buat repo baru di GitHub.
2. Tambahkan file ini (manga_translator_gradio.py).
3. Buat file requirements.txt dengan isi:
   gradio\neasyocr\ntransformers\ntorch\nsentencepiece\nopencv-python\npillow\nnumpy
4. Buka repo di Codespaces.
5. Jalankan: python manga_translator_gradio.py
6. Klik link port 7860 yang muncul.

Catatan:
- Tidak perlu GPU, tapi OCR bisa lambat di CPU.
- Tidak menghapus teks lama dengan sempurna.
- Untuk teks vertikal centang rotate.
"""

import os
import io
import zipfile
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import easyocr
from transformers import pipeline
import gradio as gr

reader = easyocr.Reader(['ja'], gpu=False)
translator_ja_en = pipeline('translation', model='Helsinki-NLP/opus-mt-ja-en')
translator_en_id = pipeline('translation', model='Helsinki-NLP/opus-mt-en-id')

def overlay_translations(image_pil, results, translations, font_path=None):
    img = image_pil.convert('RGBA')
    W, H = img.size
    draw = ImageDraw.Draw(img)

    try:
        if font_path and os.path.exists(font_path):
            font = ImageFont.truetype(font_path, 18)
        else:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    for (bbox, text, conf), trans in zip(results, translations):
        xs = [int(p[0]) for p in bbox]
        ys = [int(p[1]) for p in bbox]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)

        pad_x = max(6, int((x2-x1)*0.05))
        pad_y = max(4, int((y2-y1)*0.2))
        rect_x1 = max(0, x1 - pad_x)
        rect_y1 = max(0, y1 - pad_y)
        rect_x2 = min(W, x2 + pad_x)
        rect_y2 = min(H, y2 + pad_y)

        draw.rectangle([rect_x1, rect_y1, rect_x2, rect_y2], fill=(255,255,255,230))

        max_width = rect_x2 - rect_x1 - 4
        words = trans.split()
        lines = []
        cur = ''
        for w in words:
            test = (cur + ' ' + w).strip()
            wbox = draw.textbbox((0,0), test, font=font)
            if wbox[2] <= max_width:
                cur = test
            else:
                if cur:
                    lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)

        ytext = rect_y1 + 2
        for line in lines:
            draw.text((rect_x1+2, ytext), line, fill=(0,0,0,255), font=font)
            ytext += font.getsize(line)[1] + 2

    return img.convert('RGB')

def process_image(image_bytes, rotate=False, target_lang='id'):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    if rotate:
        image = image.rotate(90, expand=True)

    np_img = np.array(image)
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    results = reader.readtext(gray, detail=1)

    texts = [r[1] for r in results]
    translations = []
    for t in texts:
        if not t.strip():
            translations.append('')
            continue
        try:
            en = translator_ja_en(t, max_length=256)[0]['translation_text']
            if target_lang == 'en':
                translations.append(en)
            else:
                idt = translator_en_id(en, max_length=256)[0]['translation_text']
                translations.append(idt)
        except Exception:
            translations.append(t)

    out_img = overlay_translations(image, results, translations)

    buf = io.BytesIO()
    out_img.save(buf, format='JPEG', quality=90)
    buf.seek(0)
    return buf

with gr.Blocks() as demo:
    gr.Markdown("Manga Translator (untuk GitHub Codespaces)")
    with gr.Row():
        inp = gr.File(label='Upload image(s)', file_count='multiple', file_types=['.png', '.jpg', '.jpeg'])
        rotate = gr.Checkbox(label='Rotate 90° (untuk teks vertikal)', value=False)
    lang = gr.Dropdown(['id','en'], value='id', label='Target language')
    btn = gr.Button('Translate')
    gallery = gr.Gallery(label='Translated pages')
    out_zip = gr.File(label='Download ZIP hasil')

    def run(files, rotate, lang):
        if not files:
            return [], None
        out_files = []
        mem = io.BytesIO()
        z = zipfile.ZipFile(mem, 'w', zipfile.ZIP_DEFLATED)
        for i, f in enumerate(files):
            data = f.read()
            buf = process_image(data, rotate=rotate, target_lang=lang)
            name = f'translated_{i+1}.jpg'
            z.writestr(name, buf.getvalue())
            out_files.append(Image.open(io.BytesIO(buf.getvalue())))
        z.close()
        mem.seek(0)
        return out_files, mem

    btn.click(run, inputs=[inp, rotate, lang], outputs=[gallery, out_zip])

if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0', server_port=7860)

