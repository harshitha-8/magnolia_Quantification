# Handoff: GitHub Pages → Google Sites

The app follows the same static deployment pattern as the Mildew Detector. It needs no Python server, GPU, API key, account, or paid inference service in production. ONNX Runtime and the model are served with the website. Visitor photos are processed in browser memory.

## GitHub Pages

1. In this repository, open **Settings → Pages → Build and deployment → Source: GitHub Actions**.
2. The included `Deploy Magnolia detector` workflow runs on pushes to `main`. If necessary, run it manually under **Actions**.
3. Wait for a successful deployment. The expected address is:
   **https://harshitha-8.github.io/magnolia_Quantification/**
4. Open it, choose **Try a held-out example**, and choose **Count Magnolia plants**. Confirm the count, boxes and downloads work.

The workflow publishes only the static app, ONNX weights, runtime, example, and public audit reports. Training data, Python environments, Git metadata, and PyTorch weights are not published to Pages.

## Embed in Google Sites

1. Open your Google Site in the editor.
2. Choose **Insert → Embed → By URL**.
3. Paste **https://harshitha-8.github.io/magnolia_Quantification/**.
4. Resize the embed to full width and at least **1,100 px high**; narrow/mobile layouts may need additional height.
5. Preview on desktop and mobile, then publish your Google Site.

Alternatively use **Embed code**:

```html
<iframe
  src="https://harshitha-8.github.io/magnolia_Quantification/"
  title="Magnolia plant quantification"
  width="100%" height="1100" style="border:0"
  loading="lazy">
</iframe>
```

Google Sites may restrict downloads inside embeds. If a browser blocks a download, open the Pages app in its own tab. The app performs inference without SharedArrayBuffer or cross-origin isolation, so it can run inside a Google Sites iframe.

## Files your UI collaborator needs

- `models/magnolia_detector.onnx`: model weights.
- `models/model.json`: exact input/output and preprocessing contract.
- `detector-core.js`: box decoding, confidence filtering, NMS, coverage.
- `app.js`: image preprocessing, ONNX session, drawing and downloads.
- `vendor/`: pinned browser runtime and license; deploy these files together.

Keep all URLs relative to the app directory. The repository lives under a GitHub Pages subpath. The input is **not** the Mildew Detector's 224×224 NHWC raw RGB tensor. Magnolia requires 640×640 letterboxed **NCHW**, RGB float32 **divided by 255**. The output `[1,5,8400]` contains center-x, center-y, width, height, and Magnolia probability. There is no objectness channel and no extra sigmoid. Run confidence filtering and NMS after inference, then undo the padding and scaling.

## Local preview

```bash
python3 -m http.server 8000
```

Open http://localhost:8000. Opening `index.html` directly as a file will not work because the model and JavaScript modules are fetched over HTTP. The first model load may take a few seconds. A modern browser with WebAssembly support is required.
