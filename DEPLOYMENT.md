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

## Reference boxes and simplified labels

The app now draws only plant numbers, with no confidence percentages beside the boxes. The optional confidence filter is under **Advanced detection settings** and is not an accuracy control.

Deploy `references/annotations.json` with the app. It contains the provided visible-region annotations for 298 source files and hashes of the 298 originals plus 278 prepared crops. This enables an explicitly labeled **Supplied annotations (reference)** view for exact file matches. Reference display is separate from ONNX inference: `100%` means every supplied visible-region box is drawn. It must never be presented as 100% model accuracy. Reference exports carry `resultSource: provided_annotations`, null scores, and no model confidence threshold.

For a matched truncated original, the browser uses its audited recoverable top region, resized to 1280 pixels wide. Counts and box coverage refer to this working region; source dimensions and coordinate-space notes are included in JSON. A file with the same name but different contents is not matched. Unknown uploads use automatic model detections.

The repository includes the ONNX model, fine-tuned `.pt` checkpoint, model contract, all training and evaluation scripts, reports, browser runtime, reference labels, and deployment workflow. The full raw image dataset remains local. The model is **YOLO11n**, fine-tuned for 30 epochs; this UI update does not retrain or change the weights.

## Troubleshooting uploads and local use

- Use the permanent GitHub Pages URL above. A stopped localhost server or a `file://` page cannot load the application reliably.
- Wait for **Model ready**, choose a supported image under 40 MB and 60 megapixels, then select **Count Magnolia plants**.
- A filename or file-picker selection alone does not run inference. The count button starts it.
- If an original JPEG fails to decode in your browser, upload its recovered crop from local `data/magnolia/images/` instead. The source JPEGs are truncated.
- If a confidence change seems ineffective, expand Advanced settings and confirm Automatic model detections is selected. Reference boxes intentionally ignore model thresholds.
- For stale assets after deployment, reload the permanent URL. Inference runs locally, but the page, runtime, reference index, and model must finish downloading first.

## Local preview

```bash
python3 -m http.server 8000
```

Open http://localhost:8000. Opening `index.html` directly as a file will not work because the model and JavaScript modules are fetched over HTTP. The first model load may take a few seconds. A modern browser with WebAssembly support is required.
