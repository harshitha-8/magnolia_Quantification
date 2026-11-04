# Third-party software

The original repository's MIT license applies to original application code.

- The detector is fine-tuned from Ultralytics YOLO11n. Ultralytics software and associated trained models are offered under **AGPL-3.0**, or a separate Ultralytics enterprise license. Redistribution of these weights does not make them MIT-licensed. See https://www.ultralytics.com/license and https://github.com/ultralytics/ultralytics/blob/main/LICENSE. Review the applicable terms before proprietary redistribution.
- ONNX Runtime Web 1.20.1 is bundled locally under the MIT license; see `vendor/ONNXRUNTIME-LICENSE`. Source: https://github.com/microsoft/onnxruntime/tree/v1.20.1.
- Training images and annotations were provided by the repository owner. The full source dataset is not redistributed. One recovered test crop and its label are included as a runnable example.

Reference architecture: https://github.com/samvadajainm/mildew-detector. This project follows its static browser-inference deployment pattern. It uses a newly fine-tuned Magnolia object detector; the mildew classifier and its disease scores are not used.

## Botanical profile photograph

- Local asset: `assets/southern-magnolia.jpg`.
- Title on source page: “Flowers & New Growth - June 4 - Halifax Co., NC.”
- Photographer: **Cathy DeWitt**.
- Source: [NC State Extension, Magnolia grandiflora](https://plants.ces.ncsu.edu/plants/magnolia-grandiflora/).
- Original image: https://eit-planttoolbox-prod.s3.amazonaws.com/media/images/Magnolia-grandiflour_MLOf9yWv0v1P.jpeg
- License: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
- Changes: original image bytes retained; the website scales and crops its displayed frame with CSS. This photo is educational profile imagery and is not training data or a model prediction.
