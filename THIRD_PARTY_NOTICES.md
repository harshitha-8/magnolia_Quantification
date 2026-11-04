# Third-party software

The original repository's MIT license applies to original application code.

- The detector is fine-tuned from Ultralytics YOLO11n. Ultralytics software and associated trained models are offered under **AGPL-3.0**, or a separate Ultralytics enterprise license. Redistribution of these weights does not make them MIT-licensed. See https://www.ultralytics.com/license and https://github.com/ultralytics/ultralytics/blob/main/LICENSE. Review the applicable terms before proprietary redistribution.
- ONNX Runtime Web 1.20.1 is bundled locally under the MIT license; see `vendor/ONNXRUNTIME-LICENSE`. Source: https://github.com/microsoft/onnxruntime/tree/v1.20.1.
- Training images and annotations were provided by the repository owner. The full source dataset is not redistributed. One recovered test crop and its label are included as a runnable example.

Reference architecture: https://github.com/samvadajainm/mildew-detector. This project follows its static browser-inference deployment pattern. It uses a newly fine-tuned Magnolia object detector; the mildew classifier and its disease scores are not used.
