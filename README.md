# Thermal-Image Human Detection: Dataset Preparation

RGB person detectors fail in darkness and smoke, but people still show up clearly on thermal cameras. This project prepares a thermal-image dataset in YOLO format (one class, `person`) for training a detector.

The notebook `thermal_human_detection_preprocessing.ipynb` covers sampling, balancing, enhancement, augmentation and export. It does not train a model.

## Steps

1. **Data:** thermal frames with YOLO label files, one line per person.
2. **Exploration:** a random sample of 1,966 labeled images has 3.13 people per image on average (median 3, std 2.18, max 12).
3. **Balancing:** images with 1 to 3 people dominate, so the set was rebalanced by person count. Two methods were compared:
   - square-root bins with 150 images each
   - a cap of 300 images for every count from 1 to 12

   The capped version was kept. It has 2,282 images.
4. **Enhancement:** CLAHE with clip limit 2.0 and 8x8 tiles. The intensity of person regions, taken from the boxes, was compared with the background.
5. **Augmentation:** brightness and contrast, blur, gamma, CLAHE and hue/saturation, using Albumentations. These change pixel values only, so the boxes stay valid. One augmented copy is kept per image.
6. **Export:** the image and label pairs are uploaded to Roboflow and exported in COCO format.

| Random sample | After balancing |
|---|---|
| ![before](assets/count_distribution_random_sample.png) | ![after](assets/count_distribution_balanced.png) |

![CLAHE example](assets/clahe_example.png)

## Usage

- **Environment:** the notebook runs on Google Colab.
- **Data:** the dataset comes from Google Drive and is not part of this repository.
- **Roboflow key:** the export step reads it from the `ROBOFLOW_API_KEY` environment variable.
- **Config:** `data.yaml` is the YOLO dataset config.

## Team

Hrishikesh Muniraju, Trisha Annambhotla, Ameesha Talukdar.
