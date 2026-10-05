# Brain Tumor MRI Image Classification

Deep-learning classifier for brain MRI images (glioma, meningioma, pituitary tumor, no tumor): a custom CNN built from scratch, three ImageNet transfer-learning models (MobileNetV2, EfficientNetB0, ResNet50V2), and a Streamlit app for real-time predictions.

> Educational project. **Not a medical device.**

## Results (test split, no source-scan overlap with training)

![Model Comparison Results](results.png)

Deployed model: _fill_ (reason: _fill_).


Deployed model: _fill_ (reason: _fill_).

## Data and an important caveat

Dataset: Roboflow "Labeled MRI Brain Tumor Dataset v1" (CC BY 4.0), 2,443 images, 4 classes.
The provided train/valid/test folders are a Roboflow export in which augmented copies of the **same scan** appear in different splits (detected through the source ID in the file name), which inflates test scores. The notebook merges the data and re-splits it **grouped by source scan** (about 64 / 16 / 20 %), with an assertion that no scan is shared between splits.

## Project structure

```
Brain_Tumor_MRI_Classification.ipynb   # EDA, leakage check, training, evaluation, comparison
app.py                                 # Streamlit app
requirements.txt
models/
  best_model.h5                        # produced by the notebook
  class_names.json
```

## How to run

1. Open the notebook on Kaggle or Colab (GPU), set `DATA_ROOT`, run with `QUICK_TEST = True` once, then `False`.
2. Download `best_model.h5` and `class_names.json` from the notebook's `outputs/` folder into `models/`.
3. Start the app:

```
pip install -r requirements.txt
streamlit run app.py
```

Model files above 100 MB cannot be pushed to GitHub normally (use Git LFS or deploy a smaller model).

## Limitations

Small single-source dataset, no external validation, uncalibrated confidence scores. Do not use for any clinical decision.
