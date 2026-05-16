# OVLAFND: Open-Vocabulary Logic-Aware Fake News Detection

This repository contains the official PyTorch implementation of the OVLAFND framework. OVLAFND tackles multimodal multi-domain fake news detection by utilizing a continuous Open-Vocabulary Topic Manifold and a deep Logic Consistency Module (LCM).

## Requirements

The required packages are listed in `requirements.txt`. You can install the dependencies using pip:

```bash
pip install -r requirements.txt
```

*(Note: Please ensure you have compatible versions of PyTorch, CUDA, etc., installed based on your system setup.)*

## Data Preparation

To run this code, you need to prepare the datasets and their corresponding preprocessed loader files:

* Obtain the enhanced dataset file, which is located in the directory weibo_enhance/ weibo21_enhance/
* To obtain the preprocessing loader, please refer to the pretrained model section in [链接显示的文本](完整的网址 URL) to download the required models and directly place them in the directory src/
## Pretrained Models

To prepare the pretrained encoders:

* Download the required vision/language weights: `clip_cn_vit-b-16.pt` and `mae_pretrain_vit_base.pth`.
* Place them directly into the `src/` directory.
* Please refer to the pre trained model section of the [链接显示的文本](完整的网址 URL) on downloading the required model.
## Training

To train and evaluate the OVLAFND model from scratch, navigate to the source code directory:

```bash
cd src
python main.py
```
Switching between different datasets requires modifying the relevant dataset path configuration in the script.

## Log

Based on the pre written code, experiments were conducted. Please refer to log_weibo and log_weibo21 under src for relevant logs.
Note: Slight differences in results may be observed in different hardware configurations.

