# OVLAFND: Open-Vocabulary Logic-Aware Fake News Detection

This repository contains the official PyTorch implementation of the OVLAFND framework. OVLAFND tackles multimodal multi-domain fake news detection by utilizing a continuous Open-Vocabulary Topic Manifold and a deep Logic Consistency Module (LCM).

## Requirements

The required packages are listed in `requirements.txt`. You can install the dependencies using pip:

```bash
pip install -r requirements.txt
Data PreparationTo run this code, you need to prepare the datasets and their corresponding preprocessed loader files:Obtain the enhanced dataset files (including images and preprocessed loaders like train_clip_loader.pkl, test_loader.pkl, etc.).Place them into the weibo_enhance/ and weibo21_enhance/ directories at the root level of the project.Pretrained ModelsTo prepare the pretrained encoders:Download the required vision/language weights: clip_cn_vit-b-16.pt and mae_pretrain_vit_base.pth.Place them directly into the src/ directory.TrainingTo train and evaluate the OVLAFND model from scratch, navigate to the source code directory:Bashcd src
(Note: Execute your training script here, e.g., python main.py)Log(You can put your training log descriptions or evaluation results here)
