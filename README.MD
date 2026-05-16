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

* Obtain the enhanced dataset file, which is located in the directory weibo_enhance/and weibo21_enhance/
* To obtain the preprocessing loader, please refer to the pretrained model section in [weibo loaders](https://drive.google.com/drive/folders/1OP8jI0bHskcuDyC5DoCHFy9eXo4GuK6W?usp=drive_link) and[weibo21 loaders](https://drive.google.com/drive/folders/15pgl50QiWtbTKK5NZIX0VWl0aeDm1-kC?usp=drive_link)to download the required models and directly place them in the directory src/
* To obtain the extracted LLM semantic feature data, please download it from [here](https://drive.google.com/drive/folders/1qtA5Tkgnron_4jodVfEQqVzIAykNBLe8?usp=drive_link) and extract it to the llm_topic_features/ directory
## Pretrained Models

To prepare the pretrained encoders:

* Download the required vision/language weights:  `clip_cn_vit-b-16.pt` and `mae_pretrain_vit_base.pth`.
* Place them directly into the `src/` directory.
* Please refer to the pre trained model section of the [clip_cn_vit-b-16.pt](https://drive.google.com/file/d/1S4LvHvYlDgofvwXPCz_V-JF4yuhEfLC8/view?usp=drive_link) and [mae_pretrain_vit_base.pth](https://drive.google.com/file/d/13Exyg1t7S4uR7iKbzzkjG5X10qpqajqX/view?usp=drive_link) downloading the required model.
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

