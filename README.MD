
```markdown
OVLAFND: Open-Vocabulary Logic-Aware Fake News Detection

This repository contains the official PyTorch implementation of the OVLAFND framework. OVLAFND tackles multimodal multi-domain fake news detection by utilizing a continuous Open-Vocabulary Topic Manifold and a deep Logic Consistency Module (LCM).

🛠️ Requirements

The required packages are listed in `requirements.txt`. You can install the dependencies using pip:

```bash
pip install -r requirements.txt

📦 Data & Pretrained Models Preparation

To run this code, you need to manually download the datasets and pretrained weights:

1. Datasets:
* Obtain the `Weibo` and `Weibo-21` datasets.
* Place the raw files into the `/weibo/` and `/weibo21/` directories at the root level.


2. Pretrained Encoders:
* Download the required vision/language weights (e.g., `clip_cn_vit-b-16.pt` and `mae_pretrain_vit_base.pth`).
* Place them in the designated `src/` or `src/pretrained_model/` directory as required by the code.



🚀 Training & Evaluation

To train and evaluate the OVLAFND model from scratch:

1. Navigate to the source code directory:
```bash
cd src

```

2. Start training:
```bash
python main.py 

```


📊 Acknowledgements

Parts of the data preprocessing and foundational pipeline are inspired by previous multimodal fake news detection frameworks. We thank the open-source community for their invaluable contributions.

```

```
