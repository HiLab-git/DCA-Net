# Domain Composition and Attention for Unseen-Domain Generalizable Medical Image Segmentation
Pytorch implementation of our 'Domain Composition and Attention for Unseen-Domain Generalizable Medical Image Segmentation', which is accepted in MICCAI 2021.

## Usage

### 1. Training StyleGAN
StyleGAN was implemented by following (https://github.com/NVlabs/stylegan2-ada-pytorch), and it was trained for each domain respectively, as `image_gan`.

### 2. Prepare Dataset.
Prepare the following files: `image`, `label`, `image_gan`.
```
 ├── images
 ├── label
 └── image_gan
```
### 3. Train the model.
1) Modify the configuration settings in settings.ini according to your requirements.
2) Run the training script:
```python
python train_multi_fundus.py
```
### 4. Test the model.
1) Not using TTFA test model.
```python
python test_multi_fundus.py
```
2) Using TTFA test models.
```python
python test_multi_fundus_ttfa.py
```
