# Domain Composition and Attention for Unseen-Domain Generalizable Medical Image Segmentation
Pytorch implementation of our 'Domain Composition and Attention for Unseen-Domain Generalizable Medical Image Segmentation', which is accepted in MICCAI 2021.

# DataSet
Prepare the following files: `image`, `label`, `image_gan` generated by StyleGAN, and preprocessed Fourier Transform enhanced images `image_fft`.
├── image
├── label
├── image_gan
└── image_fft
# Training
1. Modify the configuration settings in settings.ini according to your requirements.
2. Run the training script:
```python
python train_multi_fundus.py
