# Captcha-OCR-Models
Different OCR models implementations in PyTorch \
(CNN-RNN, CNN-Transformer, ViT-RNN)

## Data

Dataset is collected using synthetic generator [trdg](https://github.com/Belval/TextRecognitionDataGenerator).

Generated images contain several different types of fonts, should consist of a maximum of two words and contain 
Latin letters and Arabic numerals, without special characters. Size is 64px in height and variable width.

Train part, consists of 20k images has no fixed augmentations, they are applied randomly during training.

Validation part is 1500 captcha pre-augmented images. \
Test part #1 is 5k pre-augmented images and Test part #2 is 5k zero-augmented images.

All information about dataset collection is presented in `data_generation.ipynb` and in `utils.OCRDataset` class

### Captcha Transforms
These transforms was designed to imitate real world captchas. \
Transformations include: rotations, geometric transformations, noises, color and brightness transformations,
as well as random curved lines crossing the image.

    train_transforms = A.Compose([
        A.Compose([  # Rescale transform
            A.RandomScale(scale_limit=(-0.3, -0.1), always_apply=True),
            A.PadIfNeeded(min_height=64, min_width=30,
                          border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255), always_apply=True),
            A.Rotate(limit=4, p=0.5, crop_border=True),
        ], p=0.4),
        A.Lambda(image=add_black_lines, p=0.3),  # Add lines to image
        A.GaussianBlur(blur_limit=(1, 7), p=0.5),
        A.OneOf([  # Geometric transforms
            A.GridDistortion(always_apply=True, num_steps=7, distort_limit=0.5,
                             border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255), normalized=True),
            A.OpticalDistortion(always_apply=True, border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255)),
            A.Perspective(scale=0.1, always_apply=True, fit_output=True, pad_val=(255, 255, 255))
        ], p=0.7),
        A.RGBShift(p=0.5, r_shift_limit=90, g_shift_limit=90, b_shift_limit=90),
        A.ISONoise(p=0.1),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.3, p=0.4),
        A.ImageCompression(quality_lower=30, p=0.2),
        A.GaussNoise(var_limit=70, p=0.3),
    ])

### Test captcha examples
![dataset_example_1.png](resources%2Fdataset_example_1.png)
![dataset_example_2.png](resources%2Fdataset_example_2.png)
![dataset_example_3.png](resources%2Fdataset_example_3.png)

## Models result

CRNN cnn_v2_128_64seq_lstm_2l_100e \
Clean Test: (0.06764, 0.19687, 0.04471)
Captchas Test: (0.1209, 0.25071, 0.05986)

CARNN cnn_v2_128_64seq_alstm_2h_2l_80e \
Clean Test: 0.08446, 0.19403, 0.04476 \
Captchas Test: 0.17378, 0.27079, 0.06931

VITRNN vit_128_512_6l_2h_65seq_lstm_2l_400e \
Clean Test: 0.52554, 0.85327, 0.38014 \
Captchas Test: 0.7041, 0.87289, 0.41252