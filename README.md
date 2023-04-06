# Captcha-OCR-Models
This is my pet project, the task of which I set as much as possible to complicate 
the OCR task by equating it with solving captcha.
I implement and train some models, then compare them on the task.

## Data

Dataset is collected using synthetic generator [trdg](https://github.com/Belval/TextRecognitionDataGenerator).
For captcha-augmentations, the `albumentations` library is used.

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

Models are broken down into: 
encoder-based (one forward) and seq2seq-based (2 forwards)

### Encoder-based

CTCLoss is used to train these models. \
Implementations are in `modeling.encoders`

Training notebook: `ctc_trainin.ipynb`

Metrics are in format: **CTCLoss, WER, CER**

#### CRNN cnn_v2_128_64seq_lstm_2l_100e
Clean Test: 0.09644, 0.24037, 0.0532 \
Captchas Test: 0.1445, 0.29487, 0.07253)

#### CNNBERT cnn_v2_128_64seq_bert_4h_3l_100e \
Clean Test: 0.08731, 0.50241, 0.12054 \
Captchas Test: 0.1443, 0.54003, 0.13632

#### ResNetRNN resnet18_128_lstm_2l_100e
Clean Test: 0.09526, 0.26832, 0.06179 \
Captchas Test: 0.15172, 0.31687, 0.08084

#### ResNetRNN resnet34_128_lstm_2l_100e
Clean Test: 0.0897, 0.22237, 0.05195 \
Captchas Test: 0.13997, 0.27212, 0.07055

#### ResNetRNN resnet50_256_lstm_2l_100e
Clean Test:  0.09287, 0.24386, 0.05388 \
Captchas Test: 0.13561, 0.29682, 0.07257

### Seq2Seq-based

Experiment notebook: `trocr_seq2seq_playground.ipynb`

TODO: TrOCR in work
