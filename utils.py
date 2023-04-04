import json
import pickle
from collections import Counter
from glob import glob
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm.notebook import tqdm

from augmentation_utils import add_black_lines


class OCRTokenizer:
    unk_token, unk_token_id = '<UNK>', 0
    pad_token, pad_token_id = '<PAD>', 1

    def __init__(self, labels_file):
        with open(labels_file) as f:
            texts = ([l.strip().split(' ', maxsplit=1)[1] for l in f.readlines()])
        self.counter = Counter(''.join(texts))

        self.char2id = dict(zip(self.counter.keys(), range(2, len(self.counter) + 2)))
        self.char2id[self.unk_token] = self.unk_token_id
        self.char2id[self.pad_token] = self.pad_token_id

        self.id2char = {v: k for k, v in self.char2id.items()}

    def encode(self, text: str) -> torch.LongTensor:
        return torch.LongTensor([self.char2id.get(ch, self.unk_token_id) for ch in text])

    def decode(self, encoded: torch.LongTensor, drop_special: bool = False, to_text=False):
        if drop_special:
            tokens = [self.id2char[x.item()] for x in encoded.squeeze() if
                      x.item() != self.pad_token_id and x.item() != self.unk_token_id]
        else:
            tokens = [self.id2char[x.item()] for x in encoded.squeeze()]
        if to_text:
            return "".join(tokens)
        else:
            return tokens

    def decode_batch(self, encoded_batch: torch.LongTensor, drop_special: bool = False, to_text=False):
        return [self.decode(x, drop_special, to_text) for x in encoded_batch]

    def save_to(self, file_name: str):
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def from_pretrained(file_name: str):
        with open(file_name, 'rb') as f:
            return pickle.load(f)

    def __len__(self):
        return len(self.char2id)


class OCRDataset(Dataset):
    labels_file = 'labels.txt'

    train_transforms = A.Compose([
        A.RandomScale(scale_limit=(-0.4, 0.0), p=0.3),
        A.PadIfNeeded(min_height=64, min_width=30,
                      border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255)),
        A.Lambda(image=add_black_lines, p=0.2),
        A.GaussianBlur(p=0.5),
        A.RGBShift(p=0.4, r_shift_limit=90, g_shift_limit=90, b_shift_limit=90),
        A.Rotate(limit=3, p=0.15, crop_border=True),
        A.GridDistortion(p=0.4, normalized=True),
        A.ColorJitter(p=0.3),
        A.ImageCompression(quality_lower=30, p=0.1),
        A.ISONoise(p=0.1),
    ])

    basic_transforms = A.Compose([
        A.Resize(64, 256),
        # A.Normalize(),  # Вобще никакие нормализации не нужны за счет BatchNorm2d
        A.ToFloat(max_value=255.0),
        ToTensorV2(),
    ])

    def __init__(self, data_folder: str,
                 tokenizer: OCRTokenizer,
                 do_train_transform=False,
                 image_size=(64, 256)):
        self.tokenizer = tokenizer
        self.data_folder = Path(data_folder)
        self.do_train_transform = do_train_transform
        self.basic_transforms.transforms[0].height = image_size[0]
        self.basic_transforms.transforms[0].width = image_size[1]
        with open(self.data_folder / self.labels_file) as f:
            self.data = [l.strip().split(' ', maxsplit=1) for l in f.readlines()]  # [(img_path, text)]

    def __getitem__(self, idx):
        img_path = self.data_folder / self.data[idx][0]

        image = cv2.imread(img_path.as_posix())
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.do_train_transform:
            image = self.train_transforms(image=image)['image']
        image = self.basic_transforms(image=image)['image']

        text = self.data[idx][1]
        labels = self.tokenizer.encode(text)

        return {'image': image, 'labels': labels}

    def __len__(self):
        return len(self.data)


def collate_batch(batch: list, tokenizer: OCRTokenizer):
    return {
        'inputs': torch.cat([x['image'].unsqueeze(0) for x in batch], dim=0),
        'labels': pad_sequence([x['labels'] for x in batch], batch_first=True,
                               padding_value=tokenizer.pad_token_id).long(),
        'lengths': torch.LongTensor([len(x['labels']) for x in batch]),
    }


def save_experiment_info(model, losses_history: dict, file_name='experiment_info.json'):
    best_epoch_n = np.argmin(losses_history['eval'])
    result_dict = {
        'best_epoch': {
            'number': int(best_epoch_n),
            'train_loss': losses_history['train'][best_epoch_n],
            'eval_loss': losses_history['eval'][best_epoch_n]
        },
        'history': losses_history,
        'architecture': str(model)
    }
    with open(file_name, 'w') as outfile:
        json.dump(result_dict, outfile, indent=4)


def augment_dataset(path: str, percent: float):
    image_transforms = OCRDataset.train_transforms
    files = glob(path + '/*.jpg')

    for im_path in tqdm(files[:int(len(files) * percent)]):
        image = cv2.imread(im_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image_transforms(image=image)['image']
        cv2.imwrite(im_path, image)
