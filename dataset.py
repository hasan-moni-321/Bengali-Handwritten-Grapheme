
import albumentations
from PIL import Image


class Custom_Dataset:
    def __init__(self, features, targets, train_data=False):
        self.features = features
        self.grapheme_root = targets['grapheme_root']
        self.vowel_diacritic = targets['vowel_diacritic']
        self.consonant_diacritic = targets['consonant_diacritic']

        if train_data:
            self.aug = albumentations.Compose([
                                albumentations.Resize(137, 236, always_apply=True),
                                albumentations.ShiftScaleRotate(shift_limit=0.0625,
                                                                scale_limit=0.1,
                                                                rotate_limit=5,
                                                                p=0.9),
                                albumentations.RandomBrightnessContrast(always_apply=False),
                                albumentations.RandomRotate90(always_apply=False),
                                albumentations.HorizontalFlip(),
                                albumentations.VerticalFlip()
                                albumentations.Normalize(mean=(0.485, 0.456, 0.406), 
                                                         std=(0.229, 0.224, 0.225), 
                                                         always_apply=True)              
                                                ])

        else:
            self.aug = albumentations.Compose([
                                albumentations.Resize(img_height, img_width, always_apply=True),
                                albumentations.Normalize(mean=(0.485, 0.456, 0.406), 
                                                         std=(0.229, 0.224, 0.225),
                                                         always_apply=True) 
                                ])                       
            

    def __len__(self):
        return len(self.targets) 


    def __getitem__(self, idx):
        image = self.features[idx]
        image = image.reshape(137, 236).astype(float)
        image = Image.fromarray(image).convert("RGB")
        image = self.aug(image=np.array(image))['image']
        image = np.transpose(image, (2, 0, 1)).astype(np.float) 

        return {
            'image': torch.tensor(image, dtype=torch.float),
            'grapheme_root': torch.tensor(self.grapheme_root[idx], dtype=torch.long),
            'vowel_diacritic': torch.tensor(self.vowel_diacritic[idx], dtype=torch.long),
            'consonant_diacritic': torch.tensor(self.consonant_diacritic[idx], dtype=torch.long)  
        }

