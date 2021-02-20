 ## Bengali Handwritten Grapheme Classification


#### About Problem
This is a multi-label multi-class classification. Where It have 3 level grapheme_root, vowel_diacritic, consonant_diacritic and 3 class in every label. One class have 168 type of images, another class have 11 type of images and the last class have 7 type of images. Where each image size is (137, 236). 




## Accuracy and Loss

|            |  Accuracy  |  Loss   |
| ---------- | :---------:| ------: |
|  Train     |  97.68     |  0.0123 |
|  Valid     |  96.12     |  0.0132 |




## Used Library

numpy  
pandas  
tqdm  
sklearn  
albumentations  
PIL  
torch  
pretrainedmodels  
iterstrat


 ## Used Special Classes  
 
 MultilabelStratifiedKFold  
 Custom_Dataset  
 albumentations.Compose for image augmentation  
 DataLoader  
 Resnet34 model
 Adam  
 BCEWithLogitsLoss  
 ReduceLROnPlateau  
 EarlyStopping  
 
  ## Models
  
  1. model_file.py
  2. model_file2.py
  3. model_file3.py  and a different model for Tensorflow 
  4. multi_layer_tensorflow_model.py
 
 
 
