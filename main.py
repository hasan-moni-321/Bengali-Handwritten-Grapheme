import numpy as np 
import pandas as pd 


import torch 
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau 

from pytorchtools import EarlyStopping

import model_file
import dataset
import train_evaluation


# Some Global variable
device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 50



def main(fold):

    # Reading All the Feature Dataset
    feature1 = pd.read_parquet("/home/hasan/Data Set/bengaliai_handwritten_gramheme/train_image_data_0.parquet").drop('image_id', axis=1)
    feature2 = pd.read_parquet("/home/hasan/Data Set/bengaliai_handwritten_gramheme/train_image_data_1.parquet").drop('image_id', axis=1)
    feature3 = pd.read_parquet("/home/hasan/Data Set/bengaliai_handwritten_gramheme/train_image_data_2.parquet").drop('image_id', axis=1)
    feature4 = pd.read_parquet("/home/hasan/Data Set/bengaliai_handwritten_gramheme/train_image_data_3.parquet").drop('image_id', axis=1)

    features = pd.concat([feature1, feature2, feature3, feature4], ignore_index=True)

    # Reading All the Label Dataset
    labels = pd.read_csv("/home/hasan/Desktop/Code to keep on Github/Bengali Handwritten Grapheme/train_fold.csv").drop(['image_id', 'grapheme'], axis=1)

    # Feature and label Dataset in one DataFrame
    dataframe = pd.concat([features, labels], axis=1)

    # Feature and Valid Data 
    feature_data = dataframe[dataframe.kfold != fold].drop('kfold', axis=1)
    valid_data = dataframe[dataframe.kfold == fold].drop('kfold', axis=1)

    train_feature = feature_data.drop(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic'], axis=1).values
    train_label = feature_data[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']] 

    valid_feature = valid_data.drop(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic'], axis=1).values
    valid_label = valid_data[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']]


    train_feature_label = dataset.Custom_Dataset(
                                                features=train_feature, 
                                                targets=train_label,
                                                train_data=True
                                                )
    valid_feature_label = dataset.Custom_Dataset(
                                                features=valid_feature,
                                                targets=valid_label,
                                                train_data=False
                                                )

    train_loader = DataLoader(
                            dataset = train_feature_label,
                            batch_size = 64,
                            shuffle = True 
                            num_workers = 4
                            )     
    valid_loader = DataLoader(
                            dataset = valid_feature_label,
                            batch_size = 8,
                            shuffle = False 
                            num_worker = 4
                            )   


    # Model, Scheduler, EarlyStopping 
    model = model_file.Resnet34()
    model.to(device) 

    optimizer = Adam(model.parameters(), lr=le-4) 
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.3, verbose=True) 
    early_stopping = EarlyStopping(patience=5, verbose=True) 


    # Training the model
    train_accuracy = []
    train_losses = []
    valid_accuracy = []
    valid_losses = []
    for epoch in range(epochs):
        train_acc, train_loss = train_evaluation.train(train_feature, train_loader, model, device, optimizer, scheduler)
        valid_acc, valid_loss = train_evaluation.evaluation(valid_feature, valid_loader, model, device) 
        train_accuracy.append(train_acc)
        train_losses.append(train_loss)
        valid_accuracy.append(valid_acc)
        valid_losses.append(valid_loss) 

        early_stopping(valid_loss, model) 
        if early_stopping.early_stop:
            break 

    print("Final train accuracy is :", np.mean(train_accuracy))
    print("Final train loss is :", np.mean(train_losses))
    print("Final valid accuracy is :", np.mean(valid_accuracy))
    print("Final valid loss is :", np.mean(valid_losses))



if __name__ == "__main__":
    for i in range(10):
        main(i)


