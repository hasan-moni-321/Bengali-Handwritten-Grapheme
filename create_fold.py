import pandas as pd 
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


df = pd.read_csv("/home/hasan/Data Set/bengaliai_handwritten_gramheme/train.csv")
df.loc[:, 'kfold'] = -1

df = df.sample(frac=1).reset_index(drop=True)

X = df['image_id'].values
y = df[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']]

mskf = MultilabelStratifiedKFold(n_splits=10)

for fold, (train_index, valid_index) in enumerate(mskf.split(X,y)):
    df.loc[valid_index, 'kfold'] = fold

print(df.kfold.value_counts())

df.to_csv("/home/hasan/Desktop/Code to keep on Github/Bengali Handwritten Grapheme/train_fold.csv", index=False)  

