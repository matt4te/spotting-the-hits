# model for dash

import os
import sys
import numpy as np
import pandas as pd
import sklearn
import pickle

from sklearn.ensemble import GradientBoostingClassifier

# load the data and the model
newSongz = pd.read_csv('C:\\Users\\mattf\\Desktop\\insight\\THE_PROJECT\\final\\model\\X.csv')
newSongs = newSongz.copy(deep=True)

filename = 'C:\\Users\\mattf\\Desktop\\insight\\THE_PROJECT\\final\\model\\week3_bdt.sav'
bdt = pickle.load(open(filename, 'rb'))


# run the model
predictions= bdt.predict(newSongs)
probs = pd.DataFrame(bdt.predict_proba(newSongs))


# use the prediction probabilities to put together a dataframe of "hits"
probs.columns = ['Non-Hit', 'Hit']
sort = probs.sort_values('Hit', ascending=False)
sorta = sort.reset_index(drop=False)

hitList = pd.DataFrame()
for i in np.arange(25):
    idx = sorta.loc[i]['index']
    track = newSongz.loc[idx].Track
    artist = newSongz.loc[idx].Artist
    
    hitList = hitList.append(pd.Series({'Track':track, 'Artist':artist}),ignore_index=True)

