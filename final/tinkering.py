# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 22:17:44 2018

@author: mattf
"""


(Xt_train, Xt_test, yt_train, yt_test) = train_test_split(Xtest, yt, test_size=0.2, random_state=40, stratify=yt)


forest = GradientBoostingClassifier(learning_rate=0.05, max_depth=10, subsample=0.8,
                                    max_features='auto', min_samples_split=25,
                                    n_estimators=75, loss='deviance')

forest.fit(Xt_train, yt_train)

colList = []
# Plot the feature importance ranks
importances = forest.feature_importances_
#std = np.std([tree.feature_importances_ for tree in forest.estimators_],axis=0)
indices = np.argsort(importances)[::-1]

feats = {} 
for feature, importance in zip(Xtest.columns, importances):
    feats[feature] = importance 

impp = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
#impp.sort_values(by='Gini-importance', ascending=False).plot(kind='bar', color='r', rot=90, yerr=std[indices], align='center')

imppS = impp.loc[impp['Gini-importance'] > 0.006] # set this value for aesthetics
imppS.sort_values(by='Gini-importance', ascending=False).plot(kind='bar', color='c', rot=45, align='center', figsize=(11,7))

plt.title('Feature importances', fontsize=24, fontweight='bold')
plt.xlabel('Feature Name', fontsize=18)
plt.ylabel('Relative Importance', fontsize=18)
plt.xticks(np.arange(15), colList , fontsize=14)

colList = ['# of Tweets', 'Canadian Pop', 'Dance Pop', 'Country', 'Song Length',
           'Rap', 'Tweet length', 'Minus stopwords', 'Loudness', 's112',
           'Acousticness', 's298', 'Favorites', 's37', 's133']

scaled = imputer.copy()

Xtest = scaled.drop(dropCol, axis=1)

dropCol = ['Acousticness', 'Album Type', 'Danceability', 'Energy', 'Explicit', 
           'Instrumentalness', 'Key', 'Length', 'Liveness', 'Loudness',
           'Speechiness', 'Track Number', 'Valence', 'set', 'chart', 'PeakPos',
           'Duration', 'impact', 'Track', 'Date', 'timeToChart', 'Album Type', 
           'Album', 'Genre', 'Label', 'Artist', 'Tempo']


dropList2 = ['set', 'chart', 'PeakPos', 'Duration', 'impact', 'Track', 
             'Date', 'timeToChart', 'Album Type', 'Album']









ytest = pd.Series()
yprob = pd.DataFrame(yt_probas)

for i in np.arange(len(yt_test)):
    if yprob.loc[i,0] > 0.3:
        ytest.loc[i] = 0
    else:
        ytest.loc[i] = 1



cm =  confusion_matrix(yt_test, ytest)
fig = plt.figure(figsize=(10, 10))
plot = plot_confusion_matrix(cm, classes=['Non-Hits','Hits'], normalize=False, title='Confusion matrix', cmap=plt.cm.Blues)
plt.show()


yt_probas = forest.predict_proba(Xt_test)
skplt.metrics.plot_roc_curve(yt_test, yt_probas)
plt.show()


p = 0.80
r = 0.76

2 * ((p*r)/(p+r))


import spotipy 
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy.util as util


username = 'MattForetich' # spotify developer account, account of the playlist
SPOTIPY_CLIENT_ID = '55e278fb84844a49a48911143295fafe' # an API generated when you register an app
SPOTIPY_CLIENT_SECRET = 'bec3776b658846509fae6e9b27fb2b5a' # as above
SPOTIPY_REDIRECT_URI  = 'http://localhost/' # the URL the user will be directed to after the authorization process
scope = 'playlist-modify-public' # the single permission we’ll need to create and modify a public playlist
token = util.prompt_for_user_token(username,scope,client_id=SPOTIPY_CLIENT_ID,client_secret=SPOTIPY_CLIENT_SECRET,redirect_uri=SPOTIPY_REDIRECT_URI)
client_credentials_manager = SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET)


spotify = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
spotify = spotipy.Spotify(auth=token)

results = spotify.user_playlists('1268126297')

spotify:user:1268126297:playlist:6ut8CWc9CUP1V9iLg4nPkL







