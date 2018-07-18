## the model 



### Set Up Shop ###

# general modules
import os
import sys
import numpy as np
import itertools
import pandas as pd
import sklearn
import pickle


# model metrics and visualizations 

import scikitplot as skplt
import matplotlib
import matplotlib.pyplot as plt

# import method to train/test split data 
from sklearn.model_selection import train_test_split

# import model evaluation metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, classification_report
from sklearn.metrics import confusion_matrix


# function to get model metrics 
def get_metrics(y_test, y_predicted):  
    # true positives / (true positives+false positives)
    precision = precision_score(y_test, y_predicted, pos_label=None, average='weighted')             
    # true positives / (true positives + false negatives)
    recall = recall_score(y_test, y_predicted, pos_label=None, average='weighted')
    # harmonic mean of precision and recall
    f1 = f1_score(y_test, y_predicted, pos_label=None, average='weighted')
    # true positives + true negatives/ total
    accuracy = accuracy_score(y_test, y_predicted)
    
    return accuracy, precision, recall, f1


# function to plot confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.winter):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=30)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="black" if cm[i, j] < thresh else "white", fontsize=40)
    
    plt.tight_layout()
    plt.ylabel('True label', fontsize=30)
    plt.xlabel('Predicted label', fontsize=30)

    return plt


# path
projdir = 'C:\\Users\\mattf\\Desktop\\insight\\THE_PROJECT\\final'
datadir = projdir + '\\data'

os.chdir(datadir)






### Load and Clean Data ###

# load the spotify data files
billSpot = pd.read_csv('spotify_billboard_merged.csv')
newRel = pd.read_csv('spotify_new_releases.csv')
rando = pd.read_csv('spotify_random_songs.csv')


# Load the twitter data files
newDir = datadir + '\\newTweets'
newDirFiles = os.listdir(newDir)
newRelT = pd.DataFrame()

for filename in newDirFiles:
    if filename.endswith('.csv'):
        file = pd.read_csv(os.path.join(newDir,filename))
        newRelT = newRelT.append(file)
 
print('Missing ', (len(newRel) - len(newRelT)) / len(newRel) * 100 , '%  new tweets')
       

ranDir = datadir + '\\randomTweets'
ranDirFiles = os.listdir(ranDir)
randoT = pd.DataFrame()

for filename in ranDirFiles:
    if filename.endswith('.csv'):
        file = pd.read_csv(os.path.join(ranDir,filename))
        randoT = randoT.append(file)
        
print('Missing ', (len(rando) - len(randoT)) / len(rando) * 100 , '%  random tweets')


billDir = datadir + '\\spotBillTweets'
billDirFiles = os.listdir(billDir)
billSpotT = pd.DataFrame()

for filename in billDirFiles:
    if filename.endswith('.csv'):
        file = pd.read_csv(os.path.join(billDir,filename))
        billSpotT = billSpotT.append(file)

print('Missing ', (len(billSpot) - len(billSpotT)) / len(billSpot) * 100 , '% billSpot tweets')


# clean the data a bit
drop_columns = ['Artist_x', 'Title', 'id']
billSpot = billSpot.drop(drop_columns,1)
billSpot.rename(columns={'Artist_y':'Artist'}, inplace=True)

# engineer two new features in the 'hits' dataset
billSpot['Date'] = pd.to_datetime(billSpot['Date'])
billSpot['DateS'] = pd.to_datetime(billSpot['DateS'])
billSpot['timeToChart'] = billSpot.Date - billSpot.DateS # WAY too many of these are negative
billSpot['impact'] = billSpot['Duration'] / billSpot['PeakPos']
billSpot.rename(columns = {'AlbumType':'Album Type', 'TrackNumber':'Track Number'}, inplace = True)

# filter out data with super weird dates
billSpot.timeToChart = billSpot.timeToChart.astype('timedelta64[D]')
billSpot = billSpot.loc[(billSpot['timeToChart'] < 365) & (billSpot['timeToChart'] > -365)]

# reduce milliseconds to seconds
billSpot['Length'] = billSpot['Length'] / 1000
newRel['Length'] = newRel['Length'] / 1000
rando['Length'] = rando['Length'] /1000


# merge spotify and twitter datasets
billSpot = billSpot.sort_values('PeakPos', ascending=False).drop_duplicates('Track').sort_index()
billSpotT = billSpotT.drop_duplicates(subset=['Track']) 
bst = billSpot.merge(billSpotT, how='left', on='Track') 
bst = bst.reset_index(drop=True)

newRel = newRel.drop_duplicates(subset=['Track']) 
newRelT = newRelT.drop_duplicates(subset=['Track']) 
nst = pd.merge(newRel, newRelT, how='left', on='Track')
nst = nst.reset_index(drop=True)

rando = rando.drop_duplicates(subset=['Track']) 
randoT = randoT.drop_duplicates(subset=['Track']) 
rst = pd.merge(rando, randoT, how='left', on='Track')
rst = rst.reset_index(drop=True)


# add chart-related columns to new / random data
nst['PeakPos'] = np.nan
nst['Duration'] = np.nan

rst['PeakPos'] = np.nan
rst['Duration'] = np.nan


# add new binary variable to train/test data
bst['chart'] = True
rst['chart'] = False


# combine them
bst = bst.drop('DateS',axis=1)
spotter = bst.append(rst, sort=True)
spotter['set'] = 'model'
nst['set'] = 'pred'
spotter = spotter.append(nst, sort=True)


# imputation for songs without twitter data
imputer = spotter.copy(deep=True).reset_index(drop=True)
zero_cols= ['Max Favorites', 'Max Retweets', 'Mean Favorites', 'Mean Length',
            'Mean Retweets', 'Mean Stops', 'Median Favorites', 'Median Length', 
            'Median Retweets', 'Median Stops', 'Relative#']
imputer[zero_cols] = imputer[zero_cols].fillna(0)

for i in np.arange(len(imputer)):
    if np.isnan(imputer.iloc[i]['s0']):
        imputer.iloc[i,35:335] = np.random.uniform(-0.2,0.2,300)


# one-hot encoding the genres
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
import ast

genrized = imputer.copy(deep=True)

genrized['Genre'] = genrized['Genre'].apply(ast.literal_eval)

genrized = genrized.join(pd.DataFrame(mlb.fit_transform(genrized.pop('Genre')),
                          columns=mlb.classes_,
                          index=genrized.index))



## inspect the data frame
#spotter.dtypes
#print('\nPercentange of Missing Values: \n')
#spotter.isnull().sum() / len(spotter) * 100


# fix types 
genrized['Album Type'] = genrized['Album Type'].astype('category')
genrized['Artist'] = genrized['Artist'].astype('category')
genrized['Date'] = genrized['Date'].astype('category')
genrized['Explicit'] = genrized['Explicit'].astype('category')
genrized['Key'] = genrized['Key'].astype('category')
genrized['chart'] = genrized['chart'].astype('category')
genrized['set'] = genrized['set'].astype('category')
genrized['Label'] = genrized['Label'].astype('category')






### Transform the data into different sets for regression / classification ###

# preserve original data, make a copy to transform
scaled = genrized.copy(deep=True)


# Import methods to transform data

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

from sklearn.preprocessing import OneHotEncoder
encf = OneHotEncoder(sparse=False)



# Standard Scaler for numberical variables
norm_cols = ['Length', 'Max Favorites', 'Max Retweets', 'Mean Favorites', 
             'Mean Length', 'Mean Retweets', 'Mean Stops', 'Median Favorites', 
             'Median Length', 'Median Retweets', 'Median Stops', 'Key',
             'Track Number', 'Acousticness', 'Danceability', 'Energy',
             'Instrumentalness', 'Liveness', 'Loudness', 'Relative#',
             'Speechiness', 'Tempo', 'Valence', 'Track Number']
scaled[norm_cols] = scaler.fit_transform(scaled[norm_cols])



### label encoder for categorical variables
#encoded = scaled.copy(deep=True)
#
#cat_cols = ['Album Type', 'Artist', 'Label', 'Album']
#
#for col in encoded.columns.values:
#    for col in cat_cols:
#        dle = encoded[col]
#        le.fit(dle.values)
#        encoded[col] = le.transform(encoded[col])
#
## make training data
#dropList = ['set', 'chart', 'PeakPos', 'Duration', 'impact', 
#            'Track', 'Date', 'timeToChart','Album']
#
#encodedm = encoded.loc[encoded['set'] == 'model']
#Xle = encodedm.drop(dropList, axis=1)
#
## make prediction data
#encodedp = encoded.loc[encoded['set'] == 'pred']
#Xlep = encodedp.drop(dropList, axis=1)
#
## plot sparsity
#fig, ax = plt.subplots()
#plt.spy(Xle)
#ax.get_xaxis().set_visible(False)
#plt.title('Final Data Frame')
#
#
#
### One hot encoder for non-ordinal categories
#hot = encoded.copy(deep=True)
#
#cat_cols2 = ['Album Type','Artist', 'Label','Album']
#        
#for col in cat_cols2:
#    dencf = hot[[col]]
#    encf.fit(dencf)
#    temp = encf.transform(hot[[col]])
#    temp = pd.DataFrame(temp,columns=[(col+'_'+str(i)) for i in dencf[col].value_counts().index])
#    temp = temp.set_index(hot.index.values)
#    hot = pd.concat([hot,temp],axis=1)
#
## make training data
#dropList2 = ['set', 'chart', 'PeakPos', 'Duration', 'impact', 'Track', 
#             'Date', 'timeToChart', 'Artist', 'Album Type', 'Label','Album']
#
#hotm = hot.loc[hot['set'] == 'model']
#Xoh = hotm.drop(dropList2, axis=1)
#
## make prediction data
#hotp = hot.loc[hot['set'] == 'pred']
#Xohp = hotp.drop(dropList2, axis=1)
#
## plot sparsity
#fig, ax = plt.subplots()
#plt.spy(Xoh)
#ax.xaxis.tick_bottom()
#plt.title('One-Hot Encoded Data Frame')




## One hot encode only album type
hot = scaled.copy(deep=True)

cat_cols = ['Album Type']

for col in hot.columns.values:
    for col in cat_cols:
        dle = hot[col]
        le.fit(dle.values)
        hot[col] = le.transform(hot[col])

cat_cols2 = ['Album Type']
        
for col in cat_cols2:
    dencf = hot[[col]]
    encf.fit(dencf)
    temp = encf.transform(hot[[col]])
    temp = pd.DataFrame(temp,columns=[(col+'_'+str(i)) for i in dencf[col].value_counts().index])
    temp = temp.set_index(hot.index.values)
    hot = pd.concat([hot,temp],axis=1)

# make training data
dropList2 = ['set', 'chart', 'PeakPos', 'Duration', 'impact', 'Track', 
             'Date', 'timeToChart', 'Album Type', 'Album']

hotm = hot.loc[hot['set'] == 'model']
Xoh = hotm.drop(dropList2, axis=1)

# make prediction data
hotp = hot.loc[hot['set'] == 'pred']
Xohp = hotp.drop(dropList2, axis=1)



## Probability encoder for non-ordinal categories
condo = hot.copy(deep=True)
true = condo[condo.chart==True]


# find proportion of chart belonging to each unique artist
artprobs = true.groupby('Artist').count() / len(true)
artprobs = artprobs.iloc[:,0]
artprobs = artprobs[artprobs > 0]

# map a dictionary of artist probability to a new dataframe column 
artDict = artprobs.to_dict()
condo['artProb'] = condo['Artist'].map(artDict)

## find proportion of chart belonging to each album type
#albprobs = true.groupby('Album Type').count() / len(true)
#albprobs = albprobs.iloc[:,0]
#albprobs = albprobs[albprobs > 0]
#
## map a dictionary of artist probability to a new dataframe column 
#albDict = albprobs.to_dict()
#condo['albProb'] = condo['Album Type'].map(albDict)

# find proportion of chart belonging to each label
labprobs = true.groupby('Label').count() / len(true)
labprobs = labprobs.iloc[:,0]
labprobs = labprobs[labprobs > 0]

# map a dictionary of artist probability to a new dataframe column 
labDict = labprobs.to_dict()
condo['labProb'] = condo['Label'].map(labDict)

# make training data
dropList3 = ['set', 'chart', 'PeakPos', 'Duration', 'impact', 'Track', 
             'Date', 'timeToChart', 'Artist', 'Label', 'Album']

condom = condo.loc[condo['set'] == 'model']
Xcp = condom.drop(dropList3, axis=1)
Xcp.fillna(0, inplace=True)

condop = condo.loc[condo['set'] == 'pred']
Xcpp = condop.drop(dropList3, axis=1)
Xcpp.fillna(0, inplace=True)

# plot sparsity
fig, ax = plt.subplots()
plt.spy(Xcp)
ax.xaxis.tick_bottom()
plt.xticks(np.arange(0, 601, step=300))
plt.xlabel('Number of Features')
plt.ylabel('Number of Songs')
plt.title('Proportionally Encoded Data Frame')






### Exploring the Data ###


# PCA 

from sklearn.decomposition import PCA, TruncatedSVD

def plot_LSA(data, labels, savepath="PCA_demo.csv", plot=True):
        lsa = TruncatedSVD(n_components=2)
        lsa.fit(data)
        lsa_scores = lsa.transform(data)
        color_mapper = {label:idx for idx,label in enumerate(set(labels))}
        color_column = [color_mapper[label] for label in labels]
        colors = ['orange','blue','blue']
        if plot:
            plt.scatter(lsa_scores[:,0], lsa_scores[:,1], s=8, alpha=.8, c=labels, cmap=matplotlib.colors.ListedColormap(colors))
            red_patch = mpatches.Patch(color='orange', label='Hits')
            green_patch = mpatches.Patch(color='blue', label='Non-Hits')
            plt.legend(handles=[red_patch, green_patch], prop={'size': 30})

            
#dropFeat = ['Album Type', 'Artist', 'Explicit', 'Key'] # all the non-numeric
#X2 = X.drop(dropFeat, axis=1)     

fig = plt.figure(figsize=(8, 8))
fig.suptitle('Dimensionality Reduction', size='xx-large')          
plot_LSA(Xcp, condom.chart) 
plt.show()



#  boxplots for tweet favorites / retweets

# favorites
fig, (ax1,ax2) = plt.subplots(1,2,sharey=True)

fig.suptitle('Tweet Favorites By Hit-Class')

ax1.boxplot(condom.loc[condom.chart==True]['Mean Favorites'])
ax1.set_ylabel('Number of Favorites')
ax1.set_xlabel('Hits')

ax2.boxplot(condom.loc[condom.chart==False]['Mean Favorites'])
ax2.set_xlabel('Non-Hits')
axes = plt.gca()

#axes.set_ylim([0,30])

# retweets
fig, (ax1,ax2) = plt.subplots(1,2,sharey=True)

fig.suptitle('Retweets By Hit-Class')

ax1.boxplot(condom.loc[condom.chart==True]['Mean Retweets'])
ax1.set_ylabel('Number of Retweets')
ax1.set_xlabel('Hits')

ax2.boxplot(condom.loc[condom.chart==False]['Mean Retweets'])
ax2.set_xlabel('Non-Hits')
axes = plt.gca()

#axes.set_ylim([0,30])



## artist contributions to hits

condomHa = condom[condom.chart==True]['artProb']
condomNa = condom[condom.chart==False]['artProb']
condomNa = condomNa.iloc[0:1378]
condomNa.fillna(0, inplace=True)

ca = condomHa.hist(label='Hits', align='right', bins=25, color='C1')
ca = condomNa.hist(ax=ca, label='Non-Hits', align='left', color='C0')
ca.legend()
ca.set(xlabel='Proportion of Hits by Single Artist', ylabel='Count', title='Artist Contribution to Hot-100 Charts')




## number of tweets collected for hits/non-hits

condomH = condom[condom.chart==True]['Relative#']
condomN = condom[condom.chart==False]['Relative#']
condomN = condomN.iloc[0:1378]

c = condomN.hist(cumulative=True, label='Non-Hits')
c = condomH.hist(ax=c, cumulative=True, label='Hits', align='right')
c.legend()
c.set_xlim(-2.5,0.5)
c.set(xlabel='Normalized Number of Tweets per Song', ylabel='Cumulative Density', title='Cumulative Density of Tweets by Class')

#c = condomH.plot.kde(label='Hits')
#c = condomN.plot.kde(ax=c,bw_method=0.3, label='Non-Hits')
#c.set_xlim(-3,1.5)
#c.set(xlabel='Normalized Number of Tweets', ylabel='Probability Density', title='Number of Tweets - Probability Density')
#c.legend()



## tweet length


imputerHl = imputer[imputer.chart==True]['Mean Favorites']
imputerHl = imputerHl[imputerHl < 10]
imputerNl = imputer[imputer.chart==False]['Mean Favorites']
imputerNl = imputerNl[imputerNl < 10]


cl = imputerNl.hist(label='Non-Hits', color='C0', align='right', bins=30)
cl = imputerHl.hist(ax=cl, label='Hits', color='C1', bins=30)
cl.legend()
cl.set_xlim(0,6)
cl.set(xlabel='Mean Tweet Favorites', ylabel='Count', title='Mean Tweet Favorites Per Song')

  






### Fitting A Classifier (Logistic Regression) ###

ylog = condom.chart.astype(int)
        
# split into train/test subsets
Xlog_train, Xlog_test, ylog_train, ylog_test = train_test_split(Xcp, ylog, test_size=0.2, random_state=40, stratify=ylog)


# fit the model
from sklearn.linear_model import LogisticRegression
lc = LogisticRegression(C=0.5, class_weight='balanced')
     
lc.fit(Xlog_train, ylog_train)


# make predictions for the test set
ylog_predicted = lc.predict(Xlog_test)


# assess model performance
accuracy, precision, recall, f1 = get_metrics(ylog_test, ylog_predicted)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))


# plot confusion matrix 
cm =  confusion_matrix(ylog_test, ylog_predicted)
fig = plt.figure(figsize=(10, 10))
plot = plot_confusion_matrix(cm, classes=['Non-Hits','Hits'], normalize=False, title='Confusion matrix')
plt.show()


# ROC curve
ylog_probas = lc.predict_proba(Xlog_test)
skplt.metrics.plot_roc(ylog_test, ylog_probas)
plt.show()


## save the model to disk
#filename = 'week3_logistic.sav'
#pickle.dump(lc, open(filename, 'wb'))






#### Decision trees for predicting hit classes ###

yt = condom.chart.astype(int)

# training set with CP encoding
(Xt_train, Xt_test, yt_train, yt_test) = train_test_split(Xcp, yt, test_size=0.2, random_state=40, stratify=yt)


# Fit some different decisiont rees
#from sklearn.ensemble import ExtraTreesClassifier
#from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

#forest = ExtraTreesClassifier(n_estimators=50)
#forest = RandomForestClassifier(n_estimators=50)
forest = GradientBoostingClassifier()

forest.fit(Xt_train, yt_train)


# make predictions
yt_predicted = forest.predict(Xt_test)

# get metrics and plot confusion matrix / ROC curve
accuracy, precision, recall, f1 = get_metrics(yt_test, yt_predicted)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

cm =  confusion_matrix(yt_test, yt_predicted)
fig = plt.figure(figsize=(10, 10))
plot = plot_confusion_matrix(cm, classes=['Hits','Non-Hits'], normalize=False, title='Confusion matrix')
plt.show()


yt_probas = forest.predict_proba(Xt_test)
skplt.metrics.plot_roc_curve(yt_test, yt_probas)
plt.show()


# Plot the feature importance ranks
importances = forest.feature_importances_
#std = np.std([tree.feature_importances_ for tree in forest.estimators_],axis=0)
indices = np.argsort(importances)[::-1]

feats = {} 
for feature, importance in zip(Xcp.columns, importances):
    feats[feature] = importance 

impp = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
#impp.sort_values(by='Gini-importance', ascending=False).plot(kind='bar', color='r', rot=90, yerr=std[indices], align='center')

imppS = impp.loc[impp['Gini-importance'] > 0.0077] # set this value for aesthetics
imppS.sort_values(by='Gini-importance', ascending=False).plot(kind='bar', color='r', rot=90, align='center')

plt.title('Feature importances')
plt.xlabel('Feature Name')
plt.ylabel('Relative Importance')
plt.xticks(fontsize=8)



## Cross-validate and grid search
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV


def modelfit(alg, X, y, t, performCV=True, printFeatureImportance=True, cv_folds=5):
    #Fit the algorithm on the data
    alg.fit(X, y)
        
    #Predict training set:
    y_predictions = alg.predict(X)
    y_predprob = alg.predict_proba(X)[:,1]
    
    #Perform cross-validation:
    if performCV:
        cv_score = cross_validation.cross_val_score(alg, X, y, cv=cv_folds, scoring='roc_auc')
    
    #Print model report:
    print("\nModel Report")
#    print("Accuracy : %.4g" % accuracy_score(y.values, y_predictions))
#    print("AUC Score (Train): %f" % roc_auc_score(y, y_predprob))
    
    if performCV:
        print("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
        
    #Print Feature Importance:
    if printFeatureImportance:
        feat_imp = pd.Series(alg.feature_importances_, list(X)).sort_values(ascending=False)
        feat_imp = feat_imp[feat_imp > t]
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')
 

#p_grid = [
#  {'learning_rate': [0.05, 0.1], # lower is better, but time consuming
#   'n_estimators': [25, 50, 75],
#   'max_depth': [3, 5, 10],
#   'min_samples_split': [5,10,25], # generally about 1% of the sample
#   'subsample': [0.8, 1],
#   'loss': ['deviance'],
#   'max_features': ['auto']},
#  {'learning_rate': [0.05, 0.1], 
#   'n_estimators': [25, 50, 75],
#   'max_depth': [3, 5, 10],
#   'min_samples_split': [5,10,225],
#   'subsample': [0.8, 1],
#   'loss': ['exponential'],
#   'max_features': ['auto']}
#  ]
#
#
#forest = GradientBoostingClassifier()
#
#gsearch = GridSearchCV(estimator = forest, param_grid = p_grid, 
#                        scoring='roc_auc',n_jobs=1,iid=False, cv=5, refit=True)
#
#gsearch.fit(Xt_train, yt_train)

#gsearch.best_params_, gsearch.best_score_
#best_estimator = {'learning_rate': 0.05,
#                  'loss': 'deviance',
#                  'max_depth': 10,
#                  'max_features': 'auto',
#                  'min_samples_split': 25,
#                  'n_estimators': 75,
#                  'subsample': 0.8}


forest = GradientBoostingClassifier(learning_rate=0.05, max_depth=10, subsample=0.8,
                                    max_features='auto', min_samples_split=25,
                                    n_estimators=75, loss='deviance')


modelfit(forest, Xt_train, yt_train, t=0.005)


forest.fit(Xt_train, yt_train)


## save the model to disk
#filename = 'week3_bdt.sav'
#pickle.dump(forest, open(filename, 'wb'))


# make predictions
yt_predicted = forest.predict(Xt_test)

# get metrics and plot confusion matrix / ROC curve
accuracy, precision, recall, f1 = get_metrics(yt_test, yt_predicted)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

cm =  confusion_matrix(yt_test, yt_predicted)
fig = plt.figure(figsize=(10, 10))
plot = plot_confusion_matrix(cm, classes=['Hits','Non-Hits'], normalize=False, title='Confusion matrix', cmap=plt.cm.Blues)
plt.show()


yt_probas = forest.predict_proba(Xt_test)
skplt.metrics.plot_roc(yt_test, yt_probas, title='ROC Curve', classes_to_plot=[1],plot_micro=False, plot_macro=False)
plt.show()




### Look at the highest probability predictions only
probSort = pd.DataFrame(yt_probas)
probSort['prediction'] = pd.Series(yt_predicted)
probSort['real'] = yt_test.values
probSort = probSort.sort_values(1, ascending=False)

# and make a confusion matrix for just the top 25
subSort = probSort.iloc[0:50,:]
subcm = confusion_matrix(subSort['real'], subSort['prediction'])
fig = plt.figure(figsize=(10, 10))
plot = plot_confusion_matrix(subcm, classes=['Non-Hits','Hits'], normalize=False, title='Confusion matrix')
plt.show()


billSpot = billSpot.sort_values('PeakPos', ascending=False).drop_duplicates('Track').sort_index()











#### Fit an elastic net regression for the impact / peakPos / duration variables
#yEn = condom.impact
#yEn = yEn[condom.chart==True]
#
#
#XEn = Xcp[condom.chart==True]
#
## split into train/test subsets
#
#XEn_train, XEn_test, yEn_train, yEn_test = train_test_split(XEn, yEn, test_size=0.2, random_state=40)
#
#    
## fit the model 
#from sklearn.linear_model import ElasticNet
#from sklearn.metrics import r2_score
#
## parameters of model fit
#alpha = 0.9
#l1_ratio = 0.7
#enet = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, normalize=True)
#
#enet.fit(XEn_train, yEn_train)
#
## evaluate predictions
#y_pred_En = enet.predict(XEn_test)
#r2_score_enet = r2_score(yEn_test, y_pred_En)
#print("r^2 on test data : %f" % r2_score_enet)
#
#plt.plot(enet.coef_, color='lightgreen', linewidth=2,
#         label='Elastic net coefficients')
#plt.title("Elastic Net R^2: %f"
#          % (r2_score_enet))
#plt.show()












### Running the Model on New Data ###

#filename = 'week3_logistic.sav'
#lcl = pickle.load(open(filename, 'rb'))

# run the model
predictions= lcl.predict(Xcpp)
prob = pd.DataFrame(lcl.predict_proba(Xcpp))


# use the prediction probabilities to put together a dataframe of "hits"
prob.columns = ['Non-Hit', 'Hit']
sort = prob.sort_values('Hit', ascending=False)
sorta = sort.reset_index(drop=False)

hitList = pd.DataFrame()
for i in np.arange(25):
    idx = sorta.loc[i]['index']
    track = condop.loc[idx].Track
    artist = condop.loc[idx].Artist
    
    hitList = hitList.append(pd.Series({'Track':track, 'Artist':artist}),ignore_index=True)
    
    
    
p = 0.71
r = 0.77
2 * ((p*r)/(p+r))    



curve = {'Model': ['Random Forest (OH)', 'Boosted Decision Tree (OH)', 'Logistic Regression (CP)',
                   'Random Forest (CP)', 'Logistic Regression (OH)', 'Boosted Decision Tree (CP)'],
    'F1 Score': [0.66, 0.71, 0.73, 0.74, 0.74, 0.82],
    'Precision': [0.64, 0.70, 0.71, 0.75, 0.71, 0.71],
    'Recall' : [0.69, 0.73, 0.76, 0.72, 0.77, 0.91]}

curveD = pd.DataFrame(data=curve)

p = curveD.plot(x='Model', y=['F1 Score', 'Precision', 'Recall'], kind='bar', title='Model Learning Curve')
#p = curveD.plot(ax=p, x='Model', y=['F1 Score'], kind='line', title='Model Learning Curve')

p.set(xlabel='Model', ylabel='Model Score')
p.set_ylim(0.5,1)
p.xaxis.label.set_fontsize(16)
p.yaxis.label.set_fontsize(16)
p.title.set_fontsize(20)
#p.get_yticklabels().set_fontsize(12)
plt.xticks(curveD.index, curveD['Model'], fontsize=10, rotation=15)




### full figure for the bonus slides
curveF = {'Model': ['Random Forest (OH)', 'Boosted Decision Tree (OH)', 'Logistic Regression (CP)',
                   'Random Forest (CP)', 'Logistic Regression (OH)', 'Boosted Decision Tree (CP)',
                   'Tweaked BDT'],
    'F1 Score': [0.66, 0.71, 0.73, 0.74, 0.74, 0.82, 0.78],
    'Precision': [0.64, 0.70, 0.71, 0.77, 0.71, 0.71, 0.80],
    'Recall' : [0.69, 0.73, 0.76, 0.72, 0.77, 0.91, 0.76]}


curveDF = pd.DataFrame(data=curveF)

pF = curveDF.plot(x='Model', y=['F1 Score', 'Precision', 'Recall'], kind='bar', title='Model Learning Curve')
#pF = curveDF.plot(ax=pF, x='Model', y=['F1 Score', 'Precision'], kind='line', title='Model Learning Curve')

pF.set(xlabel='Model', ylabel='Model Score')
pF.set_ylim(0.5,1)
pF.xaxis.label.set_fontsize(16)
pF.yaxis.label.set_fontsize(16)
pF.title.set_fontsize(20)
plt.xticks(curveDF.index, curveDF['Model'], fontsize=8, rotation=25)



## subset figure for in-demo comparison

curveS = {'Model': ['Random Forest (OH)','Logistic Regression (CP)','Boosted Decision Tree (CP)'],
    'F1 Score': [0.66,0.73,0.82],
    'Precision': [0.64,0.71,0.71],
    'Recall' : [0.69,0.76,0.91]}

curveSS = pd.DataFrame(data=curveS)

ps = curveSS.plot(x='Model', y=['F1 Score'], kind='bar', title='Model Learning Curve')
#p = curveD.plot(ax=p, x='Model', y=['F1 Score'], kind='line', title='Model Learning Curve')

ps.set(xlabel='Model', ylabel='Model Score')
ps.set_ylim(0.5,1)
ps.xaxis.label.set_fontsize(16)
ps.yaxis.label.set_fontsize(16)
ps.title.set_fontsize(20)
#p.get_yticklabels().set_fontsize(12)
plt.xticks(curveSS.index, curveSS['Model'], fontsize=10, rotation=15)



