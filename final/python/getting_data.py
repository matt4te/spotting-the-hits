
##### Acquire Date From Billboard Hot 100, Spotify, and Twitter #####
# Matt Foretich



# Import general modules
import datetime
import pandas as pd
import numpy as np
import sys
import os
import time
import string
import random
import pickle


# Set project and data directories
projdir = 'C:\\Users\\mattf\\Desktop\\insight\\THE_PROJECT\\final'
datadir = projdir + '\\data'

os.chdir(datadir)





### Get billboard hot-100 songs since some date

import billboard

dateStop = '2015-01-01' # date we want to stop getting charts at
tracks = pd.DataFrame() # empty dataframe to store all tracks
chart = billboard.ChartData('hot-100') # get most recent hot-100 chart

while pd.to_datetime(chart.date) > pd.to_datetime(dateStop): #for all charts before 2015
    chartDF = pd.DataFrame() # empty dataframe to store tracks on this chart
    date = chart.date # whats the date of this chart
    for s in np.arange(100): # for all tracks on the chart

        song = chart[s].title # get the song name
        artist = chart[s].artist # the artist name
        peak = chart[s].peakPos # the songs peak position, until now
        dur = chart[s].weeks # the duration the song has been on the charts
        
        # and append that info to a new dataframe for the current chart
        chartDF = chartDF.append(pd.Series({'Title':song, 'Artist':artist, 'PeakPos':peak, 'Duration':dur, 'Date':date}), ignore_index=True)
    
    # then append this chart to the master list of tracks
    tracks = tracks.append(chartDF)
    
    # and get the previous hot-100 chart
    chart = billboard.ChartData('hot-100', chart.previousDate)


# re-index the dataframe
trackz = tracks.reset_index(drop=True)

# set the date column to datetime
trackz['Date'] = pd.to_datetime(trackz['Date'])


# create summary table for each track
functionMap = {'PeakPos':['min'], 'Duration':['max'], 'Date':['min']}

summ = tracks.groupby('Title').agg(functionMap)


# find first instance of each track, and append into dataframe
unique_track_idx = {val: i for i, val in reversed(list(enumerate(trackz.Title)))}

tlist = pd.DataFrame()

for track in trackz.Title.unique():
    idx = unique_track_idx[track]
    entry = trackz.loc[idx]
    tlist = tlist.append(entry)
    
    
# drop vacuous columns
tlist = tlist.drop(['Duration', 'PeakPos', 'Date'], axis=1)

    
# merge the non-duplicate data frame with summarized data frame
finaltracks = pd.merge(tlist, summ, right_index=True, left_on='Title')
finaltracks = finaltracks.reset_index(drop=True)
                

## save the file
#finaltracks.to_csv('billboard_since_2015.csv', index=False)







### Get Spotify Information for: 
# (1) billboard hot-100 songs
# (2) new releases on spotify
# (3) random songs in the same date range as our hot-100 charts


# credentials for requesting authorization for spotify API
username = 'MattForetich' # spotify developer account, account of the playlist
SPOTIPY_CLIENT_ID = '55e278fb84844a49a48911143295fafe' # an API generated when you register an app
SPOTIPY_CLIENT_SECRET = 'bec3776b658846509fae6e9b27fb2b5a' # as above
SPOTIPY_REDIRECT_URI  = 'http://localhost/' # the URL the user will be directed to after the authorization process
scope = 'playlist-modify-public' # the single permission we’ll need to create and modify a public playlist



## client credential authorization
#import spotipy
#from spotipy.oauth2 import SpotifyClientCredentials
#
#client_credentials_manager = SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET)
#sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)



import spotipy
import spotipy.util as util


# authorization code flow token
token = util.prompt_for_user_token(username,scope,client_id=SPOTIPY_CLIENT_ID,client_secret=SPOTIPY_CLIENT_SECRET,redirect_uri=SPOTIPY_REDIRECT_URI)


## (1) billboard hot-100 songs

# clean up column titles in billboard data
bill = finaltracks      
bill.columns = ['Artist', 'Title', 'PeakPos', 'Duration', 'Date']


# remove artist words that mess up billboard / spotify matches
mismatch = ['Featuring', '&', '+', 'Or', 'x', 'X', 'With']
bill['Artist'] = bill['Artist'].apply(lambda x: " ".join(x for x in x.split() if x not in mismatch))


# create this column to match entries with spotify track info later
bill['id'] = np.arange(len(bill))


# empty dataframe to store requested data
spotDF = pd.DataFrame()

# for each entry in the billboard hot-100
for i, r in bill.iterrows():
    try:
        
        # get the song information
        title = bill.loc[i]['Title']
        artist = bill.loc[i]['Artist'] 
        billid = bill.loc[i]['id']
        
        # search for that song name / artist on spotify
        search_str = title + ' ' + artist 
        sp = spotipy.Spotify(auth=token)
        result = sp.search(search_str)
        
        # pull out the unique spotify track ID from the returned structure 
        spotify_id = result['tracks']['items'][0]['uri']
    
    # or report that the song can not be found on spotify
    except IndexError:
        print('No track information for ', title, ' by ', artist)
        continue
    
    # if the track is found, use the unique spotify ID to get information on the track
    urn = spotify_id 
    sp = spotipy.Spotify(auth=token)
    track = sp.track(urn)
    features = sp.audio_features(urn)
    
    name = track['name']
    album_type = track['album']['album_type']
    length = track['duration_ms']
    explicit = track['explicit']
    track_num = track['track_number']
    artist = track['artists'][0]['name']
    artID = track['artists'][0]['uri']
    date = track['album']['release_date']
    alb = track['album']['uri']
    
    da = features[0]['danceability']
    en = features[0]['energy']
    key = features[0]['key']
    vol = features[0]['loudness']
    sp = features[0]['speechiness']
    acoust = features[0]['acousticness']
    inst = features[0]['instrumentalness']
    live = features[0]['liveness']
    val = features[0]['valence']
    temp = features[0]['tempo']
    
    # also search for the album and artist info
    ap = spotipy.Spotify(auth=token) # new spotipy instance for album lookup
    albu = ap.album(alb)
    albName = albu['name']
    label = albu['label']
    arp = spotipy.Spotify(auth=token) # new spotipy instance for artist lookup
    art = arp.artist(artID)
    genre = art['genres']

    
    # store the track info 
    spotDF = spotDF.append(pd.Series({'Track':name, 'Album Type':album_type, 
                                      'Length':length, 'Track Number':track_num, 
                                      'Explicit':explicit, 'Danceability':da, 
                                      'Energy':en, 'Key':key, 'Loudness':vol, 
                                      'Speechiness':sp, 'Acousticness':acoust, 
                                      'Instrumentalness':inst, 'Liveness':live, 
                                      'Valence':val, 'Tempo':temp, 'id':billid, 
                                      'Artist':artist, 'DateS':date, 'Album':albName,
                                      'Label':label, 'Genre':genre}), ignore_index=True)


# combine the spotify data with the billboard hot-100 data 
billSpot = bill.merge(spotDF, on='id')    
    
## save the file
#billSpot.to_csv('spotify_billboard_merged.csv', index=False)




# (2) Getting new releases on spotify

# Some parameters for our spotify search
lim = 50 # number of tracks to get in each search (max is 50)
skip = 0 # the number of albums we want to skip down from the top of the list

# collect new releases
sp = spotipy.Spotify(auth=token)
response = sp.new_releases(limit=lim, offset=skip) 

newReleasesDF = pd.DataFrame()

for i in np.arange(lim):
    try:
        
        sp = spotipy.Spotify(auth=token)
        auri = response['albums']['items'][i]['uri'] # get the album
        arelease = response['albums']['items'][i]['release_date'] # the release date
        album = sp.album(auri) # and then lookup the album information
        albName = album['name'] # including the name
        label = album['label'] # and the label
        
        albumTracks = pd.DataFrame()
        
        for j in np.arange(len(album['tracks']['items'])):
            uri = album['tracks']['items'][j]['uri']
            sp = spotipy.Spotify(auth=token)
            track = sp.track(uri)
            features = sp.audio_features(uri)
            
            name = track['name']
            album_type = track['album']['album_type']
            length = track['duration_ms']
            explicit = track['explicit']
            track_num = track['track_number']
            artist = track['artists'][0]['name']
    
            da = features[0]['danceability']
            en = features[0]['energy']
            key = features[0]['key']
            vol = features[0]['loudness']
            sp = features[0]['speechiness']
            acoust = features[0]['acousticness']
            inst = features[0]['instrumentalness']
            live = features[0]['liveness']
            val = features[0]['valence']
            temp = features[0]['tempo']
            
            # get artist genre
            artID = track['artists'][0]['uri']
            arp = spotipy.Spotify(auth=token) 
            art = arp.artist(artID)
            genre = art['genres']
            
            
            albumTracks = albumTracks.append(pd.Series({'Track':name, 'Album Type':album_type, 
                                                        'Length':length, 'Track Number':track_num, 
                                                        'Explicit':explicit, 'Danceability':da, 
                                                        'Energy':en, 'Key':key, 'Loudness':vol, 
                                                        'Speechiness':sp, 'Acousticness':acoust,
                                                        'Instrumentalness':inst, 'Liveness':live, 
                                                        'Valence':val, 'Tempo':temp, 'Date':arelease,'Artist':artist,
                                                        'Label':label, 'Genre':genre, 'Album':albName}), ignore_index=True)
                
                
        newReleasesDF = newReleasesDF.append(albumTracks)
 
    except TypeError:
        continue

    
# and do it once more to get double the number of releases       

# skip past what we've already pulled
skip = skip + lim 

# collect new releases
sp = spotipy.Spotify(auth=token)
response = sp.new_releases(limit=lim, offset=skip) 

newReleasesDF2 = pd.DataFrame()

for i in np.arange(lim):
    try:
        
        sp = spotipy.Spotify(auth=token)
        auri = response['albums']['items'][i]['uri'] # get the album
        arelease = response['albums']['items'][i]['release_date'] # the release date
        album = sp.album(auri) # and then lookup the album information
        albName = album['name'] # including the name
        label = album['label'] # and the label
        
        albumTracks = pd.DataFrame()
        
        for j in np.arange(len(album['tracks']['items'])):
            uri = album['tracks']['items'][j]['uri']
            sp = spotipy.Spotify(auth=token)
            track = sp.track(uri)
            features = sp.audio_features(uri)
            
            name = track['name']
            album_type = track['album']['album_type']
            length = track['duration_ms']
            explicit = track['explicit']
            track_num = track['track_number']
            artist = track['artists'][0]['name']
    
            da = features[0]['danceability']
            en = features[0]['energy']
            key = features[0]['key']
            vol = features[0]['loudness']
            sp = features[0]['speechiness']
            acoust = features[0]['acousticness']
            inst = features[0]['instrumentalness']
            live = features[0]['liveness']
            val = features[0]['valence']
            temp = features[0]['tempo']
            
            # get artist genre
            artID = track['artists'][0]['uri']
            arp = spotipy.Spotify(auth=token) 
            art = arp.artist(artID)
            genre = art['genres']
            
            albumTracks = albumTracks.append(pd.Series({'Track':name, 'Album Type':album_type, 
                                                        'Length':length, 'Track Number':track_num, 
                                                        'Explicit':explicit, 'Danceability':da, 
                                                        'Energy':en, 'Key':key, 'Loudness':vol, 
                                                        'Speechiness':sp, 'Acousticness':acoust,
                                                        'Instrumentalness':inst, 'Liveness':live, 
                                                        'Valence':val, 'Tempo':temp, 'Date':arelease,'Artist':artist,
                                                        'Label':label, 'Genre':genre, 'Album':albName}), ignore_index=True)
                
                
        newReleasesDF2 = newReleasesDF2.append(albumTracks)
 
    except TypeError:
        continue   


# combine the two efforts
newReleasesDF = newReleasesDF.append(newReleasesDF2)
newReleasesDF = newReleasesDF.reset_index(drop=True)


# check to see if any of the new releases are already hits
matches = newReleasesDF.Track.isin(billSpot.Track)
newReleasesDF = newReleasesDF.drop(matches[matches==True].index)


# save the file
newReleasesDF.to_csv('spotify_new_releases.csv', index=False)
        



# (3) Getting random songs from spotify
    
#search for 2015-2018 songs (requires at least one search condition after date spec)
search_str = 'year:2015-2018 NOT Drake'


# some parameters to set for the spotify request
skip = 1000 # how many of the tracks to skip at the top of the list
lim = 50 # number of tracks to get in each search (max is 50)
searchNum = 40 # the number of searches to do

# expected number of tracks is lim * searchNum

results = list()

for i in np.arange(searchNum):
    sp = spotipy.Spotify(auth=token)
    result = sp.search(search_str, limit=lim,offset=skip)
    
    results.append(result)
    
    skip = skip + lim # redefine skip so we skip past the songs we already found
    

# extract all the track random track uris    
randTrackList = []    

for i in np.arange(len(results)):
    for j in np.arange(lim):
        trackUri = results[i]['tracks']['items'][j]['uri']
        randTrackList.append(trackUri)
        
    
# get all the track information for those songs
randomDF = pd.DataFrame()

for u in np.arange(lim*searchNum):
    try:
        
        urn = randTrackList[u]
        sp = spotipy.Spotify(auth=token)
        track = sp.track(urn)
        features = sp.audio_features(urn)
        
        name = track['name']
        album_type = track['album']['album_type']
        length = track['duration_ms']
        explicit = track['explicit']
        track_num = track['track_number']
        artist = track['artists'][0]['name']
        date = track['album']['release_date']
        albName = track['album']['name']
        
        da = features[0]['danceability']
        en = features[0]['energy']
        key = features[0]['key']
        vol = features[0]['loudness']
        sp = features[0]['speechiness']
        acoust = features[0]['acousticness']
        inst = features[0]['instrumentalness']
        live = features[0]['liveness']
        val = features[0]['valence']
        temp = features[0]['tempo']
        
        # get artist genre
        artID = track['artists'][0]['uri']
        arp = spotipy.Spotify(auth=token) 
        art = arp.artist(artID)
        genre = art['genres']
        
        # and record label
        ap = spotipy.Spotify(auth=token) # new spotipy instance for album lookup
        alb = track['album']['uri']
        albu = ap.album(alb)
        label = albu['label']

    
        randomDF = randomDF.append(pd.Series({'Track':name, 'Album Type':album_type, 
                                              'Length':length, 'Track Number':track_num, 
                                              'Explicit':explicit, 'Danceability':da, 
                                              'Energy':en, 'Key':key, 'Loudness':vol, 
                                              'Speechiness':sp, 'Acousticness':acoust, 
                                              'Instrumentalness':inst, 'Liveness':live, 
                                              'Valence':val, 'Tempo':temp, 'Artist':artist, 'Date':date,
                                              'Genre':genre, 'Label':label, 'Album':albName}), ignore_index=True)
    
    except TypeError:
        continue
    
        

# check to see if any of the random songs were hits
matches = randomDF.Track.isin(billSpot.Track)
randomDF = randomDF.drop(matches[matches==True].index)



# save the file
randomDF.to_csv('spotify_random_songs.csv', index=False)
    


    
    
    

### Get tweets about all of the songs we have from spotify

import got3 as got

import keras 
import nltk
import re
import codecs
import copy

from datetime import datetime, timedelta

from textblob import Word

from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

# stop words to remove from tweet text 
from nltk.corpus import stopwords
stop = stopwords.words('english')
keepwords = ["wouldn't", "won't", "shouldn't", "isn't", "haven't", "hadn't", "wasn't", "isn't", "ain't", 
             "doesn't", "didn't", "couldn't", "should", "should've", "don't", "very", "not", "nor", 
             "no", "most"]
for keepword in keepwords:
    if keepword in stop:
        stop.remove(keepword);


# function to clean up the text data from the tweets
def standardize_text(df, text_field):
    df[text_field] = df[text_field].str.replace(r"http\S+", "")
    df[text_field] = df[text_field].str.replace(r"http", "")
    df[text_field] = df[text_field].str.replace(r"@\S+", "")
    df[text_field] = df[text_field].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    df[text_field] = df[text_field].str.replace(r"@", "at")
    df[text_field] = df[text_field].str.lower()
    return df

# google's trained word2vec model 
import gensim
word2vec_path = "C:\\Users\mattf\\Desktop\\insight\\THE_PROJECT\\dev_setup\\GoogleNews-vectors-negative300.bin.gz"
word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
    if len(tokens_list)<1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged

def get_word2vec_embeddings(vectors, tweets, generate_missing=False):
    embeddings = tweets['tokens'].apply(lambda x: get_average_word2vec(x, vectors, generate_missing=generate_missing))
    return list(embeddings)

# function to convert datetime objects to strings
def to_string(date_obj):
    fmt = '%Y-%m-%d'
    return date_obj.strftime(fmt)
    


songDF = pd.read_csv('spotify_new_releases.csv')
# parameters for tweet number
twtmax = 25 # number of tweets per song to request
twtwindow = 3 # number of days on either side of release date to look for tweets

# convert date strings to date objects
songDF['Date'] = pd.to_datetime(songDF['Date'], infer_datetime_format=True)

## note: only necessary for spotify/billboard merged data
#songDF['DateS'] = pd.to_datetime(songDF['DateS'], infer_datetime_format=True)
#songDF['Artist'] = songDF['Artist_y']

masterTDF = pd.DataFrame()
#
for i in np.arange(len(songDF)): 
#for i in range(334,392):     
            
    print('song ', i+1, ' processing: ', i/len(songDF)*100,'% complete')
    print(len(masterTDF), ' entries appended')
    
    # extract track name/artist for the row, and make a search query from them
    trackName = songDF.loc[i].Track 
    artistName = songDF.loc[i]['Artist']
    artistKey = ' '.join(artistName.split()[:2])
    query = trackName + ' ' + artistKey
    
    # add query terms to a temporary stop list
    newStop = query.split()
    stopTemp = copy.deepcopy(stop, memo=None, _nil=[])
    stopTemp.extend(newStop)
    
#    # deal with some erroneous billboard dates
#    # i.e., there were multiple hits with same name, so the date gets skewed
#    date = max(songDF.loc[i].Date, songDF.loc[i].DateS)
    
    # use the release date to create a twitter search window 
    date = songDF.loc[i].Date # comment this out for billSpot data 
    dateStart = date - timedelta(days=twtwindow)
    dateStart = ' '.join(to_string(dateStart).split()[:1])
    dateStop = date + timedelta(days=twtwindow)
    dateStop = ' '.join(to_string(dateStop).split()[:1])
    
    tweetDF = pd.DataFrame()
    
    # get tweets by date bounding and query search
    tweetCriteria = got.manager.TweetCriteria().setQuerySearch(query).setSince(dateStart).setUntil(dateStop).setTopTweets(True).setMaxTweets(twtmax) 
    
    # pull tweets and extract the parameters of interest
    for t in np.arange(twtmax):
        
        try:
        
            tweet = got.manager.TweetManager.getTweets(tweetCriteria)[t]
            
            txt = tweet.text
            rtw = tweet.retweets
            fav = tweet.favorites
            date = tweet.date
            
            tweetDF = tweetDF.append(pd.Series({'Text':txt, 'Retweets':rtw, 'Favorites':fav, 
                                                'Date':date}), ignore_index=True)
            
        except IndexError:
            continue
        
    # NLP for the collection of tweets
    
    try:
    
        # clean up text
        tweets = standardize_text(tweetDF, "Text")
        
        # get rid of stop words
        tweets['stopwords'] = tweets['Text'].apply(lambda x: len([x for x in x.split() if x in stopTemp]))
        tweets['Text'] = tweets['Text'].apply(lambda x: " ".join(x for x in x.split() if x not in stopTemp))
        
        # lemmatize
        tweets['Text'] = tweets['Text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
        
        # tokenize the text strings
        tweets["tokens"] = tweets["Text"].apply(tokenizer.tokenize)
        
        # add a sentence lengths variable
        sentence_lengths = [len(tokens) for tokens in tweets["tokens"]]
        tweets['length'] = pd.Series(sentence_lengths, index=tweets.index)
        
        # apply sentiment analysis
        embeddings = get_word2vec_embeddings(word2vec, tweets)
        totSent = pd.DataFrame(embeddings).mean(axis=0)
        
        # create index labels
        totSent.index = totSent.index.map(str)
        totSent.index = 's' + totSent.index.astype(str)
        
        # summarize the tweet information for a single song
        tn = tweets.shape[0] / twtmax
        rtMean = tweets['Retweets'].mean()
        rtMax = tweets['Retweets'].max()
        rtMed = tweets['Retweets'].median()
        favMean = tweets['Favorites'].mean()
        favMax = tweets['Favorites'].max()
        favMed = tweets['Favorites'].median()
        stopsMean = tweets['stopwords'].mean()
        stopsMed = tweets['stopwords'].median()
        lengthMean = tweets['length'].mean()
        lengthMed = tweets['length'].median()
        
        
        
        tweetStats = pd.Series({'Relative#':tn, 'Mean Retweets':rtMean, 
                                 'Max Retweets':rtMax, 'Median Retweets':rtMed, 
                                 'Mean Favorites':favMean, 'Max Favorites':favMax, 
                                 'Median Favorites':favMed, 'Mean Stops':stopsMean,
                                 'Median Stops':stopsMed, 'Mean Length':lengthMean,
                                 'Median Length':lengthMed, 'Track':trackName})
        
        tweetSeries = tweetStats.append(totSent)
        
                   
        masterTDF = masterTDF.append(tweetSeries, ignore_index=True)
    
    except KeyError:
        continue
                 
                                       
    
                                        
                                        
                                        
                                        