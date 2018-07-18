# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 17:12:06 2018

@author: mattf
"""

## creainge a playlist
#playlist = sp.user_playlist_create(SPOTIFY_USERNAME, 'SPOTting The Hits')
#
## building a playlist
#for track in whatever():
#    url = track.spotifyLink # or whatever the link ID is stored as in my final list
#    sp.user_playlist_add_tracks(SPOTIFY_USERNAME, playlist_id, [url])
#
#
## get url for an artist image - useful for the web app
#    
#import sys
#
#spotify = spotipy.Spotify()
#
#if len(sys.argv) > 1:
#    name = ' '.join(sys.argv[1:])
#else:
#    name = 'Radiohead'
#
#results = spotify.search(q='artist:' + name, type='artist')
#items = results['artists']['items']
#if len(items) > 0:
#    artist = items[0]
#    print artist['name'], artist['images'][0]['url']

