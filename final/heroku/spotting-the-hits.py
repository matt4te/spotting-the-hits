
### load modules

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import plotly.figure_factory as ff

import os
import sys
import numpy as np
import pandas as pd
import sklearn
import pickle

import spotipy
import spotipy.util as util
from spotipy.oauth2 import SpotifyClientCredentials

from sklearn.ensemble import GradientBoostingClassifier


# spotify credentials
username = 'MattForetich' # spotify developer account, account of the playlist
SPOTIPY_CLIENT_ID = '55e278fb84844a49a48911143295fafe' # an API generated when you register an app
SPOTIPY_CLIENT_SECRET = 'bec3776b658846509fae6e9b27fb2b5a' # as above
# SPOTIPY_REDIRECT_URI  = 'http://localhost/' # the URL the user will be directed to after the authorization process
# scope = 'playlist-modify-public' # the single permission we’ll need to create and modify a public playlist
# token = util.prompt_for_user_token(username,scope,client_id=SPOTIPY_CLIENT_ID,client_secret=SPOTIPY_CLIENT_SECRET,redirect_uri=SPOTIPY_REDIRECT_URI)
client_credentials_manager = SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

colorscale = [[0, '#458B00'],[.5, '#DCDCDC'],[1, '#C1FFC1']]




#### do the computations

#load the data and the model
fullSongInfo = pd.read_csv('newSongsCoded.csv')
newSongs = pd.read_csv('X.csv')


filename = 'week3_bdt.sav'
bdt = pickle.load(open(filename, 'rb'))

# run the model
predictions= bdt.predict(newSongs)
probs = pd.DataFrame(bdt.predict_proba(newSongs))



# Merge prediction probabilities with full data and sort by prediction of being a hit, and drop songs predicted to be non-hits
probs.columns = ['Non-Hit', 'Hit']
fullData = fullSongInfo.merge(probs, left_index=True, right_index=True)
sort = fullData.sort_values('Hit', ascending=False)
fullI = sort.reset_index(drop=True)
normI = fullI[fullI.Hit > 0.5]


    
# build a list of list (possible genres and sub-genres to exclude)
genres = []
countryList = ['alternative country', 'australian country', 'bluegrass', 'contemporary country', 'country', 'country dawn', 'country road', 'deep texas country', 'modern country rock', 'outlaw country', 'progressive bluegrass']
rapList = ['rap','r&b', 'hip hop', 'alternative hip hop', 'alternative r&b', 'canadian hip hop', 'deep pop r&b', 'deep southern trap', 'deep underground hip hop', 'detroit hip hop', 'dirty south rap', 'east coast hip hop', 'french hip hop', 'gangster rap', 'hardcore hip hop']
rockList = ['alternative metal', 'alternative rock', 'art rock', 'boston rock', 'canadian metal', 'canadian rock', 'classic rock', 'dance rock', 'experimental rock', 'jam band', 'modern alternative rock', 'modern rock', 'rock', 'soft rock']
popList = ['teen pop', 'acoustic pop', 'art pop', 'australian pop', 'boy band', 'britpop', 'canadian pop', 'candy pop', 'chamber pop', 'dance pop', 'deep danish pop', 'europop', 'german pop', 'girl group', 'italian pop', 'k-pop', 'pop']
elecList = ['trance', 'house', 'grime', 'german techno', 'alternative dance', 'australian dance', 'bass music', 'bass trap', 'brostep', 'deep euro house', 'deep groove house', 'deep pop edm', 'deep trap', 'deep tropical house', 'dutch house', 'edm', 'electro', 'electro house', 'electronic trap'] 
latList = ['argentine hip hop', 'latin', 'latin electronica', 'latin hip hop', 'latin pop', 'mexican pop', 'spanish pop']
emoList = ['alternative emo', 'emo', 'emo rap', 'pop emo', 'screamo']
christList = ['anthem worship', 'christian hip hop', 'christian music', 'christian rock', 'gospel']
genres.append(countryList)
genres.append(rapList)
genres.append(rockList)
genres.append(popList)
genres.append(elecList)
genres.append(latList)
genres.append(emoList)
genres.append(christList) 
    
    
    
    

### layout the application
app = dash.Dash(__name__)
server = app.server



# set aesthetics
colors = {
    'background':'#000000',
    'text':'#66CD00'
}

# bootstrap css
app.css.append_css({'external_url':'https://codepen.io/amyoshino/pen/jzXypZ.css'})

# the layout
app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[

        # row 1
        html.Div(
            [   
                html.Img(
                    src='https://www.themarysue.com/wp-content/uploads/2017/08/spotify-logo.jpg',
                    className='three columns',
                    style={
                        'height': '20%',
                        'width': '20%',
                        'float': 'left',
                        'position': 'relative',
                        'margin': 25
                    }
                ),
                
                html.H1(
                    children='Spotting the Hits!',
                    className='six columns',
                    style={
                        'textAlign':'center',
                        'color':colors['text'],
                        'fontSize':80,
                        'margin-left':60
                    }
                ),
                
                html.Img(
                    src='https://vignette.wikia.nocookie.net/dnd4/images/b/bb/Twitter_logo_dark.png/revision/latest?cb=20150602032047',
                    className='three columns',
                    style={
                        'height': '15%',
                        'width': '15%',
                        'float': 'right',
                        'position': 'relative',
                        'margin-right': 25,
                        'margin-bottom': 25
                    }    
                )
            ], className='row'
        ),
        
        
        # row 2
        html.Div(
            [
                html.Div(
                    children='Welcome to Spotting the hits! Harness the immense musical library of Spotify and the trendiness of Twitter to produce a playlist of songs that are about to take off! Stay ahead of the curve (and your friends) by spotting the hits before they make it big!',
                    style={
                        'textAlign':'center',
                        'color':colors['text'],
                        'margin-top':50,
                        'margin-bottom':100,
                        'fontSize':23
                    }    
                )
            ], className='row'
        ),
        
        
        # row 3
        html.Div(
            [
                html.Div([
                    html.Div(
                        children='Genres you DONT want included in your playlist:',
                        style={
                            'textAlign':'center',
                            'color':colors['text'],
                            'margin-bottom':10,
                            'fontSize':28
                        }
                    )
                ], className='six columns'
                ),
                
                html.Div([
                    html.Div(
                        children='What type of music do you usually like?',
                        style={
                            'textAlign':'center',
                            'color':colors['text'],
                            'margin-bottom':10,
                            'fontSize':28
                        }
                    )
                ], className='six columns'
                )  
            ], className = 'row'
        ),
        
        
        # row 4
        html.Div(
            [
                html.Div([
                    dcc.Checklist(
                        id='genre',
                        options=[
                            {'label': 'Country / Bluegrass', 'value': 0},
                            {'label': 'Rap / R&b', 'value': 1},
                            {'label': 'Rock', 'value': 2},
                            {'label': 'Pop', 'value': 3},
                            {'label': 'Electronic', 'value': 4},
                            {'label': 'Latin', 'value': 5},
                            {'label': 'Emo', 'value': 6},
                            {'label': 'Christian', 'value': 7}
                        ],
                        values = [],
                        style={
                            'textAlign':'center',
                            'color':colors['text']
                        },
                    )
                ], className='six columns'
                ),
                
                html.Div([
                    dcc.RadioItems(
                        id='vibe',
                        options=[
                            {'label': 'Chill instrumental music - I want to relax...', 'value': '1'},
                            {'label': 'Hype music - I want to dance!', 'value': '2'},
                            {'label': 'Vocal heavy Music - I want to sing!', 'value': '3'},
                            {'label': 'Just give the best tracks!', 'value':'4'}
                        ],
                        value='4',
                        style={
                            'textAlign':'center',
                            'color':colors['text']
                        }        
                    )
                ], className='six columns'
                )
            ], className = 'row'
        ),
        
        # row 5
        html.Div(
            [
                html.Div([
                    html.Div(
                        children='Are there any artists you want to exclude?',
                        style={
                            'textAlign':'center',
                            'color':colors['text'],
                            'margin-top': 50,
                            'margin-bottom':10,
                            'fontSize':28
                        }
                    )
                ], className='six columns'
                ),
                
                html.Div([
                    html.Div(
                        children='How many songs are you interested in listening to?',
                        style={
                            'textAlign':'center',
                            'color':colors['text'],
                            'margin-top':50,
                            'margin-bottom':10,
                            'fontSize':28
                        }
                    )
                ], className='six columns'
                )
            ], className='row'
        ),
        
        # row 6
        html.Div(
            [
                html.Div([
                    dcc.Input(
                        id='buzzkills',
                        placeholder='Buzzkills',
                        type='text',
                        value='',
                        style ={
                            'margin-left':250
                        }
                    )
                ], className='six columns'
                ),
                
                html.Div([
                     dcc.RadioItems(
                        id='optim',
                        options=[
                            {'label': 'The most likely tracks to become hits', 'value':0},
                            {'label': 'Manageable - 10 songs at most', 'value':1},
                            {'label': 'Bigger is better - I dont want to miss a potential hit!', 'value':2}
                        ],
                        value=0,
                        style={
                            'textAlign':'center',
                            'color':colors['text']
                        }        
                    )      
                ], className='six columns'
                )
            ], className='row'
        ),

        # row 7
        html.Div(
            html.Button('Create My Playlist!', 
                id='button'),
                style={
                    'textAlign':'center',
                    'color':colors['text'],
                    'margin-bottom': 40,
                    'margin-top': 25
                }                
        ),
        
        # row 8
        html.Div(id='output'),
        
        
        # row 9
        html.Div(
            [
                html.Div(
                    id='embed',
                    className='six columns'
                ),
                
                html.Div(
                    id='image',
                    className='six columns'
                )
            ], className='row'
        )  
        
    ]
)



### interactive functions
@app.callback(
    Output('output','children'),
    [Input('button','n_clicks')],
    state = [State('genre','values'), State('buzzkills','value'), State('optim','value')]
)
def print_playlist(n_clicks, genre_values, buzzkills_value, optim_value):
    
    
    if optim_value == 2:
        bigI = fullI[fullI.Hit > 0.25]
        tableI = bigI
    else:
        tableI = normI
        
           
           
    newI = pd.DataFrame()
    if genre_values:
        genList = []
        dropList = []
        for value in genre_values:
            genList.extend(genres[value])
        for g in genList:
            for r in np.arange(len(tableI)):
                if tableI.iloc[r][g] == 1:
                    dropList.append(r)
        newI = tableI.drop(tableI.index[list(np.unique(dropList))])
        newList = pd.DataFrame()
        for i in np.arange(len(newI)):
            track = newI.iloc[i].Track
            artist = newI.iloc[i].Artist
            newList = newList.append(pd.Series({'Track':track, 'Artist':artist}),ignore_index=True)
    else:
        hits = pd.DataFrame()
        for i in np.arange(len(tableI)):
            track = tableI.iloc[i].Track
            artist = tableI.iloc[i].Artist
            hits = hits.append(pd.Series({'Track':track, 'Artist':artist}),ignore_index=True)
            
            
            
    subList = pd.DataFrame()
    if buzzkills_value:
        if len(newI.index) > 0:
            newList = newList[newList.Artist != buzzkills_value]
        else:
            subList = hits
            subList = subList[subList.Artist != buzzkills_value]

               

    if n_clicks > 0:
        if len(newI.index) > 0:
            hitList = newList
        elif len(subList.index) > 0:
            hitList = subList
        else:
            hitList = hits
        if optim_value == 1:
            hitList = hitList.iloc[0:10,:]
        table = ff.create_table(hitList, colorscale=colorscale, height_constant=40)
        for i in range(len(table.layout.annotations)):
            table.layout.annotations[i].font.size = 18
        fig = html.Div(dcc.Graph(id='playlist', figure = table))
    return fig
    
    
@app.callback(
    Output('embed','children'),
    [Input('button','n_clicks')]
)
def embed_playlist(n_clicks):
    
    link = html.Iframe()
    if n_clicks > 0:
        link = html.Iframe(src="https://open.spotify.com/embed/user/1268126297/playlist/6ut8CWc9CUP1V9iLg4nPkL", style={'width':600, 'height':600, 'frameborder':25, 'allowtransparency':True, 'allow':'encrypted-media'})
    return link
    

@app.callback(
    Output('image','children'),
    [Input('button','n_clicks')]
)

def embed_image(n_clicks):
    
    spotify = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    user = '1268126297'
    results = spotify.user_playlists(user)
    url = results['items'][1]['images'][0]['url']
    
    if n_clicks > 0:
        link = html.Img(src=url, style={'width':500, 'height':500, 'margin-top':80,'margin-left':80,'margin-right':100},)
    return link

    

   


if __name__ == '__main__':
    app.run_server(debug=True)