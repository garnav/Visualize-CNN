import base64
import cv2
import io
import math
import numpy as np
import os
from PIL import Image

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

from werkzeug.utils import secure_filename

import serve
import visualize

# ===== CONSTANTS =============================================================
DEVICE = serve.get_device()
MODEL = serve.get_model().to(DEVICE)
MODEL_LAYERS = serve.get_conv_layers(MODEL)
MODEL_CLASSES = serve.get_model_classes()
TRANSFORMS = serve.get_transforms()

EXTERNAL_STYLESHEETS = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
UPLOAD_DIRECTORY = ""

# ===== INIT APP ==============================================================
app = dash.Dash(__name__, external_stylesheets=EXTERNAL_STYLESHEETS)

# ===== COMPONENTS =================================================================
def class_dropdown(div_id, model_classes):
    return dcc.Dropdown(id=div_id, options=[{'label' : name, 'value' : value} for name, value in model_classes])  

def layer_checklist(div_id, model_layers):
    return dcc.Checklist(id=div_id, options=[{'label' : layer, 'value' : layer} for layer in model_layers])  

def image_upload(div_id):
    return dcc.Upload(id=div_id,
                      children=html.Div(['Drag and Drop or ',
                                         html.A('Select Files')
                                        ]),
                      style={
                        'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px'
                        },
                      multiple=False)

def parse_image(contents, filename):
    content_type, content_string = contents.split(',')
    if 'image' in content_type:
        decoded = base64.b64decode(content_string)
        return Image.open(io.BytesIO(decoded)), secure_filename(filename)

# ===== VIEWS =================================================================
@app.callback([Output('storage', 'data'),
               Output('original-image', 'children')],
              [Input('upload-image', 'contents')],
              [State('upload-image', 'filename'),
               State('upload-image', 'last_modified')])
def upload_image(contents, filename, last_modified):
    if contents is not None:
        img, fname = parse_image(contents, filename)
        img_pth = os.path.join(UPLOAD_DIRECTORY, fname)
        img.save(img_pth)
        return img_pth, html.Img(src=contents, style={'maxWidth':'50%'})
    
    return None, None

@app.callback(Output('plot', 'children'),
              [Input('submit-button', 'n_clicks')],
              [State('layer-choice', 'value'),
               State('class-choice', 'value'),
               State('storage', 'data')])
def compare_layers(n_clicks, chosen_layers, chosen_class, img_pth):    
    if (chosen_layers is None) or (chosen_class is None) or (img_pth is None):
        return dash.no_update
    
    image = serve.get_resize_transform()(cv2.imread(img_pth))
    run_gradcam = serve.serve_gradcam(MODEL, TRANSFORMS, chosen_layers)
    cams = run_gradcam(image, int(chosen_class), DEVICE)
    fig = visualize.create_subplots(cams, image)
    return dcc.Graph(figure=fig)

def compare_classes(n_clicks, chosen_layers, chosen_class1, chosen_class2, img_pth):
    # Compare Layers for Different Classes
    pass

def compare_images(n_clicks, chosen_layers, chosen_class, img_pth1, img_pth2):
    # Compare Layers for Two different Images for a Chosen Class
    pass
    
# ===== MAIN =================================================================
app.layout = html.Div([
    html.Div([
        html.Div([
            html.Label('Choose An Image'),
            html.Div(id="original-image", 
                     style={"textAlign" : "center", "display" : "block"}),
            dcc.Store(id="storage"),
            image_upload('upload-image'),
            html.Div([
                html.Label('Choose Class'),
                class_dropdown('class-choice', MODEL_CLASSES),
                html.Label('Choose Layers'),
                layer_checklist('layer-choice', MODEL_LAYERS),
                html.Button(id='submit-button', n_clicks=0, children='Submit') 
            ])
        ], style={'height':'100vh'}),
    ], style={'width': '25%', 'display': 'inline-block', 'float' : 'left'}),
    html.Div(id='plot',
             style={'width': '74%', 'display': 'inline-block', "textAlign" : "center", 'float' : 'right', 'overflowY' : 'scroll', 'height' : '95vh'})
])

if __name__ == "__main__":
    app.run_server(debug=True)

# TODO
# 1. When more subplots are added then it is superimposed on the current one [Bug]
# 2. Order of chosen_layers should reflect the list its provided in
# 3. Update the dropdown based on the image run