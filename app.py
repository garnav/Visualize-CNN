import base64
import cv2
import io
import os
from PIL import Image

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

from werkzeug.utils import secure_filename

import serve
import visualize

# ===== MODEL =================================================================
DEVICE = serve.get_device()
MODEL = serve.get_model().to(DEVICE)
MODEL_LAYERS = serve.get_conv_layers(MODEL)
MODEL_CLASSES = serve.get_model_classes()
INV_MODEL_CLASSES = serve.inv_model_classes(MODEL_CLASSES)
TRANSFORMS = serve.get_transforms()

# ===== CONSTANTS =================================================================
MAP_TABS = {
    'single-image-tab' : {
        'name' : 'Compare Layers for a Single Image',
        'image' : ['single-image-choose-image'],
        'image-display' : ['single-image-display-image'],
        'class' : ['single-image-choose-class'],
        'layer' : 'layer-choice',
        'storage' : 'single-image-storage',
        'submit' : 'single-image-submit',
        'plot' : 'single-image-plot'
    },
    'multiple-images-tab' : {
        'name' : 'Compare Layers for Multiple Images',
        'image' : ['multiple-images-choose-image1', 
                   'multiple-images-choose-image2'],
        'image-display' : ['multiple-images-display-image1', 
                           'multiple-images-display-image2'],
        'class' : ['multiple-images-choose-class'],
        'layer' : 'layer-choice',
        'storage' : 'multiple-images-storage',
        'submit' : 'multiple-images-submit',
        'plot' : 'multiple-images-plot'
    },
    'multiple-classes-tab' : {
        'name' : 'Compare Layers and Classes for a Single Image',
        'image' : ['multiple-classes-choose-image'],
        'image-display' : ['multiple-classes-display-image'],
        'class' : ['multiple-classes-choose-class1', 
                   'multiple-classes-choose-class2'],
        'layer' : 'layer-choice',
        'storage' : 'multiple-classes-storage',
        'submit' : 'multiple-classes-submit',
        'plot' : 'multiple-classs-plot'
    }
}

EXTERNAL_STYLESHEETS = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
UPLOAD_DIRECTORY = ""

# ===== INIT APP ==============================================================
app = dash.Dash(__name__, external_stylesheets=EXTERNAL_STYLESHEETS) 
app.config['suppress_callback_exceptions']=True

# ===== USER INPUT =================================================================
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
    
def submit_button(div_id):
    return html.Button(id=div_id, n_clicks=0, children='Submit')

# ===== IMAGE UPLOAD =================================================================
def parse_image(contents, filename, save_dir):
    content_type, content_string = contents.split(',')
    if 'image' in content_type:
        decoded = base64.b64decode(content_string)
        image = Image.open(io.BytesIO(decoded))
        fname = secure_filename(filename)
        
        if save_dir is not None:
            img_pth = os.path.abspath(os.path.join(save_dir, fname))
            image.save(img_pth)
            return img_pth, image, fname
        return None, image, fname
    return None

# contents_fname = [(c1, f1), (c2, f2), ...]
# image_paths = list of existing image paths
def upload_images(contents_fname, image_paths):
    for i, (contents, fname) in enumerate(contents_fname):
        if contents is not None:
            image_paths[i] = parse_image(contents, fname, UPLOAD_DIRECTORY)[0]
    return image_paths

@app.callback([Output(MAP_TABS['single-image-tab']['storage'], 'data'),
               Output(MAP_TABS['single-image-tab']['image-display'][0], 'children')],
             [Input(MAP_TABS['single-image-tab']['image'][0], 'contents')],
             [State(MAP_TABS['single-image-tab']['image'][0], 'filename'), 
              State(MAP_TABS['single-image-tab']['storage'], 'data')])
def single_image_upload(contents, filename, image_paths):
    img_div = dash.no_update if contents is None else  html.Img(src=contents, style={'maxWidth':'50%'})
    return upload_images([(contents, filename)], image_paths), img_div

@app.callback([Output(MAP_TABS['multiple-images-tab']['storage'], 'data'), 
               Output(MAP_TABS['multiple-images-tab']['image-display'][0], 'children'), 
               Output(MAP_TABS['multiple-images-tab']['image-display'][1], 'children')],
              [Input(MAP_TABS['multiple-images-tab']['image'][0], 'contents'), 
               Input(MAP_TABS['multiple-images-tab']['image'][1], 'contents')],
              [State(MAP_TABS['multiple-images-tab']['image'][0], 'filename'), 
               State(MAP_TABS['multiple-images-tab']['image'][1], 'filename'),
               State(MAP_TABS['multiple-images-tab']['storage'], 'data')])
def multiple_image_upload(contents1, contents2, filename1, filename2, image_paths):
    img_div1 = dash.no_update if contents1 is None else  html.Img(src=contents1, style={'maxWidth':'50%'})
    img_div2 = dash.no_update if contents2 is None else  html.Img(src=contents2, style={'maxWidth':'50%'})
    return upload_images([(contents1, filename1), (contents2, filename2)], image_paths), img_div1, img_div2

@app.callback([Output(MAP_TABS['multiple-classes-tab']['storage'], 'data'),
               Output(MAP_TABS['multiple-classes-tab']['image-display'][0], 'children')],
             [Input(MAP_TABS['multiple-classes-tab']['image'][0], 'contents')],
             [State(MAP_TABS['multiple-classes-tab']['image'][0], 'filename'), 
              State(MAP_TABS['multiple-classes-tab']['storage'], 'data')])
def multiple_class_upload(contents, filename, image_paths):
    img_div = dash.no_update if contents is None else  html.Img(src=contents, style={'maxWidth':'50%'})
    return upload_images([(contents, filename)], image_paths), img_div

# ===== COMPONENTS =================================================================
@app.callback(Output(MAP_TABS['single-image-tab']['plot'], 'children'),
              [Input(MAP_TABS['single-image-tab']['submit'], 'n_clicks')],
              [State(MAP_TABS['single-image-tab']['layer'], 'value'),
               State(MAP_TABS['single-image-tab']['class'][0], 'value'),
               State(MAP_TABS['single-image-tab']['storage'], 'data')])
def compare_layers(n_clicks, chosen_layers, chosen_class, img_pths):
    img_pth = img_pths[0]    
    if (chosen_layers is None) or (chosen_class is None) or (img_pth is None):
        return dash.no_update
    print(img_pth)
    print(type(cv2.imread(img_pth)))
    image = cv2.imread(img_pth)
    run_gradcam = serve.serve_gradcam(MODEL, TRANSFORMS, chosen_layers)
    cams = run_gradcam(image, int(chosen_class), DEVICE)
    fig = visualize.layer_comp_subplots(cams, serve.get_resize_transform()(image))
    return dcc.Graph(figure=fig)

@app.callback(Output(MAP_TABS['multiple-classes-tab']['plot'], 'children'),
              [Input(MAP_TABS['multiple-classes-tab']['submit'], 'n_clicks')],
              [State(MAP_TABS['multiple-classes-tab']['layer'], 'value'),
               State(MAP_TABS['multiple-classes-tab']['class'][0], 'value'),
               State(MAP_TABS['multiple-classes-tab']['class'][1], 'value'),
               State(MAP_TABS['multiple-classes-tab']['storage'], 'data')])
def compare_classes(n_clicks, chosen_layers, chosen_class1, chosen_class2, img_pths):
    img_pth = img_pths[0]
    # Compare Layers for Different Classes
    if (chosen_layers is None) or (chosen_class1 is None) or (chosen_class2 is None) or (img_pth is None):
        return dash.no_update
    
    image = cv2.imread(img_pth)
    run_gradcam = serve.serve_gradcam(MODEL, TRANSFORMS, chosen_layers)
    cams1 = run_gradcam(image, int(chosen_class1), DEVICE)
    cams2 = run_gradcam(image, int(chosen_class2), DEVICE)
    fig = visualize.class_comp_subplots(cams1, cams2, 
                                        [INV_MODEL_CLASSES[chosen_class1], INV_MODEL_CLASSES[chosen_class2]], 
                                        serve.get_resize_transform()(image))
    return dcc.Graph(figure=fig)
    
@app.callback(Output(MAP_TABS['multiple-images-tab']['plot'], 'children'),
              [Input(MAP_TABS['multiple-images-tab']['submit'], 'n_clicks')],
              [State(MAP_TABS['multiple-images-tab']['layer'], 'value'),
               State(MAP_TABS['multiple-images-tab']['class'][0], 'value'),
               State(MAP_TABS['multiple-images-tab']['storage'], 'data')])
def compare_images(n_clicks, chosen_layers, chosen_class, img_pths):
    img_pth1, img_pth2 = img_pths
    # Compare Layers for Two different Images for a Chosen Class
    if (chosen_layers is None) or (chosen_class is None) or (img_pth1 is None) or (img_pth2 is None):
        return dash.no_update
    
    image1 = cv2.imread(img_pth1)
    image2 = cv2.imread(img_pth2)
    run_gradcam = serve.serve_gradcam(MODEL, TRANSFORMS, chosen_layers)
    cams1 = run_gradcam(image1, int(chosen_class), DEVICE)
    cams2 = run_gradcam(image2, int(chosen_class), DEVICE)
    pass

# ===== TABS =================================================================
def single_image_tab():
    tab_contents = html.Div([
        html.Label('Choose Image'),
        html.Div(id=MAP_TABS['single-image-tab']['image-display'][0], 
                 style={"textAlign" : "center", "display" : "block"}),
        html.Div(id="original-image", 
                    style={"textAlign" : "center", "display" : "block"}),
        image_upload(MAP_TABS['single-image-tab']['image'][0]),
        html.Label('Choose Class'),
        class_dropdown(MAP_TABS['single-image-tab']['class'][0], MODEL_CLASSES),
        html.Label('Choose Layers'),
        layer_checklist(MAP_TABS['single-image-tab']['layer'], MODEL_LAYERS),
        submit_button(MAP_TABS['single-image-tab']['submit']),
        dcc.Store(id=MAP_TABS['single-image-tab']['storage'], data=[None])
    ], style={'height':'100vh'})
    
    return tab_contents

def multiple_image_tab():
    tab_contents = html.Div([
        html.Label('Choose Image 1'),
        html.Div(id=MAP_TABS['multiple-images-tab']['image-display'][0], 
                 style={"textAlign" : "center", "display" : "block"}),
        image_upload(MAP_TABS['multiple-images-tab']['image'][0]),
        html.Label('Choose Image 2'),
        html.Div(id=MAP_TABS['multiple-images-tab']['image-display'][1], 
                 style={"textAlign" : "center", "display" : "block"}),
        image_upload(MAP_TABS['multiple-images-tab']['image'][1]),
        html.Label('Choose Class'),
        class_dropdown(MAP_TABS['multiple-images-tab']['class'][0], MODEL_CLASSES),
        html.Label('Choose Layers'),
        layer_checklist(MAP_TABS['multiple-images-tab']['layer'], MODEL_LAYERS),
        submit_button(MAP_TABS['multiple-images-tab']['submit']),
        dcc.Store(id=MAP_TABS['multiple-images-tab']['storage'], data=[None, None])
    ], style={'height':'100vh'})
    
    return tab_contents

def multiple_class_tab():
    tab_contents = html.Div([
        html.Label('Choose An Image'),
        html.Div(id=MAP_TABS['multiple-classes-tab']['image-display'][0], 
                 style={"textAlign" : "center", "display" : "block"}),
        image_upload(MAP_TABS['multiple-classes-tab']['image'][0]),
        html.Label('Choose Class 1'),
        class_dropdown(MAP_TABS['multiple-classes-tab']['class'][0], MODEL_CLASSES),
        html.Label('Choose Class 2'),
        class_dropdown(MAP_TABS['multiple-classes-tab']['class'][1], MODEL_CLASSES),
        html.Label('Choose Layers'),
        layer_checklist(MAP_TABS['multiple-classes-tab']['layer'], MODEL_LAYERS),
        submit_button(MAP_TABS['multiple-classes-tab']['submit']),
        dcc.Store(id=MAP_TABS['multiple-classes-tab']['storage'], data=[None])
    ], style={'height':'100vh'})

    return tab_contents

@app.callback([Output('config', 'children'),
               Output('plot', 'children')],
              [Input('tabs', 'value')])
def choose_tabs(selected_tab):
    if selected_tab == 'single-image-tab':
        return single_image_tab(), html.Div(id=MAP_TABS['single-image-tab']['plot'])
    elif selected_tab == 'multiple-images-tab':
        return multiple_image_tab(), html.Div(id=MAP_TABS['multiple-images-tab']['plot'])
    elif selected_tab == 'multiple-classes-tab':
        return multiple_class_tab(), html.Div(id=MAP_TABS['multiple-classes-tab']['plot'])
    
# ===== MAIN =================================================================
app.layout = html.Div([
    html.H4("ConvNets: Peek Under the Hood"),
    html.Div(dcc.Tabs(id='tabs', value='single-image-tab', 
                      children=[dcc.Tab(label=v['name'], value=k) for k, v in MAP_TABS.items()])), 
    html.Div(children=[html.Div(id='config', 
                                style={'width': '25%', 'display': 'inline-block', 'float' : 'left'}), 
                       html.Div(id='plot',
                                style={'width': '74%', 'display': 'inline-block', 'float' : 'right', 
                                       'textAlign' : 'center', 'overflowY' : 'scroll', 'height' : '95vh'})])                       
])

if __name__ == "__main__":
    app.run_server(debug=True)