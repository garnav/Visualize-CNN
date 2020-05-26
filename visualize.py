import math
import numpy as np

from plotly import graph_objects as go
from plotly.subplots import make_subplots

def layer_comp_subplots(cams, image): 
    chosen_layers = list(cams.keys()) # TODO: Sort By Something
    sample_cam = cams[chosen_layers[0]]
    height, width = sample_cam.shape[1:]
    all_heatmaps = np.zeros((len(chosen_layers), height, width), dtype=sample_cam.dtype)
    for i, layer in enumerate(chosen_layers):
        all_heatmaps[i, :, :] = cams[layer][0, :, :]
        
    fig = create_subplots(all_heatmaps, image, subplot_titles=chosen_layers, row_titles=None, col_titles=None)
    fig.update_layout(title_text="GradCAM - Layer Comparisons")
    fig.write_html(f"fig{len(chosen_layers)}.html")
    return fig

# cam1 and cam2 should have the same keys
def class_comp_subplots(cam1, cam2, class_names, image):
    chosen_layers = list(cam1.keys()) # TODO: Sort By Something
    sample_cam = cam1[chosen_layers[0]]
    height, width = sample_cam.shape[1:]
    all_heatmaps = np.zeros((len(chosen_layers * 2),  height, width), dtype=sample_cam.dtype)
    for i, layer in enumerate(chosen_layers):
        all_heatmaps[i, :, :] = cam1[layer][0, :, :]
        all_heatmaps[i + 1, :, :] = cam2[layer][0, :, :]
    
    fig = create_subplots(all_heatmaps, image, subplot_titles=None, row_titles=chosen_layers, col_titles=class_names)   
    fig.update_layout(title_text="GradCAM - Layer & Class Comparisons")
    return fig

def image_comp_subplots(cam1, cam2, image):
    pass

# all_heatmaps : N, W, H
# subplot_titles should be None if no names are to be used
def create_subplots(all_heatmaps, image, subplot_titles, row_titles, col_titles):
    print(type(image))
    num_heatmaps, y_shape, x_shape = all_heatmaps.shape
    num_col = 2
    num_row = math.ceil(num_heatmaps / num_col)
    
    all_specs = [[{} for _ in range(num_col)] for _ in range(num_row)]
    fig = make_subplots(rows=num_row, cols=num_col,
                        specs = all_specs,
                        vertical_spacing=0.15/num_row, #TODO: Fix hardcoding
                        subplot_titles=subplot_titles,
                        row_titles=row_titles, 
                        column_titles=col_titles)
    for i in range(num_heatmaps):
        r = math.floor(i / num_col) + 1
        c = (i % num_col) + 1
        
        fig.add_layout_image(dict(
                  source=image, 
                  xref=f"x{i + 1}",
                  yref=f"y{i + 1}",
                  x=0, 
                  y=y_shape,
                  sizex=x_shape,
                  sizey=y_shape, 
                  sizing='stretch',
                  layer="below"), row=r, col=c)
        fig.add_trace(go.Heatmap(z=all_heatmaps[i, :, :], 
                                  opacity=0.6, 
                                  colorbar=dict(len=(1/num_row), 
                                                yanchor="bottom")), 
                       row=r, col=c)
        fig['layout'][f"xaxis{i + 1}"]['showgrid'] = False
        fig['layout'][f"yaxis{i + 1}"]['showgrid'] = False

    fig.update_layout(height=(400 * num_row)) # TODO: Don't want a hardcoding here
    return fig