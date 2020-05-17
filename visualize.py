import math

from plotly import graph_objects as go
from plotly.subplots import make_subplots

def create_subplots(cams, image):
    num_col = 2
    num_row = math.ceil(len(cams) / num_col)
    chosen_layers = list(cams.keys())
    all_specs = [[{} for _ in range(num_col)] for _ in range(num_row)]
    fig = make_subplots(rows=num_row, cols=num_col,
                        specs = all_specs,
                        vertical_spacing=0.01,
                        subplot_titles=chosen_layers)
    for i, key in enumerate(chosen_layers):
        r = math.floor(i / num_col) + 1
        c = (i % num_col) + 1

        fig.add_trace(go.Image(z=image), 
                      row=r, col=c)
        fig.add_trace(go.Heatmap(z=cams[key][0, :, :], 
                                 opacity=0.6, 
                                 colorbar=dict(len=(1/num_row), 
                                               yanchor="bottom")), 
                      row=r, col=c)
    
    fig.update_layout(title_text="Layer-Wise GradCAM", 
                      height=(500 * num_row))
    return fig