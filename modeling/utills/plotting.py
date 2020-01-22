# Imports from external libraries
import numpy as np
import plotly.graph_objects as go
from umap import UMAP
import matplotlib.pyplot as plt
import math
# Imports from internal libraries

def plot_MFCC(mfcc:np.ndarray,time,filename):
    fig = go.Figure()
    fig.add_trace(go.Heatmap(x=list(range(mfcc.shape[0])), y=list(range(mfcc.shape[1])), z=mfcc.T))
    fig.write_image(filename+".png")

_embedding_colors_ = np.array([
    [0, 127, 70],
    [255, 0, 0],
    [255, 217, 38],
    [0, 135, 255],
    [165, 0, 165],
    [255, 167, 255],
    [97, 142, 151],
    [0, 255, 255],
    [255, 96, 38],
    [142, 76, 0],
    [33, 0, 127],
    [0, 0, 0],
    [183, 183, 183],
    [76, 255, 0],
    [15, 224, 4],
    [255,25,179]
], dtype=np.float) / 255


def plot_projections(embeds, speakers, ax=None, colors=None, markers=None, legend=True,
                     title="", **kwargs):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    reducer = UMAP(**kwargs)

    projs = reducer.fit_transform(embeds)

    speakers = np.array(speakers)
    colors = colors or _embedding_colors_
    for i, speaker in enumerate(np.unique(speakers)):
        speaker_projs = projs[speakers == speaker]
        marker = "o" if markers is None else markers[i]
        label = speaker if legend else None
        ax.scatter(*speaker_projs.T, c=[colors[i]], marker=marker, label=label)

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.show()
    return projs