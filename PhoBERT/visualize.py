import matplotlib.patches as mpatches

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt


def plot(x, y):
    x = TSNE(n_components=3, learning_rate='auto', perplexity=30).fit_transform(x)
    colors = ['red', 'pink', 'green']
    emotions = ['Positive', 'Negative', 'Neutral']
    handles = []
    for idx, color in enumerate(colors):
        handles.append(mpatches.Patch(color=color, label=emotions[idx]))
    y_colors = [colors[emotions.index(label.capitalize())] for label in y]
    fig = plt.figure(figsize=(20, 14))
    ax = plt.axes(projection='3d')
    ax.scatter3D(x[:,0], x[:,1], x[:,2], color=y_colors, s=300)
    plt.legend(handles=handles)
    plt.show()