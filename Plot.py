#!/usr/bin/python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches 



def plot_all(v, loc, s):

    keys = list(v.columns)
    print(keys)

    v = np.array( v ,dtype=float).T
    # v_main = np.argmax(v, axis=0)  # main type
    loc1 = loc['x']
    loc2 = loc['y']

    n_loc = len(loc1)
    n_type = len(keys)
    cmap = matplotlib.colormaps['rainbow']
    colormap = cmap(np.linspace(0, 1, n_type))

    v[v <= 0.01] = 0
    plt.subplot(1, 2, 1)
    plt.axis('off')
    plt.axis('equal')
    for i in range(n_loc):
        prob = 0
        # xy = []
        for k in range(n_type):
            if v[k, i] != 0:
                prob_next = prob + v[k, i]
                x = [0] + np.cos(np.linspace(2 * np.pi * prob, 2 * np.pi * prob_next, 10)).tolist()
                y = [0] + np.sin(np.linspace(2 * np.pi * prob, 2 * np.pi * prob_next, 10)).tolist()
                xy = list(zip(x, y))
                prob = prob_next
                plt.scatter(loc1[i], loc2[i], marker=(xy), s=s, facecolor=colormap[k])


    plt.subplot(1, 2, 2)
    patches = [mpatches.Patch(color=color, label=label) for label, color in zip(keys, colormap)]
    plt.legend(patches, keys, loc='center', frameon=False, fontsize='x-large')
    plt.axis('off')
    
    plt.show()


def plot_std(v, loc, s, c_map='Blues'):
    # v = np.clip(v, a_min=0, a_max=0.04)

    type_list = ['EPL-IN','GC', 'OSs', 'M/TC', 'PGC']

    n = len(type_list)
    loc1 = loc['x']
    loc2 = loc['y']

    # v[v<0.1]=0
    nrow = 1
    fig, axes = plt.subplots(nrows=nrow, ncols=int(np.ceil(n/nrow)))
    i=0
    for ax in axes.flat:
        if i==n:
            fig.delaxes(ax)
            break
        im = ax.scatter(loc1, loc2, s=s, c=v[type_list[i]], cmap=c_map, vmin=0, vmax=np.max(np.max(v)))
        # im = ax.scatter(loc1, loc2, s=s, c=v[type_list[i]], cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_title(type_list[i])
        # fig.colorbar(im, ax=ax)
        ax.axis('off')
        ax.axis('equal')
        i+=1


    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), location='bottom', shrink=0.25)

    plt.show()