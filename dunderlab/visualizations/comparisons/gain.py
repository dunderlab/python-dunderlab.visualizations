import numpy as np
from matplotlib import pyplot as plt


# ----------------------------------------------------------------------
def accuracy_gain(method_1, method_2, ticks, labels, sort='method_1', reference_c='C1', gain_c='C0', loss_c='C3', barwidth=6, ylabel='Accuracy [%]', xlabel='Subjects', gain_labels=['gain', 'loss'], fig=None, ax=None, size=(15, 5), dpi=90, **kwargs):
    """"""
    if 'lose_c' in kwargs:
        loss_c = kwargs['lose_c']

    if fig is None:
        plt.figure(figsize=size, dpi=dpi)

    if ax is None:
        ax = plt.subplot(111)

    if sort is None:
        index = np.arange(len(method_2))
    elif sort == 'method_1':
        index = np.argsort(method_1)[::-1]
    elif sort == 'method_1r':
        index = np.argsort(method_1)
    elif sort == 'method_2':
        index = np.argsort(method_2)[::-1]
    elif sort == 'method_2r':
        index = np.argsort(method_2)
    else:
        index = sort

    colors = np.array(method_2[index]
                      - method_1[index] < 0, dtype=np.object_)

    if sort == 'method_2':
        p1, = plt.plot(method_2[index], color=reference_c, linestyle='--',)
        p2, = plt.plot(method_1[index], color=gain_c,
                       linestyle='--', alpha=0.3)
        colors[colors == 0] = loss_c
        colors[colors == 1] = gain_c
    else:
        p1, = plt.plot(method_1[index], color=reference_c, linestyle='--',)
        p2, = plt.plot(method_2[index], color=gain_c,
                       linestyle='--', alpha=0.3)
        colors[colors == 0] = gain_c
        colors[colors == 1] = loss_c

    plots = [p1, p2]

    if np.array([colors == gain_c]).any():
        p3 = plt.vlines(np.array(sorted(index))[colors == gain_c], method_1[index][colors == gain_c],
                        method_2[index][colors == gain_c], color=colors[colors == gain_c], linewidth=barwidth)
        labels.append(gain_labels[0])
        plots.append(p3)

    if np.array([colors == loss_c]).any():
        p4 = plt.vlines(np.array(sorted(index))[colors == loss_c], method_1[index][colors == loss_c],
                        method_2[index][colors == loss_c], color=colors[colors == loss_c], linewidth=barwidth)
        labels.append(gain_labels[1])
        plots.append(p4)

    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xticks(range(len(ticks)), ticks[index], rotation=90)

    try:
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
    except:
        pass

    try:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    except:
        pass

    l1 = plt.legend(plots, labels, loc='upper center',
                    ncol=2, bbox_to_anchor=(0.5, 1))
    plt.gca().add_artist(l1)

    return plt.gcf()
