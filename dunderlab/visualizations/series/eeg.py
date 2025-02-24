import numpy as np
import matplotlib
from matplotlib import pyplot as plt


# ----------------------------------------------------------------------
def plot_eeg(data, channels, sample_rate, cmap='tab20', nxlabel=10, sca=1, offset=0, fig=None, ax=None, size=(15, 5), dpi=90):
    """"""

    if fig is None:
        plt.figure(figsize=size, dpi=dpi)

    if ax is None:
        ax = plt.subplot(111)

    t = np.linspace(0, len(data[0]) / sample_rate, len(data[0]))

    data_m = data / data.max()
    for i, ch in enumerate(data_m):
        plt.plot(t, i + (ch - ch.mean()) * sca,
                 color=matplotlib.cm.get_cmap(cmap, len(data))(i))

    plt.yticks(range(len(data)), channels)

    tx = np.linspace(0, len(data[0]) / sample_rate, nxlabel + 1)

    plt.xticks(tx, [f'{s.round(2)}s' for s in tx])

    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)

    plt.xlim(offset, len(data[0]) / sample_rate)
    plt.ylim(-1, len(data))

    plt.xlabel('Time [s]')
    plt.ylabel('Channels')

    return plt.gcf()
