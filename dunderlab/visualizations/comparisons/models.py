import numpy as np
import matplotlib.pyplot as plt
import matplotlib


# ----------------------------------------------------------------------
def multi_layer_polar_plot(
    data1: np.ndarray,
    data2: np.ndarray = None,
    group_labels: list[str] = [],
    external_labels: list[str] = [],
    internal_labels: list[str] = [],
    groups_space1: float = 0.3,
    groups_space2: float = 0.3,
    cmap1: list[str] = ['cool'],
    cmap2: list[str] = ['viridis'],
    limits: list[float] = [0.4, 0.5],
    space: float = 0.1,
    theta1: float = 0,
    theta2: float = 0,
    grid: bool = True,
    rlabel_position: int = 30
) -> plt.Axes:
    """
    Plots data on a polar coordinate system with various customization options.

    Parameters
    ----------
    data1 : np.ndarray
        The primary dataset for the plot.
    data2 : np.ndarray, optional
        The secondary dataset for the plot.
    group_labels : list[str], optional
        Labels for different groups in the dataset.
    external_labels : list[str], optional
        Labels to be placed externally on the plot.
    internal_labels : list[str], optional
        Labels to be placed internally on the plot.
    groups_space1 : float, optional
        Space between groups in the primary dataset.
    groups_space2 : float, optional
        Space between groups in the secondary dataset.
    cmap1 : list[str], optional
        Colormap for the primary dataset.
    cmap2 : list[str], optional
        Colormap for the secondary dataset.
    limits : list[float], optional
        Limits for the radial axis.
    space : float, optional
        Space between different sections of the plot.
    theta1 : float, optional
        Initial angle for the primary dataset.
    theta2 : float, optional
        Initial angle for the secondary dataset.
    grid : bool, optional
        Flag to display grid.
    rlabel_position : int, optional
        Position of the radial labels.

    Returns
    -------
    plt.Axes
        The matplotlib Axes object with the plot.
    """

    theta1 = theta1 * np.pi / 180
    data1_ = data1 - np.min(data1)
    data1_ = data1_ / np.sum(data1_, axis=2).max()
    data1_ = data1_ * (1 - limits[1])

    if data2 is not None:
        theta2 = theta2 * np.pi / 180
        data2_ = data2 - np.min(data2)
        data2_ = data2_ / np.sum(data2_, axis=2).max()
        data2_ = data2_ * (limits[1] - limits[0])

    plt.figure(figsize=(15, 15), dpi=60)
    ax = plt.subplot(111, projection='polar')

    colores_cualitativos = ['Pastel1', 'Pastel2', 'Paired', 'Accent',
                            'Dark2', 'Set1', 'Set2', 'Set3', 'tab10',
                            'tab20', 'tab20b', 'tab20c']

    ancho_por_barra = (2 * np.pi - groups_space1 * len(data1_)) / sum(len(v) for v in data1_)

    cmap1 = np.array(cmap1)

    for j, vectores in enumerate(data1_):

        ancho_grupo = ancho_por_barra * len(vectores)
        theta = np.linspace(theta1, theta1 + ancho_grupo, len(vectores), endpoint=False)

        cmap1_ = matplotlib.colormaps.get_cmap(cmap1[0])
        cmap1 = np.roll(cmap1, 1)

        if cmap1_.name in colores_cualitativos:
            colores = cmap1_(range(len(vectores[0])))[::-1]

        else:
            colores = cmap1_(np.linspace(0, 1, len(vectores[0])))

        for i, v in enumerate(vectores):
            if len(v) == 1:
                colores = cmap1_(np.linspace(0, 1, len(data1_)))[j]
            plt.bar(theta[i], np.cumsum(v)[::-1], color=colores, align='edge', width=ancho_por_barra * (1), bottom=limits[1] * (1 + space))

            if external_labels:
                rot = theta[i].mean() * (180 / np.pi) % 360
                if 270 > rot > 90:
                    rot = rot - 180
                plt.text(theta[i] + ancho_por_barra / 2, limits[1] + v.sum() + space * 1.3, external_labels[i], fontdict={'fontsize': 13}, rotation=rot, horizontalalignment='center', verticalalignment='center')

        plt.plot(np.linspace(theta[0], theta[-1] + ancho_por_barra, 100), np.ones(100) * limits[1] / (1 - space / 1.5), color='k', linewidth=2)
        theta1 += ancho_grupo + groups_space1

        if group_labels:
            rot = (theta.mean() * (180 / np.pi)) - 90
            rot = rot % 360
            if (rot + 90) % 360 > 180:
                rot = rot - 180
            plt.text(np.mean([theta[0], theta[-1] + ancho_por_barra]), limits[1], group_labels[j], fontdict={'fontsize': 15}, rotation=rot, horizontalalignment='center', verticalalignment='center')

    if data2 is not None:

        ancho_por_barra = (2 * np.pi - groups_space2 * len(data2_)) / sum(len(v) for v in data2_)

        for j, vectores in enumerate(data2_):

            ancho_grupo = ancho_por_barra * len(vectores)
            theta = np.linspace(theta2, theta2 + ancho_grupo, len(vectores), endpoint=False)

            cmap2_ = matplotlib.colormaps.get_cmap(cmap2[0])
            cmap2 = np.roll(cmap2, 1)

            if cmap1_.name in colores_cualitativos:
                colores = cmap2_(range(len(vectores[0])))[::-1]

            else:
                colores = cmap2_(np.linspace(0, 1, len(vectores[0])))

            for i, v in enumerate(vectores):
                if len(v) == 1:
                    colores = cmap2_(np.linspace(0, 1, len(data2_)))[j]
                plt.bar(theta[i], np.cumsum(v)[::-1], color=colores, align='edge', width=ancho_por_barra * (1), bottom=limits[1] / (1 + space) - np.cumsum(v)[::-1])

                if internal_labels:
                    rot = theta[i].mean() * (180 / np.pi) % 360
                    if 270 > rot > 90:
                        rot = rot - 180
                    plt.text(theta[i] + ancho_por_barra / 2, limits[1] - v.sum() - space * 1.3, internal_labels[i], fontdict={'fontsize': 13}, rotation=rot + 10, horizontalalignment='center', verticalalignment='center')

            theta2 += ancho_grupo + groups_space1

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_axisbelow(True)

    if grid:
        ax.tick_params(axis='x', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

        if data2 is not None:
            haf1 = np.linspace(limits[0] - space / 2, limits[1] * (1 - space), 3)
        else:
            haf1 = []
        haf2 = np.linspace(limits[1] * (1 + space), 1 + space / 2, 5)

        ax.set_yticks(np.concatenate([haf1, haf2]))
        haf2 = np.linspace(0, data1.max(), 5)

        if data2 is not None:
            haf1 = np.linspace(data2.max(), 0, 3)
        else:
            haf2 = []

        if rlabel_position is not False:
            ax.set_yticklabels([f'{h:.2f}' for h in np.concatenate([haf1, haf2])])
            ax.set_rlabel_position(rlabel_position)

            ax.set_xticks((np.linspace(0 + rlabel_position, 360 + rlabel_position, data1.shape[0], endpoint=False) % 360) * (np.pi / 180))

        else:
            ax.set_yticklabels([])
            ax.set_xticks([])

    else:
        ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

    plt.grid(grid, which='major', axis='both')

    return ax
