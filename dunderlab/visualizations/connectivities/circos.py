from locale import normalize
from time import process_time_ns

import numpy as np
import matplotlib
from dunderlab.visualizations.connectivities import pycircos
import mne
from matplotlib import pyplot as plt



########################################################################
class CircosConnectivity:
    """"""
    Garc = pycircos.Garc
    Gcircle = pycircos.Gcircle

    # ----------------------------------------------------------------------
    def __init__(self, connectivities, channels, areas,

                 small_separation=5, big_separation=20,
                 labelsize=10, show_emisphere=True, threshold=0, limit_connections=-1, percentile=None, normalize_colors=True,
                 areas_cmap='viridis', arcs_cmap='viridis', size=10,

                 width={'hemispheres': 35, 'areas': 100, 'channels': 60},
                 text={'hemispheres': 40, 'areas': 20, 'channels': 40},
                 separation={'hemispheres': 10, 'areas': -30, 'channels': 5},
                 labelposition={'hemispheres': 60,
                                'areas': 0, 'channels': -10},

                 arcs_separation_src=30,
                 arcs_separation_dst=30,
                 hemisphere_color='C6', channel_color='#c5c5c5', connection_width=1,
                 offset=0, drop_channels=False,
                 fig=None,
                 vmin=None, vmax=None,

                 arrowhead_width=0.07,
                 arrowhead_length=50,
                 arrow_max_width=10,


                 ):
        self.params_ = dict(

            # Threshold and filtering
            threshold=threshold,
            limit_connections=limit_connections,
            percentile=percentile,
            normalize_colors=normalize_colors,

            # Colormaps and color styling
            areas_cmap=areas_cmap,
            arcs_cmap=arcs_cmap,
            hemisphere_color=hemisphere_color,
            channel_color=channel_color,

            # Label widths and font sizes per layer
            width=width,
            text=text,
            separation=separation,
            labelposition=labelposition,
            size=size,
            labelsize=labelsize,

            # Shape layout and structural configuration
            show_emisphere=show_emisphere,
            connection_width=connection_width,
            small_separation=small_separation,
            big_separation=big_separation,
            offset=offset,

            # Arc connection parameters
            arcs_separation_src=arcs_separation_src,
            arcs_separation_dst=arcs_separation_dst,

            # Arrowhead styling
            arrowhead_width=arrowhead_width,
            arrowhead_length=arrowhead_length,
            arrow_max_width=arrow_max_width

        )


        self.areas = areas
        self.arc_c = 0
        self.labelsize = labelsize
        self.areas_cmap = areas_cmap
        self.arcs_cmap = arcs_cmap
        self.width = width
        self.text = text
        self.separation = separation
        self.labelposition = labelposition
        self.arcs_separation_src = arcs_separation_src
        self.arcs_separation_dst = arcs_separation_dst
        self.hemisphere_color = hemisphere_color
        self.channel_color = channel_color
        self.connection_width = connection_width
        self.offset = offset
        self.drop_channels = drop_channels
        self.channels = channels
        self.vlim = (vmin, vmax)
        # self.markersize = markersize

        self.arrowhead_width = arrowhead_width
        self.arrowhead_length = arrowhead_length
        self.arrow_max_width = arrow_max_width


        if offset != 0:
            show_emisphere = False


        electrodes = sum([len(self.areas[k]) for k in self.areas])

        self.circle_ = self.Gcircle(fig, figsize=(size, size))

        self.small_separation = small_separation

        if show_emisphere:
            self.big_separation = big_separation
        else:
            self.big_separation = small_separation

        self.smin = (360 - ((len(self.areas) - 2)
                     * self.small_separation + (2 * self.big_separation))) / electrodes

        if show_emisphere:
            self.arcs = ['hemispheres', 'areas', 'channels']
        else:
            self.arcs = ['areas', 'channels']

        for i, arc in enumerate(self.arcs, start=1):
            getattr(self, f'arc_{arc}')(level=i)
        self.draw_arcs()

        connectivities = np.asarray(connectivities)
        c_min = connectivities.min()
        c_max = connectivities.max()

        if c_max == 0:
            raise ValueError("Cannot normalize: maximum connectivity value is zero.")

        # Normalize to [0, 1] after offsetting by minimum
        connectivities = (connectivities - c_min) / (c_max-c_min)

        if len(connectivities.shape) == 2:
            self.directional = True

        else:
            self.directional = False
            self.arrowhead_length = 0
            self.arrowhead_width = 0


        self.connectivity(connectivities, threshold, limit_connections, percentile, normalize_colors)


    @property
    def params(self):
        return self.params_

    # ----------------------------------------------------------------------
    def get_level(self, level_i):
        """"""
        level = self.arcs[level_i - 1]

        p = 1000 - \
            ((self.width[level] + self.text[level] +
             self.separation[level]) * level_i)
        return [p - self.width[level], p]

    # ----------------------------------------------------------------------
    def arc_areas(self, level=2):
        """"""
        i = 0
        for area in self.areas:
            i += 1

            if i % (len(self.areas) / 2):
                sep = self.small_separation
            else:
                sep = self.big_separation

            s = self.smin * len(self.areas[area])

            arc = self.Garc(arc_id=area.replace('_', ' '),
                            facecolor=matplotlib.cm.get_cmap(
                                self.areas_cmap, len(self.areas))(i - 1),
                            edgecolor=matplotlib.cm.get_cmap(
                                self.areas_cmap, len(self.areas))(i - 1),
                            size=s, interspace=sep,
                            raxis_range=self.get_level(level), labelposition=self.labelposition[self.arcs[level - 1]],
                            label_visible=True, labelsize=self.labelsize)
            self.circle_.add_garc(arc)

        self.arc_c += 1

    # ----------------------------------------------------------------------
    def arc_channels(self, level=3):
        """"""
        i = 0
        # self.channels = []

        for area in self.areas:
            i += 1

            if i % (len(self.areas) / 2):
                sepe = 0
            else:
                sepe = self.big_separation

            for j, e in enumerate(self.areas[area], start=1):

                if j != len(self.areas[area]):
                    sep = 0
                else:
                    sep = self.small_separation
                    if sepe:
                        sep = sepe

                arc = self.Garc(arc_id=f'{e}', facecolor=self.channel_color, size=self.smin, interspace=sep, raxis_range=self.get_level(
                    level), labelposition=self.labelposition[self.arcs[level - 1]], label_visible=True, labelsize=self.labelsize)
                self.circle_.add_garc(arc)

                # self.channels.append(f'{e}')

        self.arc_c += 1

    # ----------------------------------------------------------------------
    def arc_hemispheres(self, level=1):
        """"""
        arc = self.Garc(arc_id='Right Hemisphere', facecolor=self.hemisphere_color,
                        edgecolor=self.hemisphere_color, size=180 - self.big_separation,
                        interspace=self.big_separation, raxis_range=self.get_level(
                            level),
                        labelposition=self.labelposition[self.arcs[level - 1]], label_visible=True, labelsize=self.labelsize,
                        )
        self.circle_.add_garc(arc)

        arc = self.Garc(arc_id='Left Hemisphere', facecolor=self.hemisphere_color,
                        edgecolor=self.hemisphere_color, size=180 - self.big_separation,
                        interspace=self.big_separation, raxis_range=self.get_level(
                            level),
                        labelposition=self.labelposition[self.arcs[level - 1]], label_visible=True, labelsize=self.labelsize,
                        )
        self.circle_.add_garc(arc)
        self.arc_c += 1

    # ----------------------------------------------------------------------
    def draw_arcs(self):
        """"""
        o = self.offset * (self.smin + self.big_separation / 2)
        self.circle_.set_garcs((self.big_separation / 2) + o,
                               (360 * self.arc_c) + (self.big_separation / 2) + o)

    # ----------------------------------------------------------------------
    def format_connectivities(self, connectivities):
        """"""
        if len(connectivities.shape) == 1:  # vector
            n = len(self.channels)
            tri = np.zeros((n, n))
            tri[np.triu_indices(n, k=1)] = connectivities
            return tri

        return connectivities

    # ----------------------------------------------------------------------
    def connectivity(self, connectivities, threshold, limit_connections, percentile, normalize_colors):
        """"""
        def map_(x, in_min, in_max, out_min, out_max):
            return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

        connectivities = self.format_connectivities(connectivities)

        chords = []
        for i, j in zip(*np.where(~np.eye(connectivities.shape[0], dtype=bool))):

            if connectivities[i][j] < threshold:
                continue

            if percentile:
                p_min, p_max = np.percentile(connectivities[connectivities > threshold], percentile)
                if not (p_min <= connectivities[i][j] <= p_max):
                    continue

            if i == j:
                continue

            if connectivities[i][j] == 0:
                continue

            kk = map_(connectivities[i][j], threshold, connectivities[connectivities != 1].max(
            ), 1, self.arrow_max_width)


            if not self.directional:
                w1 = (self.smin / 2) - kk * self.connection_width
                w2 = (self.smin / 2) + kk * self.connection_width
                w3=w1
                w4=w2
            else:

                w1 = (self.smin / 4) - kk * self.connection_width
                w2 = (self.smin / 4) + kk * self.connection_width

                w3 = 3*(self.smin / 4) - kk * self.connection_width
                w4 = 3*(self.smin / 4) + kk * self.connection_width


            x1, _ = self.get_level(self.arc_c)


            source = (self.channels[i], w1, w2, x1 - self.arcs_separation_src)
            destination = (self.channels[j], w3, w4, x1 - self.arcs_separation_dst)

            chords.append([connectivities[i][j], source, destination])

        if self.vlim[0]:
            norm = matplotlib.colors.Normalize(
                vmin=self.vlim[0], vmax=self.vlim[1])

        elif not normalize_colors:
            vmin = connectivities.min()
            vmax = connectivities.max()
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

        else:
            if threshold:
                vmin = threshold
            else:
                vmin = min([c[0] for c in sorted(chords)[::-1][:limit_connections]])

            if limit_connections != -1:
                vmax = max([c[0] for c in sorted(chords)[::-1][:limit_connections]])
            else:
                vmax = connectivities[connectivities > threshold].max()

            if percentile:
                vmin, vmax = np.percentile(connectivities[connectivities > threshold], percentile)

            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)


        for v_, src, des in sorted(chords)[-limit_connections:]:
            self.circle_.chord_plot(des, src,
                                    facecolor=matplotlib.pyplot.cm.get_cmap(
                                        self.arcs_cmap)(norm(v_)),
                                    edgecolor=matplotlib.pyplot.cm.get_cmap(
                                        self.arcs_cmap)(norm(v_)),
                                    linewidth=1,
                                    arrowhead_width=self.arrowhead_width, arrowhead_length=self.arrowhead_length
                                    )

    # ----------------------------------------------------------------------
    @property
    def figure(self):
        """"""
        return self.circle_.figure

    # ----------------------------------------------------------------------
    @property
    def circle(self):
        """"""
        return self.circle_

    # ----------------------------------------------------------------------
    def topoplot_reference(self, montage_name='standard_1005', ax=None, size=10, fontsize=18, markersize=30, markerfacecolor='#ffffff00', markeredgecolor='#000000'):
        """"""
        info = mne.create_info(
            self.channels,
            sfreq=1,
            ch_types="eeg",
        )
        info.set_montage(montage_name)

        data = []
        for ch in self.channels:
            for area in self.areas:
                if ch in self.areas[area]:
                    data.append(list(self.areas.keys()).index(area))
                    continue

        if ax is None:
            plt.figure(figsize=(size, size))
            ax = matplotlib.pyplot.subplot(111)

        matplotlib.rcParams['font.size'] = fontsize
        mne.viz.plot_topomap(data,
                             info,
                             axes=ax,
                             names=self.channels,
                             sensors=True,
                             # show_names=True,
                             contours=0,
                             cmap=self.areas_cmap,
                             # outlines='skirt',
                             res=64,
                             extrapolate='head',
                             show=False,
                             image_interp='nearest',

                             mask_params=dict(
                                 marker='o',
                                 markerfacecolor=markerfacecolor,
                                 markeredgecolor=markeredgecolor,
                                 linewidth=0,
                                 markersize=markersize,
                                 zorder=3,
                             ),
                             mask=np.array([True] * len(self.channels)),

                             )

        return ax





def interact_connectivity(connectivities, channels, areas, offset_=0):
    import ipywidgets as widgets
    from IPython.display import display
    from ipywidgets import interactive

    connectivity_plot = [None]

    @widgets.interact(
        threshold=widgets.FloatSlider(min=0.0, max=1.0, step=0.1, value=0),
        limit_connections=widgets.IntSlider(min=-1, max=len(connectivities), step=1, value=-1),

        percentile=widgets.IntRangeSlider(min=0, max=100, step=1, value=[25, 75]),
        normalize_colors=widgets.Checkbox(value=True),

        areas_cmap=widgets.Dropdown(options=sorted(m for m in plt.colormaps() if not m.endswith('_r')), value='Set3'),
        arcs_cmap=widgets.Dropdown(options=sorted(m for m in plt.colormaps() if not m.endswith('_r')), value='Wistia'),
        hemisphere_color=widgets.ColorPicker(value='lightgray'),
        channel_color=widgets.ColorPicker(value='#f8f9fa'),

        width_hemispheres=widgets.IntSlider(min=-100, max=100, step=1, value=35),
        width_areas=widgets.IntSlider(min=-100, max=100, step=1, value=100),
        width_channels=widgets.IntSlider(min=-100, max=100, step=1, value=60),

        text_hemispheres=widgets.IntSlider(min=-100, max=100, step=1, value=40),
        text_areas=widgets.IntSlider(min=-100, max=100, step=1, value=20),
        text_channels=widgets.IntSlider(min=-100, max=100, step=1, value=40),

        separation_hemispheres=widgets.IntSlider(min=-100, max=100, step=1, value=10),
        separation_areas=widgets.IntSlider(min=-100, max=100, step=1, value=-30),
        separation_channels=widgets.IntSlider(min=-100, max=100, step=1, value=5),

        labelposition_hemispheres=widgets.IntSlider(min=-100, max=100, step=1, value=60),
        labelposition_areas=widgets.IntSlider(min=-100, max=100, step=1, value=0),
        labelposition_channels=widgets.IntSlider(min=-100, max=100, step=1, value=-10),

        size=widgets.IntSlider(min=-100, max=100, step=1, value=10),
        labelsize=widgets.IntSlider(min=-100, max=100, step=1, value=15),

        show_emisphere=widgets.Checkbox(value=True),
        connection_width=widgets.FloatSlider(min=0.01, max=1.0, step=0.01, value=0.1),
        small_separation=widgets.IntSlider(min=0, max=100, step=1, value=5),
        big_separation=widgets.IntSlider(min=0, max=100, step=1, value=10),
        offset=widgets.IntSlider(min=-10, max=10, step=1, value=offset_),

        arcs_separation_src=widgets.IntSlider(min=0, max=300, step=1, value=30),
        arcs_separation_dst=widgets.IntSlider(min=0, max=300, step=1, value=30),

        arrowhead_width=widgets.FloatSlider(min=0.0, max=1.0, step=0.01, value=0.05),
        arrowhead_length=widgets.IntSlider(min=0, max=100, step=1, value=30),
        arrow_max_width=widgets.IntSlider(min=0, max=100, step=1, value=10),
    )
    def update_circos(threshold=0, limit_connections=-1, percentile=[25, 75], normalize_colors=True,
                      areas_cmap='Set3', arcs_cmap='Wistia',
                      hemisphere_color='lightgray', channel_color='#f8f9fa',
                      width_hemispheres=35, width_areas=100, width_channels=60,
                      text_hemispheres=40, text_areas=20, text_channels=40,
                      separation_hemispheres=10, separation_areas=-30, separation_channels=5,
                      labelposition_hemispheres=60, labelposition_areas=0, labelposition_channels=-10,
                      size=10, labelsize=15,
                      show_emisphere=True,
                      connection_width=0.1,
                      small_separation=5,
                      big_separation=10,
                      offset=offset_,
                      arcs_separation_src=30,
                      arcs_separation_dst=30,
                      arrowhead_width=0.05,
                      arrowhead_length=30,
                      arrow_max_width=10,
                      ):

        width = {
            'hemispheres': width_hemispheres,
            'areas': width_areas,
            'channels': width_channels,
        }
        text = {
            'hemispheres': text_hemispheres,
            'areas': text_areas,
            'channels': text_channels,
        }
        separation = {
            'hemispheres': separation_hemispheres,
            'areas': separation_areas,
            'channels': separation_channels,
        }
        labelposition = {
            'hemispheres': labelposition_hemispheres,
            'areas': labelposition_areas,
            'channels': labelposition_channels,
        }

        percentile_range = list(percentile)
        if percentile_range[0] == 0 and percentile_range[1] == 100:
            percentile_range = None

        connectivity_plot[0] = CircosConnectivity(
            connectivities, channels, areas,

            # Threshold and filtering
            threshold=threshold,
            limit_connections=limit_connections,
            percentile=percentile_range,
            normalize_colors=normalize_colors,

            # Colormaps and color styling
            areas_cmap=areas_cmap,
            arcs_cmap=arcs_cmap,
            hemisphere_color=hemisphere_color,
            channel_color=channel_color,

            # Label widths and font sizes per layer
            width=width,
            text=text,
            separation=separation,
            labelposition=labelposition,
            size=size,
            labelsize=labelsize,

            # Shape layout and structural configuration
            show_emisphere=show_emisphere,
            connection_width=connection_width,
            small_separation=small_separation,
            big_separation=big_separation,
            offset=offset,

            # Arc connection parameters
            arcs_separation_src=arcs_separation_src,
            arcs_separation_dst=arcs_separation_dst,

            # Arrowhead styling
            arrowhead_width=arrowhead_width,
            arrowhead_length=arrowhead_length,
            arrow_max_width=arrow_max_width,
        )


    return lambda :connectivity_plot[0]