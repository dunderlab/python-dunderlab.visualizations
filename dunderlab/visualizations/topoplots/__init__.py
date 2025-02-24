import mne
from matplotlib import pyplot as plt


# ----------------------------------------------------------------------
def topoplot(data, channels, montage_name='standard_1020', ax=None, **kwargs):
    """"""
    info = mne.create_info(
        channels,
        sfreq=1,
        ch_types="eeg",
    )
    info.set_montage(montage_name)

    if ax is None:
        ax = plt.subplot(111)

    return mne.viz.plot_topomap(data,
                                info,
                                axes=ax,
                                **kwargs
                                )

