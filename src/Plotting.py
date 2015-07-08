
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import BrokenBarHCollection
import numpy as np

DEBUG = False

RED = [1., .3, .3],
GREEN = [.3, .8, .34]
BLUE = [.3, .4, 1.]

PaperParam = {
    'red': [1., .3, .3],
    'green': [.3, .8, .34],
    'blue': [.3, .4, 1.],
    'linewidth': 3.,
    'titleSize': 16,
    'labelSize': 14
}

DefaultPlotParams = {
    'title': "",
    'xlabel': "",
    'ylabels': [],
    'forPaper': False,
    'color': "blue",
    'majorY': 0.,
    'linewidth': -1
}


def makeBold(string):
    ''' Use LaTeX to make the string bold for plots. '''
    return string
    return r"\begin{center}{\bfseries %s}\end{center}" % string


def plotTo(ax, dataX, dataY=None, linspec='b', params={}):
    ''' Plot to the axes with style.
    Make sure the desired ylabel in `params` is the first element in a list.
    '''
    if dataY is None:
        dataY = dataX
        N = np.max(dataX.shape)
        dataX = np.arange(N)

    # Set the default values. TODO: Methodise this boilerplate
    for key in DefaultPlotParams:
        if key not in params: params[key] = DefaultPlotParams[key]

    if params['title']:
        title = makeBold(params['title'])
        ax.set_title(title, fontweight="bold", fontsize=PaperParam['titleSize'])

    # Set X and Y labels
    ax.set_ylabel(params['ylabels'][0], fontsize=PaperParam['labelSize'])
    if params['xlabel']:
        ax.set_xlabel(params['xlabel'], fontsize=PaperParam['labelSize'])
    # Tick font size:
    ax.tick_params(axis='both', which='both', direction='out',
                   labelsize=PaperParam['labelSize'])

    # Draw the line
    linewidth = params['linewidth']
    if linewidth <= 0:
        linewidth = PaperParam['linewidth'] if params['forPaper'] else 1.
    plot = ax.plot(dataX, dataY, linspec, linewidth=linewidth, color=PaperParam[params['color']])

    # Grid
    if params['majorY'] > 0:
        yloc = plt.MultipleLocator(params['majorY'])
        ax.yaxis.set_major_locator(yloc)
    ax.grid(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return plot


def plot3Axes(dataX, data=None, linspec='b', params={}, show=True):
    ''' Plot 3 axes in the data in 3 subplots.

    If the data is 1D, this assumes that the 3 axes are concatenated.
    :param data: 1D numpy array.
    '''
    return plotAxes(dataX, data, linspec, params, show, nAxes=3)


def plotAxes(dataX, data=None, linspec='b', params={}, show=True, nAxes=3):
    ''' Plot axes in the data in subplots.

    If the data is 1D, this assumes that the axes are concatenated.
    :param data: 1D or 2D numpy array.
    :param params: Dictionary of strings containing title, axis labels, and
        other parameters for plotting:
        * "title": Plot title
        * "xlabel": Label for the x data
        * "ylabels": List of labels for the y datas
        * "forPaper": Plot using lines and colors good for a paper
        * "color": "red", "green", or "blue" only
    '''
    # Set the default values
    for key in DefaultPlotParams:
        if key in params: continue
        params[key] = DefaultPlotParams[key]

    if data is None:
        data = dataX
        N = np.max(dataX.shape)
        if len(data.shape) == 1:
            N /= nAxes
        dataX = np.arange(N)

    if len(data.shape) == 1:
        data = np.reshape(data, (nAxes,-1))

    if data.shape[1] == nAxes:
        data = data.T

    defaultXtickPad = mpl.rcParams['xtick.major.pad']

    if nAxes == 3:
        mpl.rcParams['xtick.major.pad'] = 0

    plt.gcf().subplots_adjust(hspace=0.5)

    firstAx = None
    axes = []
    for i in range(nAxes):
        if i == 0:
            firstAx = plt.subplot(nAxes,1, i+1)
            ax = firstAx
            if params['title']:
                firstAx.set_title(makeBold(params['title']), fontweight="bold",
                                  fontsize=PaperParam['titleSize'])
        else:
            ax = plt.subplot(nAxes,1, i+1, sharex=firstAx)

        # Return the list of axes at the end
        axes.append(ax)

        # Set X and Y labels
        if i < len(params['ylabels']):
            ax.set_ylabel(params['ylabels'][i], fontsize=PaperParam['labelSize'])
        if i == nAxes-1:  # Last axis
            ax.set_xlabel(params['xlabel'], fontsize=PaperParam['labelSize'])
        # Tick font size:
        ax.tick_params(axis='both', which='major', direction='out',
                       labelsize=PaperParam['labelSize'])


        # Draw the line
        linewidth = params['linewidth']
        if linewidth <= 0:
            linewidth = PaperParam['linewidth'] if params['forPaper'] else 1.
        ax.plot(dataX, data[i], linspec,
                 linewidth=linewidth, color=PaperParam[params['color']])

        # Grid
        if params['majorY'] > 0:
            yloc = plt.MultipleLocator(params['majorY'])
            ax.yaxis.set_major_locator(yloc)
        ax.grid(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    if show: plt.show()
    mpl.rcParams['xtick.major.pad'] = defaultXtickPad

    return axes


def highlight3Axes(dataX, mask=None, facecolor='green', alpha=0.5, show=True):
    if mask is None:
        mask = dataX
        N = np.max(dataX.shape)
        if len(mask.shape) == 1:
            N /= 3
        dataX = np.arange(N)

    if len(mask.shape) == 1:
        mask = np.reshape(mask, (3,-1))

    if mask.shape[1] == 3:
        mask = mask.T

    firstAx = None
    for i in range(3):
        if i == 0:
            firstAx = plt.subplot(3,1,i+1)
            ax = firstAx
        else:
            ax = plt.subplot(3,1,i+1, sharex=firstAx)

        span = BrokenBarHCollection.span_where(dataX, -100, 100, mask[i],
            facecolor=facecolor, alpha=alpha)
        ax.add_collection(span)

    if show:
        plt.show()
