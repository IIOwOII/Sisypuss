# Library
import matplotlib.pyplot as plt

class meowfig:
    def __init__(self, size=(4,3), dpi=300, design=True, **kwargs):
        """
        Parameters
        ----------
        size : tuple
            The default is (4,3).
        dpi : int
            The default is 300.
        **kwargs : optional
            title, xlabel, ylabel, xlim, ylim, xticks, yticks, xticklabels, yticklabels...
        """
        # essential
        self.size = size
        self.dpi = dpi
        self.design = design
        
        # optional
        self.title = kwargs.get('title')
        self.xlabel = kwargs.get('xlabel')
        self.ylabel = kwargs.get('ylabel')
        self.xlim = kwargs.get('xlim')
        self.ylim = kwargs.get('ylim')
        self.xticks = kwargs.get('xticks')
        self.yticks = kwargs.get('yticks')
        self.xticklabels = kwargs.get('xticklabels')
        self.yticklabels = kwargs.get('yticklabels')
        self.xoffset = kwargs.get('xoffset')
        self.yoffset = kwargs.get('yoffset')
        
        # figure setting
        self.fig, self.ax = plt.subplots(figsize=self.size, dpi=self.dpi)
        for side in ['right', 'top', 'bottom']:
            self.ax.spines[side].set_visible(False)
        self.optional_setting()
        self.ax.plot(1)
    
    # additional setting
    def optional_setting(self):
        self.ax.set_title(self.title)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        if (self.xlim != None):
            xlim = self.xlim
            xrange = xlim[1] - xlim[0]
            self.ax.set_xlim([xlim[0]-0.02*xrange, xlim[1]+0.02*xrange])
        if (self.ylim != None):
            ylim = self.ylim
            yrange = ylim[1] - ylim[0]
            self.ax.set_ylim([ylim[0]-0.02*yrange, ylim[1]+0.02*yrange])
        if (self.xticks != None): self.ax.set_xticks(self.xticks)
        if (self.yticks != None): self.ax.set_yticks(self.yticks)
        if (self.xticklabels != None): self.ax.set_xticklabels(self.xticklabels)
        if (self.yticklabels != None): self.ax.set_yticklabels(self.yticklabels)
        if (self.design): self.optional_design()
    
    # figure design
    def optional_design(self):
        if (self.yoffset != None):
            self.ax.axhline(self.yoffset, linewidth=0.6, linestyle='-', color='gray', zorder=-1)
        if (self.yticks != None):
            for ytick in self.yticks:
                self.ax.axhline(ytick, linewidth=0.3, linestyle='-.', color='gray', alpha=0.3, zorder=-1)

