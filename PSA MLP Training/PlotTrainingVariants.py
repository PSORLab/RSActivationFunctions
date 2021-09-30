# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 10:39:43 2020

@author: wilhe
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator, ScalarFormatter, LogFormatterExponent,
                               LogFormatterSciNotation, NullFormatter)

import numpy as np
import pandas as pd

path = "C:\\Users\\wilhe\\Dropbox\\Apps\\Overleaf\\CACE 2020, Relaxations of Activation Functions\\Santa_Anna_2017_MP_Training\\AverageLoss.csv"
dataset = pd.read_csv(path, index_col= False, names = ["Depth", "Nonlinear Terms", "GeLU", "Swish1", "Tanh"])
dataset = dataset.sample(frac = 1)

fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=False, figsize=(6.73228, 1*3.36614) , dpi = 150)
fig.subplots_adjust(wspace=0.1,hspace=0.3)

plt.minorticks_off()

for v in [["GeLU","DarkBlue","s"],
          ["Swish1","Green", "d"],
          ["Tanh", "Red", "."]]:
    l1 = dataset[(dataset["Depth"] == 1) & (dataset["Nonlinear Terms"] <= 60)].plot.scatter(ax=axes[0,0], x="Nonlinear Terms", y=v[0], c=v[1], marker=v[2])
    l2 = dataset[(dataset["Depth"] == 2) & (dataset["Nonlinear Terms"] <= 60)].plot.scatter(ax=axes[0,1], x="Nonlinear Terms", y=v[0], c=v[1], marker=v[2])
    l3 = dataset[(dataset["Depth"] == 3) & (dataset["Nonlinear Terms"] <= 60)].plot.scatter(ax=axes[1,0], x="Nonlinear Terms", y=v[0], c=v[1], marker=v[2])
    l4 = dataset[(dataset["Depth"] == 4) & (dataset["Nonlinear Terms"] <= 60)].plot.scatter(ax=axes[1,1], x="Nonlinear Terms", y=v[0], c=v[1], marker=v[2])

#Specify font axis label 
label_font = {'fontname':'Times New Roman',
              'weight' : 'normal',
              'size'   : 12}

# Specify font for titles
title_font = {'fontname':'Times New Roman',
              'weight' : 'normal',
              'size'   : 12}

for ax_row in axes:
    for ax in ax_row:
        ax.set_yscale('log')
        ax.set_ylim(0.001, 0.01)
        ax.set_xlabel('Total Neurons', **label_font)
        ax.tick_params(axis ='y', which ='minor', length = 0)
        ax.set_yticklabels([])
        frmt = LogFormatterSciNotation()
        ax.get_yaxis().set_major_formatter(frmt)
        ax.get_yaxis().set_minor_formatter(NullFormatter())


axes[0,0].set_ylabel('min(MSE)', **label_font)
axes[1,0].set_ylabel('min(MSE)', **label_font)
axes[0,1].yaxis.set_visible(False)
axes[1,1].yaxis.set_visible(False)

axes[0,0].set_title('Hidden Layers = 1', **title_font)
axes[0,1].set_title('Hidden Layers = 2', **title_font)
axes[1,0].set_title('Hidden Layers = 3', **title_font)
axes[1,1].set_title('Hidden Layers = 4', **title_font)

axes[0,0].set_xlabel('Total Neurons', **label_font)
axes[0,1].set_xlabel('Total Neurons', **label_font)

# stores plot to file
out_put_file = 'PlotNeuronVsFit.pdf'
plt.savefig(out_put_file, format='pdf', dpi=1200, bbox_inches='tight')


