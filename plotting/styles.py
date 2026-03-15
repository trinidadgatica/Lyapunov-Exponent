"""
Plot styling configuration used across the project.

This file mirrors the original styling defined in
core.utils.plot_information so figures keep the exact same
appearance during the plotting refactor.
"""

from __future__ import annotations

import seaborn as sns

# Configuration for font sizes
X_LABEL_FONT_SIZE = 8
Y_LABEL_FONT_SIZE = 7
X_TICK_FONT_SIZE = 8
Y_TICK_FONT_SIZE = 8

# Configuration for plot dimensions (in inches)
PLOT_WIDTH = 4
PLOT_HEIGHT = 1.2

# Configuration for legend
LEGEND_SIZE = (1, 1)
LEGEND_POSITION = "upper left"
LEGEND_FONT_SIZE = 6
LEGEND_TITLE_FONT_SIZE = 6

# Set Seaborn style and color palette
sns.set(style="whitegrid")
color_palette = sns.color_palette("deep")

# Assign colors from the palette
primary_color = color_palette[0]
secondary_color = color_palette[1]
tertiary_color = color_palette[2]
quaternary_color = color_palette[3]
quinary_color = color_palette[4]
senary_color = color_palette[5]
septenary_color = color_palette[6]
octonary_color = color_palette[7]
nonary_color = color_palette[8]
denary_color = color_palette[9]

# List of colors for plots
colors = [
    primary_color,
    secondary_color,
    tertiary_color,
    quaternary_color,
    quinary_color,
    senary_color,
    septenary_color,
    octonary_color,
    nonary_color,
    denary_color,
]

# Extra constants needed by refactored map modules
DEFAULT_SAVE_DPI = 300
DEFAULT_MAP_LEVELS = 300
DEFAULT_D2_MAX = 2.5
DEFAULT_LYAPUNOV_LINTHRESH = 1e-3
DEFAULT_LYAPUNOV_NUM_DIVS = 4