"""Frozen passive results: five method rows, paired Llama and Qwen axes.
Only this figure is generated; no normalization or scientific evaluation.
"""
from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
from data_io import load
from visual_language import LLAMA, QWEN, GRAY

ROOT = Path(__file__).resolve().parent

# Data definition: explicit native units and existing, untransformed ranges.
METHODS = (
    ('LLMPrint', 'Bit accuracy', (-.04, 1.055)),
    ('REEF', 'Centered linear\nCKA', (-.04, 1.055)),
    ('HuRef', 'ICS', (-15, 105)),
    ('AWM', 'Mean Wq/Wk\nCKA', (-.04, 1.055)),
    ('ZeroPrint', 'Rescaled\ncorrelation', (-.04, 1.055)),
)
LINEAGES = (
    ('Llama', ('clean_llama', 'same_lineage_ba', 'same_lineage_bb', 'same_lineage_bc'),
     ('C', 'Ba', 'Bb*', 'Bc'), LLAMA, 'o'),
    ('Qwen', ('clean_qwen', 'cross_lineage_ba', 'cross_lineage_bb'),
     ('C', 'XBa', 'XBb*'), QWEN, 's'),
)

# Style configuration, sized for the manuscript's 5.5-inch text width.
FIGURE_SIZE = (5.5, 6.5)
PNG_DPI = 500
STYLE = {
    'font.family': 'DejaVu Sans', 'font.size': 8,
    'axes.titlesize': 9, 'axes.titleweight': 'bold',
    'axes.labelsize': 7.5, 'xtick.labelsize': 8, 'ytick.labelsize': 7.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .7, 'axes.edgecolor': '#59616A',
    'axes.facecolor': '#FCFCFD', 'figure.facecolor': 'white',
    'pdf.fonttype': 42, 'svg.fonttype': 'none',
}

@dataclass(frozen=True)
class Panel:
    method: str
    unit: str
    limits: tuple
    lineage: str
    fields: tuple
    labels: tuple
    values: tuple
    threshold: float
    color: str
    marker: str


def read_panels():
    """Load exact frozen values and thresholds without rounding or transformation."""
    source = {row['method']: row for row in load('native_detector_results.json')}
    rules = load('detector_rules.json')
    panels = []
    for method, unit, limits in METHODS:
        for lineage, fields, labels, color, marker in LINEAGES:
            panels.append(Panel(method, unit, limits, lineage, fields, labels,
                                tuple(source[method][field] for field in fields),
                                rules['thresholds'][method], color, marker))
    assert sum(len(panel.values) for panel in panels) == 35
    return panels


def create_layout():
    """Each method occupies one complete row with two equally sized panels."""
    fig, axes = plt.subplots(5, 2, figsize=FIGURE_SIZE, squeeze=False)
    fig.subplots_adjust(left=.125, right=.975, bottom=.075, top=.925,
                        wspace=.48, hspace=.80)
    for x, label, color in [(0.297, 'Same-lineage Llama', LLAMA),
                            (0.803, 'Cross-lineage Qwen', QWEN)]:
        fig.text(x, .976, label, ha='center', va='top', fontsize=9,
                 weight='bold', color=color)
    fig.text(.5, .018,
             'C = clean initialization   |   Dashed line = frozen threshold',
             ha='center', fontsize=8, color='#46505A')
    return fig, axes


def plot_panel(ax, panel):
    positions = range(len(panel.values))
    ax.set_axisbelow(True)
    ax.grid(axis='y', which='major', color='#DDE2E8', linewidth=.5)
    ax.grid(axis='y', which='minor', color='#EDF0F3', linewidth=.3)
    ax.grid(axis='x', which='major', color='#EDF0F3', linewidth=.35)
    threshold = ax.axhline(panel.threshold, color=GRAY, linewidth=.9,
                          linestyle=(0, (4, 2.5)), zorder=2)
    dots = ax.scatter(positions, panel.values, s=32, marker=panel.marker,
                      color=panel.color, edgecolors='white', linewidths=.5, zorder=3)
    ax.set(xlim=(-.45, len(panel.values)-.55), ylim=panel.limits,
           ylabel=panel.unit)
    ax.set_title(f'{panel.method} ({panel.lineage})', loc='left', pad=6)
    ax.set_xticks(list(positions), panel.labels, rotation=0)
    ax.set_yticks([0, 25, 50, 75, 100] if panel.method == 'HuRef'
                 else [0, .25, .5, .75, 1])
    minor_interval = 12.5 if panel.method == 'HuRef' else .05
    ax.yaxis.set_minor_locator(MultipleLocator(minor_interval))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
    ax.tick_params(length=3, width=.65, pad=3)
    ax.tick_params(axis='y', which='minor', length=1.8, width=.45,
                   color='#929AA3')
    ax.yaxis.labelpad = 6
    # A fixed empty score band avoids both the observations and threshold line.
    ax.legend([threshold], ['Frozen threshold'], loc='center right',
              bbox_to_anchor=(.98, .28), fontsize=6.5, handlelength=1.8,
              handletextpad=.5, borderpad=.25, borderaxespad=0,
              frameon=True, facecolor='#FCFCFD', edgecolor='none',
              framealpha=.95, labelcolor=GRAY)
    # Check actual artists, not just the arguments passed to matplotlib.
    assert tuple(dots.get_offsets()[:, 1]) == panel.values
    assert tuple(threshold.get_ydata()) == (panel.threshold, panel.threshold)
    assert tuple(ax.get_ylim()) == panel.limits
    return {'method': panel.method, 'lineage': panel.lineage,
            'fields': panel.fields, 'labels': panel.labels, 'values': panel.values,
            'threshold': panel.threshold, 'native_ylim': panel.limits,
            'major_y_ticks': [float(tick) for tick in ax.get_yticks()], 'minor_y_interval': minor_interval}


def export_figure(fig, output_dir, audit):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ('pdf',):
        fig.savefig(output_dir / f'passive_lineage.{extension}', dpi=PNG_DPI,
                    facecolor='white')



def generate(output_dir):
    panels = read_panels()
    with plt.rc_context(STYLE):
        fig, axes = create_layout()
        audit = [plot_panel(ax, panel) for ax, panel in zip(axes.flat, panels)]
        assert len(fig.axes) == 10
        export_figure(fig, output_dir, audit)
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'figures')
    generate(parser.parse_args().output_dir)
