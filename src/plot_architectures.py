"""
plot_architectures.py — horizontal block diagrams of the two classifier
networks defined in src/model.py (ClassifierMLP, Classifier3DCNN).

Each figure is a left-to-right row of rounded colour-coded boxes, each with
a bold layer label and a grey output-dimension caption, a centred title,
and a small footnote. Parameter counts in the footnotes are those of
src/model.py at N = 200 ROIs (Keras count_params()).

Run:  python src/plot_architectures.py [--out-dir DIR]
Out:  results/figures/classifier_mlp_architecture.png
      results/figures/classifier_3dcnn_architecture.png
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# colour by layer role (fill, edge) — soft pastels, darker matching edge
PALETTE = {
    'input':     ('#CDE9C6', '#7CB97A'),   # green
    'dense':     ('#FFE0B2', '#E8A33D'),   # orange  (hidden Dense)
    'batchnorm': ('#BBDEFB', '#5B9BD5'),   # blue
    'dropout':   ('#F8BBD0', '#D87BA0'),   # pink
    'conv':      ('#B2DFDB', '#4DB6AC'),   # teal
    'pool':      ('#CFD8DC', '#90A4AE'),   # blue-grey
    'flatten':   ('#ECEFF1', '#B0BEC5'),   # neutral grey
    'output':    ('#E1BEE7', '#AB6FBF'),   # purple
}


def _draw(layers, title, footnote, out_path, output_note):
    """layers: list of dicts {main, dim, type}. Render one row of boxes."""
    n = len(layers)
    fig_w = max(8.0, 1.55 * n)
    fig, ax = plt.subplots(figsize=(fig_w, 3.1))
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.axis('off')

    box_w, box_h = 0.80, 0.42
    y0 = 0.30                      # box bottom
    yc = y0 + box_h / 2            # box centre

    for i, L in enumerate(layers):
        fill, edge = PALETTE[L['type']]
        x0 = i + (1 - box_w) / 2
        ax.add_patch(FancyBboxPatch(
            (x0, y0), box_w, box_h,
            boxstyle='round,pad=0.012,rounding_size=0.06',
            linewidth=1.4, edgecolor=edge, facecolor=fill, zorder=2,
        ))
        ax.text(i + 0.5, yc, L['main'], ha='center', va='center',
                fontsize=8.5, fontweight='bold', color='#222222', zorder=3)
        if L.get('dim'):
            ax.text(i + 0.5, y0 - 0.075, L['dim'], ha='center', va='center',
                    fontsize=7.5, color='#888888', zorder=3)

        # connector arrow to next box
        if i < n - 1:
            ax.add_patch(FancyArrowPatch(
                (x0 + box_w, yc), (i + 1 + (1 - box_w) / 2, yc),
                arrowstyle='-|>', mutation_scale=10,
                linewidth=1.0, color='#BBBBBB', zorder=1,
            ))

    # output annotation above the last box
    ax.text(n - 0.5, y0 + box_h + 0.10, output_note, ha='center', va='bottom',
            fontsize=8, style='italic', color='#777777')

    ax.text(n / 2, 0.95, title, ha='center', va='center',
            fontsize=13, fontweight='bold', color='#1a1a1a')
    ax.text(0.04, 0.04, footnote, ha='left', va='bottom',
            fontsize=7.5, color='#888888')

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"wrote {out_path}")


def mlp_layers():
    return [
        {'main': 'Input',                       'dim': 'N·(N−1)/2', 'type': 'input'},
        {'main': 'Dense\n256, ELU\nL2(1e-4)',   'dim': '256',       'type': 'dense'},
        {'main': 'BatchNorm',                   'dim': '256',       'type': 'batchnorm'},
        {'main': 'Dropout\n0.3',                'dim': '256',       'type': 'dropout'},
        {'main': 'Dense\n64, ELU\nL2(1e-4)',    'dim': '64',        'type': 'dense'},
        {'main': 'BatchNorm',                   'dim': '64',        'type': 'batchnorm'},
        {'main': 'Dropout\n0.3',                'dim': '64',        'type': 'dropout'},
        {'main': 'Dense\n32, ELU',              'dim': '32',        'type': 'dense'},
        {'main': 'Dense\n1, Sigmoid',           'dim': '1',         'type': 'output'},
    ]


def cnn_layers():
    return [
        {'main': 'Input',                          'dim': '31×37×31×N', 'type': 'input'},
        {'main': 'Conv3D\n128, ELU\n3×3×3, same',  'dim': '128',        'type': 'conv'},
        {'main': 'Conv3D\n32, ELU\n3×3×3, same',   'dim': '32',         'type': 'conv'},
        {'main': 'MaxPool3D\n2×2×2, same',         'dim': '↓2',         'type': 'pool'},
        {'main': 'Flatten',                        'dim': '',           'type': 'flatten'},
        {'main': 'Dense\n32, ELU\nL2(0.005)',      'dim': '32',         'type': 'dense'},
        {'main': 'Dense\n1, Sigmoid',              'dim': '1',          'type': 'output'},
    ]


def main():
    default_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'figures')
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out-dir', default=default_dir,
                    help=f'Output directory (default: {default_dir})')
    fig_dir = ap.parse_args().out_dir
    _draw(
        mlp_layers(),
        title='MLP Architecture (ClassifierMLP)',
        footnote=('N = number of atlas ROIs; input = flattened upper triangle of the\n'
                  'Fisher-z connectivity matrix. 19,900 features and ~5.1 M parameters '
                  'at N = 200 (CC200)'),
        out_path=os.path.join(fig_dir, 'classifier_mlp_architecture.png'),
        output_note='Output: P(anxious)',
    )
    _draw(
        cnn_layers(),
        title='3D CNN Architecture (Classifier3DCNN)',
        footnote=('N = number of atlas ROIs; input = (31, 37, 31, N) voxel-wise\n'
                  'supervoxel fingerprint. ~5.8 M parameters at N = 200 (CC200)'),
        out_path=os.path.join(fig_dir, 'classifier_3dcnn_architecture.png'),
        output_note='Output: P(anxious)',
    )


if __name__ == '__main__':
    main()
