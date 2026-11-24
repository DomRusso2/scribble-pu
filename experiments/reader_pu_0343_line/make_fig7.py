"""Figure 7: Reader V3 vs Reader-PU (reader-pu-bhand.pth) on the awarded PHerc. 0343 First Letters line (Erwin Nieuwlaar).
Inputs = the two forward reads written by read_0343_line.py (team inference path, no training). The read is cropped to the
text rows and turned 180 degrees (a rotation, not a mirror) so the line reads upright; same 0-255 display window for both.
  python make_fig7.py <read dir> <out png>"""
import sys

import matplotlib
import numpy as np
import tifffile

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

D, OUT = sys.argv[1], sys.argv[2]


def img(tag):
    a = tifffile.imread(f'{D}/{tag}_forward.tif').astype(np.float32)[300:2200, :]
    return np.clip((np.rot90(a, 2) - 50) / 180, 0, 1)


BG, HI, ACC = '#111418', '#f2f2f2', '#ffb347'
fig = plt.figure(figsize=(16.5, 10.0), facecolor=BG)
fig.text(0.02, 0.952, 'The awarded PHerc. 0343 First Letters line (Erwin Nieuwlaar): Reader V3, then V3 retuned with ScribblePU',
         color=HI, fontsize=17, fontweight='bold')
fig.text(0.02, 0.912, 'NEITHER MODEL WAS TRAINED ON THIS SEGMENT. Nothing was traced on it, and the ScribblePU retune used no '
                      'PHerc. 0343 data at all.', color=ACC, fontsize=13.5, fontweight='bold')
for i, (tag, lab, col) in enumerate((('v3', 'Reader V3', '#9aa3ad'),
                                     ('pu_bhand', 'Reader V3 + ScribblePU (Reader-PU): thinner strokes, letters apart, '
                                                  'upper row legible', ACC))):
    ax = fig.add_axes([0.02, 0.47 - i * 0.445, 0.96, 0.405])
    ax.imshow(img(tag), cmap='gray', vmin=0, vmax=1, interpolation='lanczos', aspect='auto')
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(col)
        s.set_linewidth(2.5 if i else 1.2)
    fig.text(0.02, 0.882 - i * 0.445, lab, color=col, fontsize=13, fontweight='bold')
fig.savefig(OUT, dpi=130, facecolor=BG)
print('saved', OUT)
