import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, nibabel as nib, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.measure import block_reduce
import visualize_degradations as V

V.set_theme(True)  # fond clair
case = "PARSE_0001"
img = nib.load(f"nnUNet_data/nnUNet_raw/Dataset100_PARSE/labelsTr/{case}.nii.gz")
gt = np.asarray(img.get_fdata()) > 0
spacing = tuple(float(s) for s in img.header.get_zooms()[:3])
ds = 2
sl = V._bbox(gt, margin=4)
g = block_reduce(gt[sl], (ds, ds, ds), np.max)
sp = tuple(s * ds for s in spacing)

fig = plt.figure(figsize=(7.5, 7.5), facecolor=V._BG)
ax = fig.add_subplot(111, projection="3d")
ax.set_facecolor(V._BG)
V._add_mesh(ax, g, sp, (0.80, 0.16, 0.18), alpha=0.97)  # crimson artériel, solide
ax.set_box_aspect(np.array(g.shape) * np.array(sp))
ax.view_init(elev=18, azim=-60)
ax.set_axis_off()
fig.suptitle(f"GT* : clean annotation ({case})", color=V._TEXT,
             fontsize=15, fontweight="bold", y=0.95)
fig.tight_layout()
out = "report/figures/gt_clean_3D.png"
fig.savefig(out, dpi=220, bbox_inches="tight", facecolor=V._BG)
print("saved", out, os.path.getsize(out) // 1024, "Ko")
