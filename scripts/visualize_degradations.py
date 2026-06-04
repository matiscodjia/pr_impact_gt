
import os
import glob
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, binary_erosion, binary_opening, generate_binary_structure


def _ball(radius):
    r = int(radius)
    zz, yy, xx = np.ogrid[-r:r + 1, -r:r + 1, -r:r + 1]
    return (zz * zz + yy * yy + xx * xx) <= r * r

def apply_dilation(segmentation, radius=3):
    result = segmentation.copy()
    for c in range(result.shape[0]):
        mask = result[c] > 0.5
        struct = generate_binary_structure(mask.ndim, 1)
        degraded = binary_dilation(mask, structure=struct, iterations=radius)
        result[c] = degraded.astype(result.dtype)
    return result

def apply_erosion(segmentation, radius=1):
    result = segmentation.copy()
    for c in range(result.shape[0]):
        mask = result[c] > 0.5
        struct = generate_binary_structure(mask.ndim, 1)
        degraded = binary_erosion(mask, structure=struct, iterations=radius)
        result[c] = degraded.astype(result.dtype)
    return result

def apply_aggressive_omission(segmentation, radius=2):
    """Supprime les vaisseaux fins par ouverture morphologique (rayon `radius`)."""
    result = segmentation.copy()
    struct = _ball(radius)
    for c in range(result.shape[0]):
        mask = result[c] > 0.5
        if not mask.any(): continue
        opened = binary_opening(mask, structure=struct)
        result[c] = opened.astype(result.dtype)
    return result

def process_file(input_path, output_dir, file_id):
    img = nib.load(input_path)
    data = img.get_fdata()
    if len(data.shape) == 3: data = data[np.newaxis, ...]

    # Trouver la meilleure coupe
    sums = np.sum(data[0], axis=(0, 1))
    z_slice = np.argmax(sums)
    
    # Crop auto
    slice_2d_orig = data[0, :, :, z_slice]
    coords = np.argwhere(slice_2d_orig > 0)
    if coords.size > 0:
        x_min, y_min = coords.min(axis=0); x_max, y_max = coords.max(axis=0)
        m = 30
        x_min = max(0, x_min-m); y_min = max(0, y_min-m)
        x_max = min(data.shape[1], x_max+m); y_max = min(data.shape[2], y_max+m)
    else: return

    original = data.copy()
    etirement = apply_dilation(original, radius=3)
    rognage = apply_erosion(original, radius=1)
    omissions = apply_aggressive_omission(original, radius=3)

    fig, axes = plt.subplots(1, 4, figsize=(24, 7), facecolor='#f8f9fa')
    titles = ["Original (GT*)", "Étirement (Dilation)", "Rognage (Érosion)", "Omissions"]
    images = [original, etirement, rognage, omissions]
    
    for i, (ax, img_data, title) in enumerate(zip(axes, images, titles)):
        crop = img_data[0, x_min:x_max, y_min:y_max, z_slice]
        ax.imshow(crop.T, cmap='magma', interpolation='nearest')
        ax.set_title(title, fontsize=20, fontweight='bold', pad=20)
        ax.axis('off')
        
    plt.suptitle(f"Patient {file_id} - Analyse des dégradations", fontsize=24, y=1.05)
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"demo_degradations_{file_id}.png")
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Généré : {out_path}")

def main():
    output_dir = "result/figures/candidates"
    os.makedirs(output_dir, exist_ok=True)
    
    files = sorted(glob.glob("nnUNet_data/nnUNet_raw/Dataset100_PARSE/labelsTr/PARSE_00*.nii.gz"))[:10]
    
    for f in files:
        file_id = os.path.basename(f).replace(".nii.gz", "")
        process_file(f, output_dir, file_id)

if __name__ == "__main__":
    main()
