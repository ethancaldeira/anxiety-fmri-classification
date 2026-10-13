"""
generate_sp_masks.py — generate the stochastic (Poisson-disk) parcellations.

Pure MNI152 template operation: no subject data and no labels are used,
so the same SP_v2_{scale}_{sample}.nii.gz masks are valid for GSP, ABIDE,
or any dataset registered in MNI152NLin6ASym 3 mm space.

Backed by Schirmer's `pds` package (Bridson 2007 Poisson-disk sampling on
a voxel neighbourhood graph of each hemisphere,
https://github.com/mdschirmer/poisson_disk_sampling), which needs
graph-tool. graph-tool is distributed through conda-forge, so this script
runs in a separate conda environment with graph-tool and pds
(environment-pds.yml), not in the TensorFlow environment.

Outputs land alongside the standard atlases:
    src/templates/atlases/SP_v2_{scale:03d}_{sample:02d}.nii.gz

where compute_features.py finds them by name. The 120 masks used in the
thesis (4 scales × 30 samples) are included in the repository.

Run (in a conda environment with graph-tool and pds):

    python src/generate_sp_masks.py --scales 50 100 150 200 --n-samples 30

Notes
-----
* `pds.poisson_disk_sample` writes a `graph_skel_{left,right}.xml.gz`
  voxel-graph cache to CWD on first call per hemisphere. Subsequent calls
  reuse those — that's why we hold a single workdir across all samples
  (the graph is hemisphere-shape-derived, not random).
* We seed numpy *before each call* (sample index as seed) so SP_v2_050_07
  is bit-for-bit reproducible.
* `num_regions` must be even (the package halves it across hemispheres);
  it silently subtracts 1 from odd values, so we just pass even scales.
"""

import argparse
import os
import shutil
import sys
import tempfile

import numpy as np
import nibabel as nib

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GM_MASK   = os.path.join(REPO_ROOT, 'src', 'templates', 'masks', 'grey_matter_mask.nii.gz')
ATLAS_DIR = os.path.join(REPO_ROOT, 'src', 'templates', 'atlases')


def _split_hemispheres(gm_path, workdir):
    """Split a whole-brain GM mask into left + right hemisphere masks
    based on the world-x midline (MNI x ≤ 0 → left, x > 0 → right).
    Writes two NIfTI files to workdir and returns their paths."""
    img = nib.load(gm_path)
    data = img.get_fdata() > 0
    affine = img.affine

    # World x = affine @ [i, j, k, 1] — for each voxel i, world_x is determined.
    # MNI midline is world_x = 0. We split by computing world_x per i-index.
    nx = data.shape[0]
    world_x = affine[0, 0] * np.arange(nx) + affine[0, 3]
    left_i  = np.where(world_x <= 0)[0]   # MNI convention: negative x = left
    right_i = np.where(world_x >  0)[0]

    left  = np.zeros_like(data, dtype=np.uint8)
    right = np.zeros_like(data, dtype=np.uint8)
    left[left_i, :, :]   = data[left_i, :, :]
    right[right_i, :, :] = data[right_i, :, :]

    print(f"  GM mask: {data.shape}, total voxels: {data.sum()}")
    print(f"  L hemisphere: {left.sum()} voxels  ({len(left_i)} x-slices)")
    print(f"  R hemisphere: {right.sum()} voxels ({len(right_i)} x-slices)")

    l_path = os.path.join(workdir, 'gm_left.nii.gz')
    r_path = os.path.join(workdir, 'gm_right.nii.gz')
    nib.save(nib.Nifti1Image(left,  affine, img.header), l_path)
    nib.save(nib.Nifti1Image(right, affine, img.header), r_path)
    return l_path, r_path


def _generate_one(pds_module, l_path, r_path, scale, sample_idx, workdir, out_dir):
    """Generate one SP mask at the given scale + sample index. Seeded so
    SP_v2_{scale}_{sample} is reproducible."""
    final_name = f"SP_v2_{scale:03d}_{sample_idx:02d}.nii.gz"
    final_path = os.path.join(out_dir, final_name)
    if os.path.exists(final_path):
        print(f"  [skip] {final_name} already exists")
        return final_path

    np.random.seed(sample_idx)   # NB: pds uses np.random internally

    # Use a unique pickle per call so the package doesn't reuse cached
    # regions from a previous call (it only checks file existence).
    pickle_name = f"pds_{scale}_{sample_idx}.pickle"
    basename    = f"SP_v2_{scale:03d}_{sample_idx:02d}_"   # pds appends '{NNN}_{CCC}.nii.gz'
    pickle_path = os.path.join(workdir, pickle_name)
    if os.path.exists(pickle_path):
        os.remove(pickle_path)

    cwd = os.getcwd()
    os.chdir(workdir)   # graph_skel_*.xml.gz cache lives here
    try:
        pds_module.poisson_disk_sample(
            l_path, r_path, [scale],
            output_filename=pickle_name,
            create_niftii=True,
            output_basename=basename,
        )
    finally:
        os.chdir(cwd)

    # pds writes "<basename>{NNN}_{CCC}.nii.gz" — find it and move/rename.
    produced = [f for f in os.listdir(workdir)
                if f.startswith(basename) and f.endswith('.nii.gz')]
    if not produced:
        raise RuntimeError(f"pds produced no NIfTI for scale={scale} sample={sample_idx}")
    src = os.path.join(workdir, produced[0])
    shutil.move(src, final_path)
    print(f"  [ok]  {final_name}")
    return final_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scales',    type=int, nargs='+', default=[50, 100, 150, 200],
                    help='Parcellation sizes (must be even).')
    ap.add_argument('--n-samples', type=int, default=30,
                    help='Random SP samples per scale.')
    ap.add_argument('--gm-mask',   type=str, default=GM_MASK,
                    help='Path to whole-brain GM mask in MNI152 3mm.')
    ap.add_argument('--out-dir',   type=str, default=ATLAS_DIR,
                    help='Destination directory (default = src/templates/atlases/).')
    ap.add_argument('--workdir',   type=str, default=None,
                    help='Where intermediate pickle + graph cache live. '
                         'Default = a fresh tempdir, deleted at exit.')
    args = ap.parse_args()

    # Late import so --help works even if pds isn't on the path
    try:
        import pds  # noqa: F401
    except ImportError:
        sys.exit("ERROR: `pds` package not found. Run this script in a conda "
                 "environment with graph-tool and pds (see environment-pds.yml).")
    import pds as pds_module

    os.makedirs(args.out_dir, exist_ok=True)
    workdir_cm = (tempfile.TemporaryDirectory()
                  if args.workdir is None else None)
    workdir = workdir_cm.name if workdir_cm else args.workdir
    os.makedirs(workdir, exist_ok=True)

    try:
        print(f"GM mask:  {args.gm_mask}")
        print(f"Out dir:  {args.out_dir}")
        print(f"Workdir:  {workdir}\n")
        l_path, r_path = _split_hemispheres(args.gm_mask, workdir)

        n_total = len(args.scales) * args.n_samples
        n_done  = 0
        for scale in args.scales:
            if scale % 2:
                print(f"WARN: scale {scale} is odd — pds will subtract 1.")
            print(f"\n=== Scale {scale} ({args.n_samples} samples) ===")
            for s in range(1, args.n_samples + 1):
                _generate_one(pds_module, l_path, r_path, scale, s, workdir, args.out_dir)
                n_done += 1
                print(f"     ({n_done}/{n_total})")
    finally:
        if workdir_cm is not None:
            workdir_cm.cleanup()

    print(f"\nDONE. Wrote {n_total} SP masks to {args.out_dir}")


if __name__ == '__main__':
    main()
