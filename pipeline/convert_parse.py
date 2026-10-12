#!/usr/bin/env python3
"""Convert the PARSE dataset to the nnU-Net v2 raw format.

Target layout:

    Dataset100_PARSE/
    ├── dataset.json
    ├── imagesTr/   PARSE_0001_0000.nii.gz, ...   (suffix _0000 = channel 0, CT)
    ├── labelsTr/   PARSE_0001.nii.gz, ...
    ├── imagesTs/   PARSE_0051_0000.nii.gz, ...
    └── labelsTs/   (test labels, kept for manual evaluation)

Two source layouts are supported:
    1. ``data_dir/{train,val,test}/{images,labels}/`` (train + val -> imagesTr, test -> imagesTs)
    2. ``data_dir/{images,labels}/`` (flat, everything -> imagesTr)

Images must be NIfTI (.nii.gz). Cases are renumbered PARSE_0001, PARSE_0002, ... in
sorted order.

    python pipeline/convert_parse.py --data_dir /path/to/PARSE --dataset_id 100
"""

import argparse
import glob
import json
import os
import shutil


def _find_nifti(directory: str) -> list[str]:
    """Sorted list of the .nii.gz files in ``directory``."""
    pattern = os.path.join(directory, "*.nii.gz")
    return sorted(glob.glob(pattern))


def _copy_case(
    img_src: str,
    lbl_src: str,
    images_dst: str,
    labels_dst: str,
    case_id: str,
) -> None:
    """Copy one image/label pair under the nnU-Net names of ``case_id`` (e.g. PARSE_0001)."""
    img_dst = os.path.join(images_dst, f"{case_id}_0000.nii.gz")   # single CT channel
    lbl_dst = os.path.join(labels_dst, f"{case_id}.nii.gz")

    shutil.copy2(img_src, img_dst)
    shutil.copy2(lbl_src, lbl_dst)


def convert_parse_to_nnunet(
    data_dir: str,
    dataset_id: int = 100,
    nnunet_raw: str | None = None,
) -> str:
    """Convert PARSE to ``<nnunet_raw>/Dataset{dataset_id:03d}_PARSE`` and return that path.

    ``nnunet_raw`` defaults to the ``nnUNet_raw`` environment variable. Raises ValueError
    if no image is found or if images and labels do not pair up.
    """
    if nnunet_raw is None:
        nnunet_raw = os.environ.get("nnUNet_raw")
        if nnunet_raw is None:
            raise EnvironmentError(
                "The nnUNet_raw environment variable is not set. "
                "Run setup_env.sh or export it."
            )

    dataset_name = f"Dataset{dataset_id:03d}_PARSE"
    dataset_dir = os.path.join(nnunet_raw, dataset_name)

    images_tr = os.path.join(dataset_dir, "imagesTr")
    labels_tr = os.path.join(dataset_dir, "labelsTr")
    images_ts = os.path.join(dataset_dir, "imagesTs")

    os.makedirs(images_tr, exist_ok=True)
    os.makedirs(labels_tr, exist_ok=True)
    os.makedirs(images_ts, exist_ok=True)

    has_splits = os.path.isdir(os.path.join(data_dir, "train", "images"))

    case_counter = 1

    if has_splits:
        print("Source layout: train/val/test subdirectories")

        # train + val -> imagesTr / labelsTr (nnU-Net runs its own cross-validation)
        for split in ["train", "val"]:
            split_img_dir = os.path.join(data_dir, split, "images")
            split_lbl_dir = os.path.join(data_dir, split, "labels")

            if not os.path.isdir(split_img_dir):
                print(f"  [SKIP] {split_img_dir} does not exist")
                continue

            imgs = _find_nifti(split_img_dir)
            lbls = _find_nifti(split_lbl_dir)

            if len(imgs) != len(lbls):
                raise ValueError(
                    f"Mismatch in {split}: {len(imgs)} images vs "
                    f"{len(lbls)} labels"
                )

            for img, lbl in zip(imgs, lbls):
                case_id = f"PARSE_{case_counter:04d}"
                _copy_case(img, lbl, images_tr, labels_tr, case_id)
                case_counter += 1

            print(f"  {split}: {len(imgs)} cases copied")

        num_training = case_counter - 1

        # test -> imagesTs
        test_img_dir = os.path.join(data_dir, "test", "images")
        if os.path.isdir(test_img_dir):
            test_imgs = _find_nifti(test_img_dir)
            for img in test_imgs:
                case_id = f"PARSE_{case_counter:04d}"
                dst = os.path.join(images_ts, f"{case_id}_0000.nii.gz")
                shutil.copy2(img, dst)
                case_counter += 1
            print(f"  test: {len(test_imgs)} cases copied to imagesTs")

            # test labels are copied too, for manual evaluation
            test_lbl_dir = os.path.join(data_dir, "test", "labels")
            if os.path.isdir(test_lbl_dir):
                labels_ts = os.path.join(dataset_dir, "labelsTs")
                os.makedirs(labels_ts, exist_ok=True)
                test_lbls = _find_nifti(test_lbl_dir)
                ts_counter = num_training + 1
                for lbl in test_lbls:
                    case_id = f"PARSE_{ts_counter:04d}"
                    dst = os.path.join(labels_ts, f"{case_id}.nii.gz")
                    shutil.copy2(lbl, dst)
                    ts_counter += 1
                print(f"  test labels: {len(test_lbls)} cases copied to labelsTs")
    else:
        print("Source layout: flat (images/ + labels/)")
        imgs = _find_nifti(os.path.join(data_dir, "images"))
        lbls = _find_nifti(os.path.join(data_dir, "labels"))

        if len(imgs) == 0:
            raise ValueError(f"No image found in {data_dir}/images")
        if len(imgs) != len(lbls):
            raise ValueError(
                f"Mismatch: {len(imgs)} images vs {len(lbls)} labels"
            )

        for img, lbl in zip(imgs, lbls):
            case_id = f"PARSE_{case_counter:04d}"
            _copy_case(img, lbl, images_tr, labels_tr, case_id)
            case_counter += 1

        num_training = case_counter - 1
        print(f"  {num_training} cases copied to imagesTr/labelsTr")

    dataset_json = {
        "channel_names": {
            "0": "CT",
        },
        "labels": {
            "background": 0,
            "pulmonary_artery": 1,
        },
        "numTraining": num_training,
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO",
    }

    json_path = os.path.join(dataset_dir, "dataset.json")
    with open(json_path, "w") as f:
        json.dump(dataset_json, f, indent=4)

    print(f"\n{'='*60}")
    print(f"nnU-Net dataset written: {dataset_dir}")
    print(f"  training cases : {num_training}")
    print(f"  dataset.json   : {json_path}")
    print(f"{'='*60}")

    return dataset_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert PARSE to the nnU-Net v2 raw format"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="source PARSE folder",
    )
    parser.add_argument(
        "--dataset_id",
        type=int,
        default=100,
        help="nnU-Net dataset ID (default: 100)",
    )
    parser.add_argument(
        "--nnunet_raw",
        type=str,
        default=None,
        help="nnUNet_raw folder (default: environment variable)",
    )
    args = parser.parse_args()

    convert_parse_to_nnunet(
        data_dir=args.data_dir,
        dataset_id=args.dataset_id,
        nnunet_raw=args.nnunet_raw,
    )
