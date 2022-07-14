import argparse
from pathlib import Path

from picai_baseline.splits.picai_nnunet import train_splits
from picai_eval import evaluate_folder
from picai_prep.preprocessing import crop_or_pad
from report_guided_annotation import extract_lesion_candidates


def nnUNet_cropped_prediction_compatibility(img):
    return crop_or_pad(img, size=(20, 320, 320))


def extract_lesion_candidates_compatibility(pred):
    return nnUNet_cropped_prediction_compatibility(extract_lesion_candidates(pred)[0])


parser = argparse.ArgumentParser()
parser.add_argument('-t', '--task', type=str, default="Task2201_picai_baseline")
parser.add_argument('--trainer', type=str, default='nnUNetTrainerV2_Loss_FL_and_CE_checkpoints')
parser.add_argument('--results', type=str, default="/workdir/results/nnUNet/3d_fullres/")
parser.add_argument('--checkpoints', type=str, nargs="+", default="model_best")
parser.add_argument('-f', '--folds', type=int, nargs="+", default=(0, 1, 2, 3, 4))
parser.add_argument('--softmax_dir_prefix', type=str, default="picai_pubtrain_predictions_")
args = parser.parse_args()

results_dir = Path(args.results)
trainer = (args.trainer + "__nnUNetPlansv2.1")
task = args.task

for fold in args.folds:
    print(f"Fold {fold}")

    for checkpoint in args.checkpoints:
        softmax_dir = results_dir / task / trainer / f"fold_{fold}/{args.softmax_dir_prefix}{checkpoint}"
        if not softmax_dir.exists():
            softmax_dir = results_dir / task / trainer / f"{args.softmax_dir_prefix}{checkpoint}_f{fold}"
        if not softmax_dir.exists():
            raise ValueError(f"Could not find softmax directory {softmax_dir}")
        else:
            print(f"Found softmax directory with {len(list(softmax_dir.glob('*.*')))} files")
        metrics_path = softmax_dir.parent  / f"metrics-train-{checkpoint}.json"

        if metrics_path.exists():
            print(f"Metrics found at {metrics_path}, skipping..")
            continue

        # evaluate
        metrics = evaluate_folder(
            y_det_dir=softmax_dir,
            y_true_dir=f"/workdir/nnUNet_raw_data/{task}/labelsTr",
            subject_list=train_splits[fold]['subject_list'],
            y_det_postprocess_func=extract_lesion_candidates_compatibility,
            y_true_postprocess_func=nnUNet_cropped_prediction_compatibility,
        )

        # save and show metrics
        metrics.save(metrics_path)
        print(f"Results for checkpoint {checkpoint}:")
        print(metrics)
