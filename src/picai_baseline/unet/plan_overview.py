#  Copyright 2022 Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from picai_baseline.splits.picai_nnunet import nnunet_splits

# specify paths to preprocessed data in nnU-Net Raw Data Archive
preprocessed_data_path = Path('/workdir/nnUNet_raw_data/Task202_picai_swinunetr_anisotropic/')
overviews_path = Path('/workdir/results/SwinUNETR4/overviews/')
study_list_exclude = [
    "1001499"
]

# create directory to store overviews
overviews_path.mkdir(parents=True, exist_ok=True)

# iterate over each cross-validation fold
for fold, nnunet_fold in enumerate(nnunet_splits):

    # iterate over train and validation splits
    for split, nnunet_split in nnunet_fold.items():
        print(f"Preparing fold {fold}..")

        # initialize list of fields to collect for each split of each fold
        overview = {
            'pat_ids': [],
            'study_ids': [],
            'image_paths': [],
            'label_paths': [],
            'case_label': [],
            'ratio_csPCa_bg': []
        }

        # iterate over each training/validation case
        for subject_id in nnunet_split:
            patient_id, study_id = subject_id.split('_')

            # load annotation
            lbl = sitk.GetArrayFromImage(sitk.ReadImage(str(preprocessed_data_path / 'labelsTr' / f'{subject_id}.nii.gz')))

            overview['pat_ids'] += [patient_id]
            overview['study_ids'] += [study_id]
            overview['image_paths'] += [[
                str((preprocessed_data_path / 'imagesTr' / f'{subject_id}_0000.nii.gz').as_posix()),
                str((preprocessed_data_path / 'imagesTr' / f'{subject_id}_0001.nii.gz').as_posix()),
                str((preprocessed_data_path / 'imagesTr' / f'{subject_id}_0002.nii.gz').as_posix()),
            ]]
            overview['label_paths'] += [str((preprocessed_data_path / 'labelsTr' / f'{subject_id}.nii.gz').as_posix())]
            overview['case_label'] += [float(np.max(lbl))]
            overview['ratio_csPCa_bg'] += [float(np.sum(lbl)/np.size(lbl))]

        # save overview
        with open(overviews_path / f'PI-CAI_{split}-fold-{fold}.json', 'w') as fp:
            json.dump(overview, fp, indent=4)

print("Finished.")
