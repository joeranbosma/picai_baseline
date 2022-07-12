#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import torch.nn as nn
from nnunet.training.loss_functions.focal_loss import FocalLossV2
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2


class nnUNetTrainerV2_focalLossAlpha75(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        print("Setting up FocalLoss(alpha=[0.75, 0.25], apply_nonlin=nn.Softmax())")
        self.loss = FocalLossV2(alpha=[0.75, 0.25], apply_nonlin=nn.Softmax())


class nnUNetTrainerV2_focalLossAlpha75_checkpoints(nnUNetTrainerV2_focalLossAlpha75):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        print("Saving checkpoint every 50th epoch")
        self.save_latest_only = False


class nnUNetTrainerV2_focalLossAlpha75_checkpoints2(nnUNetTrainerV2_focalLossAlpha75_checkpoints):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        pass  # this is just to get a new Trainer directory


class nnUNetTrainerV2_focalLossAlpha75_checkpoints3(nnUNetTrainerV2_focalLossAlpha75_checkpoints):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        pass  # this is just to get a new Trainer directory
