import torch
from nnunet.training.network_training.nnUNetTrainerV2_Loss_FL_and_CE import nnUNetTrainerV2_Loss_FL_and_CE_checkpoints
from nnunet.network_architecture.monai_swinunetr import SwinUNETR

class nnUNetTrainerV2_SwinUNETR(nnUNetTrainerV2_Loss_FL_and_CE_checkpoints):
    """
    Set network to SwinUNETR
    Set loss to FL + CE and set checkpoints
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)

    def initialize_network(self):
        """Initialize SwinUNETR network"""
        print("Overwriting network to SwinUNETR")

        self.net_num_pool_op_kernel_sizes = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]]
        print(f"Set downsampling sizes to {self.net_num_pool_op_kernel_sizes}")

        model = SwinUNETR(
            img_size=self.patch_size,  # e.g., [ 16 320 320]
            in_channels=self.num_input_channels,
            out_channels=self.num_classes,
            patch_size=self.net_num_pool_op_kernel_sizes,
            # depths: Sequence[int] = (2, 2, 2, 2),
            # num_heads: Sequence[int] = (3, 6, 12, 24),
            # feature_size: int = 24,
            # norm_name: Union[Tuple, str] = "instance",
            # drop_rate: float = 0.0,
            # attn_drop_rate: float = 0.0,
            # dropout_path_rate: float = 0.0,
            # normalize: bool = True,
            # use_checkpoint: bool = False,
            # spatial_dims: int = 3,
        )

        self.network = model
        self.network._deep_supervision = False
        self.network.do_ds = False

        if torch.cuda.is_available():
            self.network.cuda()
