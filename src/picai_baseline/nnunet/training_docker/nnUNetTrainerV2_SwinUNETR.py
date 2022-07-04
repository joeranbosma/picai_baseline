from nnunet.training.network_training.nnUNetTrainerV2_Loss_FL_and_CE import nnUNetTrainerV2_Loss_FL_and_CE_checkpoints

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
        
        print("Hello there!")
        print("="*100)
        print(self.network)
        print("="*100)
        print(self.patch_size)
        print("="*100)
        print(self.net_num_pool_op_kernel_sizes)
        print("="*100)
        print(self.net_conv_kernel_sizes)
        print("="*100)
