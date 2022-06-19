SwinUNETR(
  (swinViT): SwinTransformer(
    (patch_embed): PatchEmbed(
      (proj): Conv3d(3, 24, kernel_size=(1, 2, 2), stride=(1, 2, 2))
    )
    (pos_drop): Dropout(p=0.0, inplace=False)
    (layers1): ModuleList(
      (0): BasicLayer(
        (blocks): ModuleList(
          (0): SwinTransformerBlock(
            (norm1): LayerNorm((24,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              (qkv): Linear(in_features=24, out_features=72, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=24, out_features=24, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): Identity()
            (norm2): LayerNorm((24,), eps=1e-05, elementwise_affine=True)
            (mlp): MLPBlock(
              (linear1): Linear(in_features=24, out_features=96, bias=True)
              (linear2): Linear(in_features=96, out_features=24, bias=True)
              (fn): GELU(approximate=none)
              (drop1): Dropout(p=0.0, inplace=False)
              (drop2): Dropout(p=0.0, inplace=False)
            )
          )
          (1): SwinTransformerBlock(
            (norm1): LayerNorm((24,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              (qkv): Linear(in_features=24, out_features=72, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=24, out_features=24, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): Identity()
            (norm2): LayerNorm((24,), eps=1e-05, elementwise_affine=True)
            (mlp): MLPBlock(
              (linear1): Linear(in_features=24, out_features=96, bias=True)
              (linear2): Linear(in_features=96, out_features=24, bias=True)
              (fn): GELU(approximate=none)
              (drop1): Dropout(p=0.0, inplace=False)
              (drop2): Dropout(p=0.0, inplace=False)
            )
          )
        )
        (downsample): PatchMerging(
          (reduction): Linear(in_features=96, out_features=48, bias=False)
          (norm): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
        )
      )
    )
    (layers2): ModuleList(
      (0): BasicLayer(
        (blocks): ModuleList(
          (0): SwinTransformerBlock(
            (norm1): LayerNorm((48,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              (qkv): Linear(in_features=48, out_features=144, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=48, out_features=48, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): Identity()
            (norm2): LayerNorm((48,), eps=1e-05, elementwise_affine=True)
            (mlp): MLPBlock(
              (linear1): Linear(in_features=48, out_features=192, bias=True)
              (linear2): Linear(in_features=192, out_features=48, bias=True)
              (fn): GELU(approximate=none)
              (drop1): Dropout(p=0.0, inplace=False)
              (drop2): Dropout(p=0.0, inplace=False)
            )
          )
          (1): SwinTransformerBlock(
            (norm1): LayerNorm((48,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              (qkv): Linear(in_features=48, out_features=144, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=48, out_features=48, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): Identity()
            (norm2): LayerNorm((48,), eps=1e-05, elementwise_affine=True)
            (mlp): MLPBlock(
              (linear1): Linear(in_features=48, out_features=192, bias=True)
              (linear2): Linear(in_features=192, out_features=48, bias=True)
              (fn): GELU(approximate=none)
              (drop1): Dropout(p=0.0, inplace=False)
              (drop2): Dropout(p=0.0, inplace=False)
            )
          )
        )
        (downsample): PatchMerging(
          (reduction): Linear(in_features=384, out_features=96, bias=False)
          (norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        )
      )
    )
    (layers3): ModuleList(
      (0): BasicLayer(
        (blocks): ModuleList(
          (0): SwinTransformerBlock(
            (norm1): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              (qkv): Linear(in_features=96, out_features=288, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=96, out_features=96, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): Identity()
            (norm2): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
            (mlp): MLPBlock(
              (linear1): Linear(in_features=96, out_features=384, bias=True)
              (linear2): Linear(in_features=384, out_features=96, bias=True)
              (fn): GELU(approximate=none)
              (drop1): Dropout(p=0.0, inplace=False)
              (drop2): Dropout(p=0.0, inplace=False)
            )
          )
          (1): SwinTransformerBlock(
            (norm1): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              (qkv): Linear(in_features=96, out_features=288, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=96, out_features=96, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): Identity()
            (norm2): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
            (mlp): MLPBlock(
              (linear1): Linear(in_features=96, out_features=384, bias=True)
              (linear2): Linear(in_features=384, out_features=96, bias=True)
              (fn): GELU(approximate=none)
              (drop1): Dropout(p=0.0, inplace=False)
              (drop2): Dropout(p=0.0, inplace=False)
            )
          )
        )
        (downsample): PatchMerging(
          (reduction): Linear(in_features=768, out_features=192, bias=False)
          (norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
      )
    )
    (layers4): ModuleList(
      (0): BasicLayer(
        (blocks): ModuleList(
          (0): SwinTransformerBlock(
            (norm1): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              (qkv): Linear(in_features=192, out_features=576, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=192, out_features=192, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): Identity()
            (norm2): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
            (mlp): MLPBlock(
              (linear1): Linear(in_features=192, out_features=768, bias=True)
              (linear2): Linear(in_features=768, out_features=192, bias=True)
              (fn): GELU(approximate=none)
              (drop1): Dropout(p=0.0, inplace=False)
              (drop2): Dropout(p=0.0, inplace=False)
            )
          )
          (1): SwinTransformerBlock(
            (norm1): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              (qkv): Linear(in_features=192, out_features=576, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=192, out_features=192, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): Identity()
            (norm2): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
            (mlp): MLPBlock(
              (linear1): Linear(in_features=192, out_features=768, bias=True)
              (linear2): Linear(in_features=768, out_features=192, bias=True)
              (fn): GELU(approximate=none)
              (drop1): Dropout(p=0.0, inplace=False)
              (drop2): Dropout(p=0.0, inplace=False)
            )
          )
        )
        (downsample): PatchMerging(
          (reduction): Linear(in_features=768, out_features=384, bias=False)
          (norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
      )
    )
  )
  (encoder1): UnetrBasicBlock(
    (layer): UnetResBlock(
      (conv1): Convolution(
        (conv): Conv3d(3, 24, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
      )
      (conv2): Convolution(
        (conv): Conv3d(24, 24, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
      )
      (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
      (norm1): InstanceNorm3d(24, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      (norm2): InstanceNorm3d(24, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      (conv3): Convolution(
        (conv): Conv3d(3, 24, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
      )
      (norm3): InstanceNorm3d(24, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    )
  )
  (encoder2): UnetrBasicBlock(
    (layer): UnetResBlock(
      (conv1): Convolution(
        (conv): Conv3d(24, 24, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
      )
      (conv2): Convolution(
        (conv): Conv3d(24, 24, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
      )
      (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
      (norm1): InstanceNorm3d(24, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      (norm2): InstanceNorm3d(24, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    )
  )
  (encoder3): UnetrBasicBlock(
    (layer): UnetResBlock(
      (conv1): Convolution(
        (conv): Conv3d(48, 48, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
      )
      (conv2): Convolution(
        (conv): Conv3d(48, 48, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
      )
      (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
      (norm1): InstanceNorm3d(48, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      (norm2): InstanceNorm3d(48, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    )
  )
  (encoder4): UnetrBasicBlock(
    (layer): UnetResBlock(
      (conv1): Convolution(
        (conv): Conv3d(96, 96, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
      )
      (conv2): Convolution(
        (conv): Conv3d(96, 96, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
      )
      (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
      (norm1): InstanceNorm3d(96, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      (norm2): InstanceNorm3d(96, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    )
  )
  (encoder10): UnetrBasicBlock(
    (layer): UnetResBlock(
      (conv1): Convolution(
        (conv): Conv3d(384, 384, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
      )
      (conv2): Convolution(
        (conv): Conv3d(384, 384, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
      )
      (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
      (norm1): InstanceNorm3d(384, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      (norm2): InstanceNorm3d(384, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    )
  )
  (decoder5): UnetrUpBlock(
    (transp_conv): Convolution(
      (conv): ConvTranspose3d(384, 192, kernel_size=(1, 2, 2), stride=(1, 2, 2), bias=False)
    )
    (conv_block): UnetResBlock(
      (conv1): Convolution(
        (conv): Conv3d(384, 192, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
      )
      (conv2): Convolution(
        (conv): Conv3d(192, 192, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
      )
      (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
      (norm1): InstanceNorm3d(192, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      (norm2): InstanceNorm3d(192, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      (conv3): Convolution(
        (conv): Conv3d(384, 192, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
      )
      (norm3): InstanceNorm3d(192, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    )
  )
  (decoder4): UnetrUpBlock(
    (transp_conv): Convolution(
      (conv): ConvTranspose3d(192, 96, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
    )
    (conv_block): UnetResBlock(
      (conv1): Convolution(
        (conv): Conv3d(192, 96, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
      )
      (conv2): Convolution(
        (conv): Conv3d(96, 96, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
      )
      (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
      (norm1): InstanceNorm3d(96, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      (norm2): InstanceNorm3d(96, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      (conv3): Convolution(
        (conv): Conv3d(192, 96, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
      )
      (norm3): InstanceNorm3d(96, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    )
  )
  (decoder3): UnetrUpBlock(
    (transp_conv): Convolution(
      (conv): ConvTranspose3d(96, 48, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
    )
    (conv_block): UnetResBlock(
      (conv1): Convolution(
        (conv): Conv3d(96, 48, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
      )
      (conv2): Convolution(
        (conv): Conv3d(48, 48, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
      )
      (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
      (norm1): InstanceNorm3d(48, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      (norm2): InstanceNorm3d(48, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      (conv3): Convolution(
        (conv): Conv3d(96, 48, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
      )
      (norm3): InstanceNorm3d(48, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    )
  )
  (decoder2): UnetrUpBlock(
    (transp_conv): Convolution(
      (conv): ConvTranspose3d(48, 24, kernel_size=(1, 2, 2), stride=(1, 2, 2), bias=False)
    )
    (conv_block): UnetResBlock(
      (conv1): Convolution(
        (conv): Conv3d(48, 24, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
      )
      (conv2): Convolution(
        (conv): Conv3d(24, 24, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
      )
      (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
      (norm1): InstanceNorm3d(24, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      (norm2): InstanceNorm3d(24, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      (conv3): Convolution(
        (conv): Conv3d(48, 24, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
      )
      (norm3): InstanceNorm3d(24, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    )
  )
  (decoder1): UnetrUpBlock(
    (transp_conv): Convolution(
      (conv): ConvTranspose3d(24, 24, kernel_size=(1, 2, 2), stride=(1, 2, 2), bias=False)
    )
    (conv_block): UnetResBlock(
      (conv1): Convolution(
        (conv): Conv3d(48, 24, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
      )
      (conv2): Convolution(
        (conv): Conv3d(24, 24, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
      )
      (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
      (norm1): InstanceNorm3d(24, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      (norm2): InstanceNorm3d(24, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      (conv3): Convolution(
        (conv): Conv3d(48, 24, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
      )
      (norm3): InstanceNorm3d(24, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    )
  )
  (out): UnetOutBlock(
    (conv): Convolution(
      (conv): Conv3d(24, 2, kernel_size=(1, 1, 1), stride=(1, 1, 1))
    )
  )
)
enc0.shape torch.Size([3, 24, 20, 256, 256])
enc1.shape torch.Size([3, 24, 20, 128, 128])
enc2.shape torch.Size([3, 48, 20, 64, 64])
enc3.shape torch.Size([3, 96, 10, 32, 32])
dec4.shape torch.Size([3, 384, 5, 8, 8])
dec3.shape torch.Size([3, 192, 5, 16, 16])
dec2.shape torch.Size([3, 96, 10, 32, 32])
dec1.shape torch.Size([3, 48, 20, 64, 64])
dec0.shape torch.Size([3, 24, 20, 128, 128])
out.shape torch.Size([3, 24, 20, 256, 256])
logits.shape torch.Size([3, 2, 20, 256, 256])
outputs.shape torch.Size([3, 2, 20, 256, 256])