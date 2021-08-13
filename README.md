# Underwater Image Enhancement

## CycleGan Model
![CycleGan Arch](./doc_assets/images)

## Training Options
```bash
Underwater Image Enhancement [-h] [--checkpoints_dir CHECKPOINTS_DIR]
                                    [--name NAME] [--name_time] --model MODEL
                                    --dataroot DATAROOT [--ngf NGF]
                                    [--ndf NDF] [--n_layers_D N_LAYERS_D]
                                    [--n_blocks_G N_BLOCKS_G] [--no_dropout]
                                    [--serial_batches] --preprocess PREPROCESS
                                    [--no_flip] [--load_size LOAD_SIZE]
                                    [--crop_size CROP_SIZE]
                                    [--batch-size BATCH_SIZE] [--hosts HOSTS]
                                    [--current-host CURRENT_HOST]
                                    [--num_gpus NUM_GPUS] [--backend BACKEND]
                                    [--model-dir MODEL_DIR]
                                    [--output-data-dir OUTPUT_DATA_DIR]
                                    [--cloud CLOUD]
                                    [--visuals_freq VISUALS_FREQ]
                                    [--artifact_freq ARTIFACT_FREQ]
                                    [--training-data-dir TRAINING_DATA_DIR]
                                    [--subdir SUBDIR] [--phase PHASE]
                                    [--epoch_count EPOCH_COUNT]
                                    [--n_epochs N_EPOCHS]
                                    [--n_epochs_decay N_EPOCHS_DECAY]
                                    [--beta1 BETA1] [--lr LR]
                                    [--gan_mode GAN_MODE]
                                    [--pool_size POOL_SIZE]
                                    [--lr_policy LR_POLICY]
                                    [--lr_decay_iters LR_DECAY_ITERS]
                                    [--lambda_A LAMBDA_A]
                                    [--lambda_B LAMBDA_B]
                                    [--lambda_identity LAMBDA_IDENTITY]
```

## Running Test
```bash
python test.py --model cyclegan2 --preprocess RAC --load_model 20 --examples 2 --phase train --visuals

python test.py --model cyclegan2 --preprocess RAC --load_model 103 --examples 20 --phase train --save_artifacts --all_metrics --log_out
```

## Metric

- PSNR
- SSIM
- Entropy
- UIQM - under water image quality metric = `c1 + UICM + c2 * UISM + c3 * UIConM` (c1=0.0282, c2=0.2953, c3=3.5753)

## Milestone

- Create a PPT about matrices
- Create a metric table

## Notes
1. Inside `model` model files should have the name `models/[model_name]_model.py`, 
   different models can be imported by the `model` opt
   
python train.py --model cyclegan2 --preprocess RRC --num_gpus 1 --cloud colab --training-data-dir dataset --ct 20 --gan_mode vanilla