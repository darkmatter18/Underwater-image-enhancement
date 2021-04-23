# Underwater Image Enhancement

## Running Test
```bash
python -m trainer.test --dataroot "../Dataset/EUVP Dataset/Paired/underwater_dark" --no_gpu --load_model 136
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