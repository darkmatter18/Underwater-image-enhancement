<h1 align="center">Underwater Image Enhancement</h1>

<p align="center">
    <em>A Deep Learning CycleGAN Based application, that can enhance the underwater images.</em>
</p>

<p align="center">
    <em>Collaboratively with 
        <a href="https://github.com/dotslash21">
            @dotslash21 
        </a>
        and
        <a href="https://github.com/pranjalb21">
            @pranjalb21
        </a>
    </em>
</p>

<p align="center">
    <em>
        <a href="https://github.com/darkmatter18/Underwater-image-enhancement/blob/master/Underwater_Image_Enhancement.pdf">
            Unpublished Research Paper (PDF)
        </a>
    </em>
</p>


## CycleGan Model
![CycleGan Arch](Latex/model-diagram.png)

## Model Architecture

### 1. Generator
![Generator Diagram](Latex/generator.jpg)

### 2. Discriminator
![Discriminator](Latex/Discriminator.jpg)

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