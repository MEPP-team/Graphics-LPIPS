# Graphics-LPIPS

**Graphics-LPIPS is The metric is an extension of the [LPIPS metric](https://github.com/richzhang/PerceptualSimilarity) 
originally designed for images and perceptual similarity tasks, which we adapted for 3D graphics and quality assessment tasks.
Graphics-LPIPS employs CNNs with learning linear weights on top, fed with reference and distorted patches of rendered images of 3D models. 
The overall quality of the 3D model is derived by averaging local patch qualities**

This project is the implementation of our paper: [Textured Mesh Quality Assessment: Large-Scale Dataset and Deep Learning-based Quality Metric](https://yananehme.github.io/publications/) 
[Yana Nehmé](https://yananehme.github.io/), [Florent Dupont](https://perso.liris.cnrs.fr/florent.dupont/), [Jean-Philippe Farrugia](http://perso.univ-lyon1.fr/jean-philippe.farrugia/), 
[Patrick Le Callet](https://scholar.google.fr/citations?user=llgwlUgAAAAJ&hl=fr), [Guillaume Lavoué](https://perso.liris.cnrs.fr/guillaume.lavoue/)

## (0) Dependencies/Setup

### Installation
- Install PyTorch 1.0+ and torchvision fom http://pytorch.org

```bash
pip install -r requirements.txt
```
- Clone this repo:
```bash
git clone https://github.com/YanaNEHME/Graphics-LPIPS
cd PerceptualSimilarity
```
## (1) About the metric
Graphics-LPIPS predicts a quality score ranging between [0,1]. 
The higher the Graphics-LPIPS values, the more different the patches are.

### (A) Quality of a patch
Example of script to compute the distance between a reference patch (p0) and a distorted patch (p1). 
You can pass in parameter (-m TheModelPath) the path of the network to use, usually located in the `./checkpoints` directory.
To use the GPU, turn on the --use_gpu parameter.
The output is the predicted quality score of the distorted patch.
```
python GraphicsLpips_2imgs.py -p0 imgs/ex_ref.png -p1 imgs/ex_p0.png --use_gpu
```
### (B) Quality of a 3D model
Example of script to compute the quality of a set of distorted 3D graphics.
The snapshots of the models are patchified/divided into small patches of size 64x64.
The number of patches obtained for each model is stored in a csv file.
Graphics-LPIPS estimates the quality locally (i.e. per patch), then the global quality score of the model is the average of the local patch qualities.
```
python GraphicsLpips_csvFile_TexturedDB.py -m './checkpoints/Trial1/latest_net_.pth' --use_gpu
```

## (2) Dataset
Grpahics-LPIPS was trained and tested on a challenging dataset of 3000 textured.
The dataset was generated from 55 source models corrupted by combinations of 5 types of compression-based distortions applied on the geometry, texture
mapping and texture image of the meshes.
The stimuli were annotated in a large-scale subjective experiment in crowdsourcing, based on the DSIS method.
Thus, each stimulus is associated with a subjective quality score a.k.a Mean Opinion Score (MOS).

[Download the dataset](https://yananehme.github.io/datasets/)(3D models and their distorted version, snapshots taken from the main viewpoint of the stimuli, subjective scores)
 
### Using the dataset to train and test the metric
Graphics-LPIPS is designed for 3D graphics and quality assessment tasks.
To predict the overall quality of a stimulus, we modified the original LPIPS metric so that:
(1) the small network (G), trained on top, suits MOS scores instead of preference scores
(2) The optimization (the loss computation and function) is done per image (instead of patch-wise)

We employed the pre-trained AlexNet network with its fixed weights and learnt the weights of a linear layer on top.
To train our model, we considered images of the 3D models taken from their main viewpoint to which we associated the MOS scores.
The images were divided into patches of size 64x64.
See scripts `train.py` and `./scripts/train_metric.txt` for an example of training and testing the metric. 
The script will train a model on randomly sampled patches of stimulus images from the training set, for `--nepoch` + `--nepoch_decay`  epochs.
As the distances computed for patches of the same image are combined for the loss calculation, the patches of the same image can not be distributed over different batches. 
Thus, each batch was made to contain  `--nInputImg` images, each represented by `--npatches` randomly sampled,
resulting in a batch size of `--nInputImg` x `--npatches` patches. 
The backpropagated error is the average loss over the images in a batch.

During training, patches are randomly sampled every epoch to ensure that as many different image patches as possible are used in training. 
80% of the stimuli in the dataset are used for training and 20% for testing. 
Subjective scores were scaled between [0,1] - 0: imperceptible distortion (highest quality), 1: very annoying distortion (lowest quality).

Training will add a subdirectory in the `checkpoints` directory.

[Download the dataset (patchified used to train and test the metric)](https://perso.liris.cnrs.fr/ynehme/datasets/Graphics-Lpips/dataset.zip), and unzip it into directory `./dataset`

## Acknowledgements
This work was supported by French National Research Agency as part of ANR-PISCo project (ANR-17-CE33-0005).

Reference: Yana Nehmé, Florent Dupont, Jean-Philippe Farrugia, Patrick Le Callet, Guillaume Lavoué, Textured mesh quality assessment: Large-scale dataset and deep learning-based quality metric, arXiv preprint arXiv:2202.02397 (2022).

## License
The Graphics-LPIPS metric is Copyright of University of Lyon, 2022.
It is distributed under the Mozilla Public License v. 2.0. (refer to the accompanying file `LICENSE-MPL2.txt` or a copy at http://mozilla.org/MPL/2.0/)




