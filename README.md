
# *3DFaceFill: An Analysis-by-Synthesis Approach for Face Completion*

## Abstract
Existing face completion solutions are primarily driven by end-to-end models that directly generate 2D completions of 2D masked faces. By having to implicitly account for geometric and photometric variations in facial shape and appearance, such approaches result in unrealistic completions, especially under large variations in pose, shape, illumination and mask sizes. To alleviate these limitations, we introduce 3DFaceFill, an analysis-by-synthesis approach for face completion that explicitly considers the image formation process. It comprises three components, (1) an encoder that disentangles the face into its constituent 3D mesh, 3D pose, illumination and albedo factors, (2) an autoencoder that inpaints the UV representation of facial albedo, and (3) a renderer that resynthesizes the completed face. By operating on the UV representation, 3DFaceFill affords the power of correspondence and allows us to naturally enforce geometrical priors (e.g. facial symmetry) more effectively. Quantitatively, 3DFaceFill improves the state-of-the-art by up to 4dB higher PSNR and 25% better LPIPS for large masks. And, qualitatively, it leads to demonstrably more photorealistic face completions over a range of masks and occlusions while preserving consistency in global and component-wise shape, pose, illumination and eye-gaze.

## Overview
3DFaceFill is an iterative inpainting approach where the masked face is disentangled into its 3D shape, pose, illumination and partial albedo by the 3DMM module, following which the partial albedo, represented in the UV space is inpainted using symmetry prior, and finally the completed image is rendered.
![alt text](images/overview.png "Architecture")

## Contributions
* Face completion using explicit 3D priors leading to geometrically and photometrically better results
* We leverage facial symmetry efficiently for face completion
* Qualitative and quantitative improvement in face completion under diverse conditions of shape, pose, illumination, etc

## Data Preparation
3DFaceFill is trained using the CelebA dataset. We preprocess the CelebA dataset to align and crop the face images. To do this, edit the process_celeba.py file to point the celeba_dir variable to the path of the CelebA dataset. Then, run the script as python process_celeba.py

## Setup
Set up a conda environment with Python 3.6 and PyTorch 1.4.0. Use the attached env.yml file to create the environment with all the required libraries conda env create -f env.yml

In addition, install the zbuffer cuda library using the following steps:
1. cd zbuffer/
2. python setup.py install

Download the pretrained 3DMM and Face Segmentation models from https://drive.google.com/drive/folders/1Pf1CEWjX1DtovTE4NDgiyE3CzAN-ucmC?usp=sharing to the ./checkpoints/ directory.

Download 3DMM_definition.zip from https://drive.google.com/file/d/1EO6Mb0ELVb82ucGxRmO8Hb_O1sb1yda9/view?usp=sharing and extract to datasets/

## Usage
To train the network from scratch, run:
python main.py --config args.txt --ngpu 1

To evaluate the network, run:
python main.py --config args.txt --ngpu 1 --resolution [256,256] --eval

## Results
Quantitative results on the CelebA, CelebA-HQ and MultiPIE datasets.
![alt text](images/quantitative.png "Quantitative Results")

Qualitative results with artificial masks.
![alt text](images/qualitative.png "Qualitative Results on Artificial Masks")

Qualitative results with real occlusions.
![alt text](images/occlusions.png "On Real Occlusions")
