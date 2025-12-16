<div align="center">
  <img src="https://github.com/user-attachments/assets/9266e5a5-e4aa-4b0a-9fb2-a07c6b78feed" alt="RoadImage" width="700">
</div>

<div align="center">
  <a href="https://www.python.org/downloads/release/python-31015/" target="_blank">
  <img src="https://img.shields.io/badge/Python-3.10.15-blue.svg" alt="Python 3.10.15"></a>
  <a href="https://pytorch.org/get-started/locally/" target="_blank">
    <img src="https://img.shields.io/badge/PyTorch-2.5.1-orange.svg" alt="PyTorch 2.5.1"></a>
  <a href="https://developer.nvidia.com/cuda-12-1-0-download-archive" target="_blank">
  <img src="https://img.shields.io/badge/CUDA-12.1-brightgreen.svg" alt="CUDA 12.1"></a>
<a href="https://developer.nvidia.com/cudnn" target="_blank">
  <img src="https://img.shields.io/badge/cuDNN-9.1.0-brightgreen.svg" alt="cuDNN 9.1.0"></a>
  <a href="https://github.com/Dalageo/IDDRoadSegmentation/blob/main/LICENSE" target="_blank">
    <img src="https://img.shields.io/badge/License-AGPL%20v3-800080" alt="License: AGPLv3"></a>
  <img src="https://img.shields.io/github/stars/Dalageo/IDDRoadSegmentation?style=social" alt="GitHub stars">
</div>

# Distinguishing Urban Roads from Non-Road Regions in the Indian Driving Dataset(IDD) Using Binary Deep Learning Segmentation 🛣️

This project develops a binary segmentation model capable of distinguishing between **Road** and **Non-Road** regions in urban driving images captured in India. Models utilizing pixel-level analysis are provided, including a U-Net model built from scratch, a pretrained U-Net, and a pretrained FPN, with the pretrained U-Net being fine-tuned on the IDD dataset. The outcomes of this project have potential applications in various domains, such as autonomous driving, road maintenance, mapping and infrastructure planning, or traffic management.

More specifically, the segmentation process is based on a [U-Net architecture](https://arxiv.org/pdf/1505.04597v1),imported from the [segmentation_models](https://github.com/qubvel-org/segmentation_models.pytorch) PyTorch library, which uses an EfficientNet backbone pre-trained on ImageNet. To improve its performance and ensure robustness across diverse scenarios, several data augmentation techniques are employed during training. These techniques include horizontal and vertical flips, as well as brightness adjustments, enabling the model to effectively learn from a more varied dataset and generalize better to unseen images, while accounting for the complexity and variability inherent in real-world urban environments.

## Dataset Description

The dataset used in this project is the [Indian Driving Dataset (IDD)](https://www.kaggle.com/datasets/mitanshuchakrawarty/new-idd-dataset/data), which consists of road images and their corresponding segmentation masks. This dataset is specifically designed for binary segmentation tasks, such as distinguishing between **Road** and **Non-Road** areas. All images and masks have been resized to uniform dimensions of **512x512 pixels**. The dataset is organized into two folders:

- `image_archive` contains the images in 3 channels (RGB), including a total of 6993 images. The files are in a `.png` format and are named as `Image_{num}`.

- `mask_archive` contains 6993 single-channel binary masks corresponding to the road images. The masks are also in a `.png` format and are named as `Mask_{num}` to align with the number of the images, making them easy to locate.

The following table represents the summary of the data as well as the number of files, shapes, and their naming conventions:

<div align="center">

| Directory       | Description                  | Number of Files | Shape       | Example Naming    |
|------------------|------------------------------|------------------|-------------|-------------------|
| `image_archive` | Road images                  | 6993             | 512x512x3   | `Image_{num}.png` |
| `mask_archive`  | Binary masks (Road/Non-Road) | 6993             | 512x512     | `Mask_{num}.png`  |

</div>

Below is an example of a road image and its corresponding mask aligned pixel-wise. An additional overlay visualization (not included in the dataset) is provided to visually demonstrate how the mask highlights specific regions (e.g., Road vs. Non-Road) in the context of the original image.

<div align="center">
  <img src="https://github.com/user-attachments/assets/c9830213-1fb3-469c-a988-d822c10cfe68" alt="Overlay-mask" width="800">
</div>

## Setup Instructions

### <img src="https://github.com/user-attachments/assets/8d36d1a5-e9b1-40d1-97c9-3d4ca49e9c95" alt="Local PC" width="18" height = "16" /> **Local Environment Setup**

1. **Clone the repository**:
   ```sh
   git clone https://github.com/Dalageo/road-segmentation-idd.git

2. **Navigate to the cloned directory**:
   ```sh
   cd IDDRoadSegmentation
  
3. **Open the `IDDRoadSegmentation.ipynb` using your preferred Jupyter-compatible environment (e.g., [Jupyter Notebook](https://jupyter.org/), [VS Code](https://code.visualstudio.com/), or [PyCharm](https://www.jetbrains.com/pycharm/))**
   
4. **Update the dataset, model and output directory paths to point to the location on your local environment.**
   
5. **Run the cells sequentially to reproduce the results.**

## Acknowledgments
Firstly, I would like to thank Olaf Ronneberger, Philipp Fischer, and Thomas Brox for introducing U-Net in their 2015 paper, *["U-Net: Convolutional Networks for Biomedical Image Segmentation."](https://arxiv.org/abs/1505.04597)*. 

Additionally, special thanks to [Pavel Iakubovskii](https://github.com/qubvel) for developing and maintaining the [segmentation_models](https://github.com/qubvel-org/segmentation_models.pytorch) pytorch library which was essential to the developement of this project.

<div align="center">
  <a href="https://github.com/qubvel-org/segmentation_models.pytorch?tab=readme-ov-file">
    <img src="https://github.com/user-attachments/assets/f59aa5cd-edbd-4f27-9a91-6fe8d5372c94" alt="Segmentation-Models" width="300">
  </a>
</div>

## License
The segmentation library is primarily licensed under the [MIT License](https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/LICENSE), with some files under other licenses. Refer to the [LICENSES](https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/licenses/LICENSES.md) directory and file statements for details, especially regarding commercial use. Meanwhile, the provided notebook, and accompanying documentation, are licensed under the [AGPL-3.0 license](https://www.gnu.org/licenses/agpl-3.0.en.html). AGPL-3.0 license was chosen to promote open collaboration, ensure transparency, and allow others to freely use, modify, and contribute to the work. 

Any modifications or improvements must also be shared under the same license, with appropriate acknowledgment. 

<div align="center">
  <br>
  <a href="https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/LICENSE" style="margin: 0 10px;">
    <img src="https://upload.wikimedia.org/wikipedia/commons/0/0c/MIT_logo.svg" alt="MIT-Logo" width="110"></a>&nbsp;&nbsp;&nbsp;
  <a href="https://www.gnu.org/licenses/agpl-3.0.en.html">
    <img src="https://github.com/user-attachments/assets/f3c6face-aa86-45da-8d20-d8ae25e49e28" alt="AGPLv3-Logo" width="200""></a>
</div>
