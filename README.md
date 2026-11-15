# Image-Segmentation-with-Multi-Scale-Line-Detector

This repository contains the implementation of a line segmentation algorithm based on the multi-scale detector by Nguyen et al., originally designed for the automatic extraction of the retinal vascular network. The project includes improvements in border handling and analyzes the viability of post-processing techniques, in addition to evaluating its generalization capability on satellite imagery.

Project developed for the **Image Processing and Computer Vision (July 2026)** course.

## Authors
*   **Junyi Jordi Jin**
*   **Francesc Vázquez Campodarbe**

---

## Project Summary

The main objective is to automate blood vessel extraction in retinas to assist in the early diagnosis of pathologies such as hypertension or stroke risk.

### Key Features:
*   **Multi-Scale Detector:** Overcomes false positive and central light reflex issues by combining linear responses at different scale lengths ($1 \le L \le W$).
*   **Advanced Border Handling:** A preprocessing technique (based on Soares et al.) has been implemented that iteratively extends the retinal boundaries outward, preventing the loss of pixel information at the edges of the FOV (*Field of View*) mask.
*   **Post-Processing Analysis:** It is experimentally demonstrated that typical post-processing steps (median filters, directional morphological filters, and reconstructions) are counterproductive due to the high quality and low noise already provided by the original multi-scale detector.
*   **Generalization Study:** The system is tested on satellite imagery (*Massachusetts Roads Dataset*) for road detection, determining that the algorithm is highly specialized and effective only on images with homogeneous backgrounds and a single type of line.

---

## Results and Accuracy

The model was evaluated using global accuracy (`ACC`) and local accuracy (`Local ACC`) metrics on the public **DRIVE** and **STARE** datasets, achieving results very close to those of a second human observer:

| Method / Configuration | DRIVE (Local ACC) | STARE (Local ACC) |
| :--- | :---: | :---: |
| 2nd Human Observer | — | **0.7706** |
| Original Method (No Border Ext. / No Post) | 0.7477 | 0.7057 |
| **Improved Method (With Border Ext. / No Post)** | **0.7642** | **0.7175** |

---

## Requirements and Execution

To properly run the Python script (`Codi.py`), make sure to follow the guidelines below regarding the dataset and system libraries.

### Important Notes on the Dataset:
*   The images retain their original database filenames and are organized into folders for easy identification.
*   The code does not account for the folder structure where the images are located. To run it, **you must extract all images from the folders** and place them in the same root execution directory.
*   The **STARE** dataset folder can be freely downloaded from its original source, which is completely public.

### Dependency Installation:
To read some of the images in specific formats (such as high-resolution `.tif` files), you may need to install the following package by running this command in your terminal:

```bash
pip install imagecodecs
```
