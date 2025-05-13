# MosquitoWingClassifier Publication

This repository contains the code and data associated with the publication:  
**"Potentials and limitations in the application of Convolutional Neural Networks for mosquito species identification using wing images"**

## Repository Structure

- `/classifier`: Scripts for dataset creation, model training, and analysis of the main models trained for 21 mosquito species (see Sections 3.1–3.3 & 3.5 in the manuscript).
- `/feasibility_experiment`: Scripts and data for the feasibility study (Section 3.6).
- `/figures`: All figures created for the publication (Figures 1–7).
- `/robustness_experiment`: Scripts and data for the robustness experiments (Section 3.5).
- `/utils`: Utility functions for data generation and metadata, including reference dataframes and dataset splits.
- `requirements.txt`: Contains all dependencies to run the scripts. 

## Workflow

### 0. Set absolute path to repository

- Open the file `utils/config.py`
- Set the absolute path to the repository (line 1) 
  - e.g. `ROOT = "/home/username/MosquitoWingClassifier_publication/"`

### 1. Dataset Creation

- Datasets for the two experiments (classifer and robustness experiment) were generated using the notebooks:
  - `classifier/dataset-splitter_classifier.ipynb`
  - `robustness_experiment/dataset-splitter_device.ipynb`

 As training was done via GoogleColab all image files were preprocessed and saved as `.npy` file.
 We provide the preprocessed images as `.npy` file via [Google Drive – Data](https://drive.google.com/drive/folders/1KVqjOPV90UKcxodv_agUO3Tx2GYhggVd?usp=share_link). If you want to follow these instructions and replicate the experiments download the files and move the `.npy` files to the `/data` folder of the respective experiments.

 The original images can accessed and downloaded here: [EBI BioImages - S-BIAD1478](https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD1478). They were all preprocessed using `utils/BALROG_pipeline.py`.

### 2. Model Training

- Models for the main classifier were trained using:  
  `classifier/trainer_classifier.ipynb`
- Models for the robustness experiment were trained using:  
  `robustness_experiment/trainer_robustness.ipynb`

Note: These scripts were optimized for Google Colab. We adapted them to be used in this repository. If you want to use original setup copy this folder to your drive and run the scripts using Google Colab. [Google Drive](https://drive.google.com/drive/folders/1PoQCBq7t0R7cGMgllmFHxpwLHtMMs_Ds?usp=share_link).

### 3. Evaluation and Analysis

#### Main Classifier Analysis

- `classifier/test_classifier.ipynb`: Evaluates models on the respective test sets.
- `classifier/analyse_dataset.ipynb`: Analyzes dataset composition and generates related figures.
- `classifier/analyse_GradCam.ipynb`: Computes and visualizes Grad-CAM heatmaps.
- `classifier/analyse_classifier.ipynb`: Computes performance metrics for the trained models.
- `classifier/analyse_UMAP.ipynb`: Generates UMAP visualizations from feature embeddings.

#### Feasibility Study

- `feasibility_experiment/analyse_feasibility.ipynb`: Evaluates model predictions on out-of-distribution (OOD) images.
- `feasibility_experiment/analyse_feasibility_zeiss-device.ipynb`: Evaluates model performance on images captured with a novel device.

#### Robustness Experiment

- `robustness_experiment/analyse_results_robustness.ipynb`: Computes performance metrics for models in the robustness experiment.
- `robustness_experiment/analyse_dataset_robustness.ipynb`:  Analyzes dataset composition as seen Supplementary Material 4.

### 4. Utility Functions

- `utils/config.py`: Contains variable to set ROOT path.
- `utils/BALROG_pipeline.py`: Contains helper functions for data preprocessing and generation.
- `references`: Dataframes describing the dataset. Refer to `utils/references/database_reference_MLREADY.xlsx` for full overview of the dataset composition.

## Citation

If you use this repository, please cite our publication:  
"Potentials and limitations in the application of Convolutional Neural Networks for mosquito species identification using wing images" [View on bioRxiv](https://www.biorxiv.org/content/10.1101/2025.01.29.635420v1.article-info)

## License

This code and all data in this repository are licensed under the [(CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license.  
You are free to share and adapt the code as long as you provide appropriate credit.  

## Contact

For questions or feedback, feel free to contact: [Kristopher Nolte](mailto:kristophernolte@bnitm.de)
