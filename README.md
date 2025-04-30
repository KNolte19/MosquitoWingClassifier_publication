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
- Open the file `utils/BALROG_pipeline.py`
- Set the absolute path to the repository (line 16) 
  - e.g. `ROOTPATH = "/home/username/MosquitoWingClassifier_publication/"`

### 1. Dataset Creation

- Datasets were generated using the notebooks:
  - `classifier/dataset-splitter_classifier.ipynb`
  - `robustness_experiment/dataset-splitter_device.ipynb`
- The datasets are based on our publicly available mosquito wing image dataset: [EBI BioImages - S-BIAD1478](https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD1478)
  - TODO: how to download this and where exactly does it need to be stored to run the classifier
  - Data needs to be saved in the root directory in a folder called `MosquitoWingDataset`
- Final processed `.npy` files for training are available for download: [Google Drive – Dataset Files](https://drive.google.com/drive/folders/1KVqjOPV90UKcxodv_agUO3Tx2GYhggVd?usp=share_link)
  - Everything in the drive (4 *.npy and 1 folder) should be downloaded and put into `classifier/data/`

### 2. Model Training

- Models for the main classifier were trained using:  
  `classifier/trainer_classifier.ipynb`
- Models for the robustness experiment were trained using:  
  `robustness_experiment/trainer_robustness.ipynb`
- Note: These scripts are optimized for Google Colab.
- Five models were trained in total, each with a different fold as the test set. Model names indicate the fold used.
- Trained models for the main classifier are located in:  
  `classifier/models/`

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

- `robustness_experiment/analyse_results_deviceexp.ipynb`: Computes performance metrics for models in the robustness experiment.

### 4. Utility Functions

- `utils/BALROG_pipeline.py`: Contains helper functions for data preprocessing and generation.
- `references`: Dataframes describing the dataset. Refer to `utils/references/database_reference_MLREADY.xlsx` for full overview of the dataset composition.

## Citation
If you use this repository, please cite our publication:  
**"Potentials and limitations in the application of Convolutional Neural Networks for mosquito species identification using wing images"**  
[View on bioRxiv](https://www.biorxiv.org/content/10.1101/2025.01.29.635420v1.article-info)


## License
This code and all data in this repository are licensed under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license.  
You are free to share and adapt the code as long as you provide appropriate credit.  


## Contact

For questions or feedback, feel free to contact:  
[Kristopher Nolte](mailto:kristophernolte@bnitm.de)
