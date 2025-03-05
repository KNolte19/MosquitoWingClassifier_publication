# MosquitoWingClassifier Publication

This repository contains the code and data associated with the publication **"Potentials and limitations in the application of Convolutional Neural Networks for mosquito species identification using wing images"**.

## General Workflow of Model Training and Evaluation

1. **Dataset Creation**:
    - The datasets were created using the notebooks `dataset_splitter.ipynb` and `dataset_splitter_device.ipynb`.
    - The publicly available dataset of mosquito wing images previously published by us was used as the basis. You can find the dataset [here](https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD1478).

2. **Model Training**:
    - The CNN models were trained using the `trainer_classifier.ipynb` script, which can be found in the `/models` folder.
    - For the final models, we trained five models, each using a different fold as the testing set. The number after the model name indexes the fold used for testing.
    - The same process was repeated for all device experiments as described in the publication using the `trainer_classifier.ipynb` script.

3. **Analysis**:
    - Different aspects of the analysis were conducted using the following scripts:
      - `test_models.ipynb`: Generates the testing data of the models on the respective testing set.
      - `analyse_dataset.ipynb`: Generates figures to illustrate the dataset composition.
      - `analyse_feasibility.ipynb`: Generates model predictions on the feasibility images and returns statistics on performance.
      - `analyse_GradCam.ipynb`: Generates averaged GradCam images and creates figures.
      - `analyse_results_deviceexp.ipynb`: Calculates metrics for the performance of the models in the device experiment.
      - `analyse_results.ipynb`: Calculates metrics for the performance of the final models used in the application.
      - `analyse_UMAP.ipynb`: Generates feature maps and calculates UMAP visualizations of them.

4. **Utility Functions**:
    - In the `BALROG_pipeline.py` script, we collected useful functions utilized to generate the data.

## Repository Structure

- `/models`: Contains the trained models and the `trainer_classifier.ipynb` script.
- `/notebooks`: Contains the Jupyter notebooks used for dataset creation and analysis.
- `BALROG_pipeline.py`: Contains utility functions for data generation.

## Citation

If you use this code or data, please cite our publication:

**"Potentials and limitations in the application of Convolutional Neural Networks for mosquito species identification using wing images"**
[here](https://www.biorxiv.org/content/10.1101/2025.01.29.635420v1.article-info)


## Contact

For any questions or issues, please contact [Kristopher Nolte](mailto:kristophernolte@bnitm.de).
