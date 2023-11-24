# ids706_python_template
Mini project12 for ids706: 
- Use MLflow to Manage an ML Biomedical Image Processing Project

## Requirements
- Python 3.8+
- Virtual environment (optional but recommended, already set up as env in Makefile)
- Packages listed in `requirements.txt` including 


## Set up
1. Run `make setup` to set up the virtual environment for the project.
note: only need to set up the environment once

2. Run `make install` to install all packages listed in requirement.txt

3. Run `make lint` to check up the style

4. Run `make format` to check up the format

5. Run `make test` to test the main.py

6. Run `make all` to finish all the set up operations at the same time


## Pipeline of model training
- In this project, Machine Learning Model Training Pipeline including **dataset creation**, **model creation**, and **model training** are written in `model.py`. 
1. Dataset Creation: 

Medical Cellular Dataset is downloaded from "https://data.mendeley.com/public-files/datasets/snkd93bnjr/files/2fc38728-2ae7-4a62-a857-032af82334c3/file_downloaded", unzipped and stored in ./data.

Crop is used for data augmentation and preprocessing


2. Model Creation:

The Model used consists of the following layers
* Input layer; 
* convolutional layers; 
* max pooling layer; 
* convolutional layers; 
* max pooling layer;
* dense layer.

3. Train the model:

The model is trained based on **SGD optimizer**, **CategoricalCrossentropy** loss function, and **accuracy** is used to indicate the model training efficiency.


- Manage the MLproject with ml flow
1. `mlflow.start_run()` and `mlflow.end_run()` are used at the start and end of the process seperately to indicate the start and end of the machine learning pipeline

2.  `mlflow.sklearn.log_model(model, "model")` is used to log model 

3.  `mlflow.log_metric("loss", hist.history['loss'][-1])` and `mlflow.log_metric("accuracy", hist.history['accuracy'][-1])` are used to log the tracking metrics for training



## Structure
1. `.devcontainer` includes a Dockerfile and devcontainer.json. The files in this container specify how the project can be set up.

2. `.github` includes the CI/CD settings

3. All packages needed for this project are in `requirements.txt`

4. `.gitignore` includes all the files that do not want to be tracked by github

5. `Makefile` includes all the make settings for this project

6. `model.py` includes all the operations related with machine learning model 

7. `Makefile` includes all the make settings for this project

8. `data` includes dataset used for traning

9. `mlruns` includes logs used for machine learning project management. log files of mlflow are written to `mlruns`


## CI/CD
GitHub Actions are configured for Continuous Integration and Continuous Deployment. See `.github/workflows/ci_cd.yml` for details.




