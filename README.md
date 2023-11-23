## README.md

This is a repository dedicated to use different AI algorithms to predict time series data, which takes form as weather data in Jena from 2009 to 2016. It consists mainly of a python notebook for experimentation and image generation and a python script for model performance comparison.

**How to run the code in this repository**

To run the code in this repository, you will need to have the download the packages specified in the `env.yml` file.

**Setting up (the environment)**

Run `conda env create -f env.yml` in the terminal
Run `conda activate weather_env`
Run `python download_dataset.py` to download the `jena_climate_2009_2016.csv` and `jena_climate_2009_2016.csv.zip` files for the dataset, the zipped file can be deleted.
Run `python main.py`