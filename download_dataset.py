import os
import urllib.request
import zipfile


def download_file(url, destination):
    print(f"Downloading: {url}")
    """Download a file from a URL and save it to the destination."""
    urllib.request.urlretrieve(url, destination)


def unzip_file(zip_file, destination):
    print(f"Extracting: {zip_file}")
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(destination)


def main():
    dataset_url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
    dataset_filename = "jena_climate_2009_2016.csv.zip"

    if os.path.exists(dataset_filename):
        print(f"The dataset file '{dataset_filename}' is already in the repo.")
    else:
        # Download the file if it's not in the repository
        download_file(dataset_url, dataset_filename)
        print(f"The dataset file '{dataset_filename}' has been downloaded.")

    unzip_file(dataset_filename, os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
