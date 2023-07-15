# Skin Cancer Image Classification

This project aims to develop a skin cancer image classification model using the Melanoma Skin Cancer Dataset. The goal is to accurately identify and classify skin images as either benign or malignant, with a specific focus on detecting melanoma. The project explores the performance of two different deep learning models: InceptionV3 and ResNet.

## Dataset

The dataset used for this project is the Melanoma Skin Cancer Dataset, which contains 10,000 images of skin lesions. It includes both benign and malignant cases, with a particular emphasis on melanoma samples. The dataset can be found [here](https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images).

## Models

Two state-of-the-art deep learning models are employed for skin cancer classification:

1. InceptionV3: This model utilizes a deep convolutional neural network architecture that has been trained on the ImageNet dataset. InceptionV3 is known for its effectiveness in capturing fine-grained features from images.

2. ResNet: This model is based on residual neural networks, which enable the training of extremely deep networks. ResNet introduces skip connections to address the vanishing gradient problem and allows for more efficient training and improved accuracy.

## Methodology

1. Data Preprocessing: Perform necessary data preprocessing steps, such as resizing, normalization, and augmentation, to prepare the images for training.

2. Model Training: Train the InceptionV3 and ResNet models using the preprocessed images. Experiment with different hyperparameters, such as learning rate and batch size, to achieve optimal results. Compare the performance of both models in terms of accuracy, precision, recall, and F1 score.

3. Model Evaluation: Evaluate the trained models using appropriate metrics and validation and test sets. Compare the performance of InceptionV3 and ResNet in terms of their ability to accurately classify skin lesions.


## Project Structure

The project repository is organized as follows:

- `notebooks/`: Jupyter notebooks for data preprocessing, model training, and evaluation.
- `README.md`: This file, providing an overview of the project.

## Requirements

- Python 3.x
- TensorFlow or PyTorch (depending on the deep learning framework used for InceptionV3 and ResNet)
- Additional Python libraries as specified in the `requirements.txt` file.

## Usage

1. Clone the repository
   
2. Install the required dependencies
   
3. Follow the instructions in the `notebooks/` directory to preprocess the data, train the InceptionV3 and ResNet models, and evaluate their performance.


## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

   
