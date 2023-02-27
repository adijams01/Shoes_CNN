# Sneakers Classification using Convolutional Neural Network
This is a project that demonstrates how to use Convolutional Neural Networks (CNNs) for classifying images of sneakers into their respective brands.

## Dataset
The dataset used for this project is the Nike, Adidas and Converse Shoes Images dataset from [Kaggle](https://www.kaggle.com/datasets/die9origephit/nike-adidas-and-converse-imaged). It consists of 825 images of sneakers belonging to 3 different categories (Nike, Adidas and Converse) The images are in JPEG format and have varying dimensions.

## Prerequisites
Python 3.x
TensorFlow 2.x
NumPy
Matplotlib
scikit-learn
Jupyter Notebook (optional:skip below steps and run it in collab)
You can install all the required Python packages by running the following command:

Copy code
```
pip install tensorflow numpy matplotlib scikit-learn jupyter
```
Usage
Clone the repository:
bash
Copy code
```
git clone https://github.com/adijams01/Sneakers_CNN.git
```
Download the dataset from Kaggle and extract it into the data folder or you can get it from the repository itself.

Open the Sneakers_CNN.ipynb Jupyter Notebook.

Run the cells in the notebook to train and evaluate the CNN model. You can modify the hyperparameters of the model and experiment with different settings.

## Results
The trained CNN model achieves an accuracy of around 53.51% on the test set, which is an average performance considering the complexity and variability of the sneaker images. You can also visualize the training and validation loss and accuracy curves using the Matplotlib library.
You can also Classify with your images or you can get samples from [here](https://github.com/adijams01/Sneakers_CNN/tree/main/samples%20for%20predictions)

## Conclusion
Convolutional Neural Networks are a powerful tool for image classification tasks, and this project demonstrates how to use them for classifying sneakers into their respective categories. With further with **Transfer Learning** or getting more data, you can improve the performance of the model and apply it to other image.
