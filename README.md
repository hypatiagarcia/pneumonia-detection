# Pneumonia Detection from Chest X-rays

## Project Overview

This project implements and evaluates a deep learning model for detecting pneumonia from chest X-ray images, based on the dataset from the [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge). The primary focus was on building an end-to-end image classification pipeline using PyTorch Lightning, leveraging transfer learning with ResNet-18, and exploring techniques to handle the inherent class imbalance in the dataset.

## Features & Workflow

*   **Data Loading:** Reads DICOM format medical images using `pydicom`.
*   **Preprocessing:** Resizes images, normalizes pixel values, and saves them in `.npy` format for efficient loading during training using `numpy` and `cv2`.
*   **Dataset Handling:** Utilizes `torchvision.datasets.DatasetFolder` and `torch.utils.data.DataLoader`.
*   **Data Augmentation:** Applies random affine transformations (rotation, translation, scaling) and random resized cropping during training via `torchvision.transforms` to improve model generalization.
*   **Model:** Implements a binary classifier using a ResNet-18 backbone pre-trained on ImageNet, adapting the input and output layers for grayscale images and binary classification.
*   **Training Framework:** Employs PyTorch Lightning for streamlined training, validation loops, GPU acceleration, and checkpointing.
*   **Class Imbalance Handling:** Addresses the dataset imbalance (approx. 3:1 non-pneumonia to pneumonia ratio) by using a weighted Binary Cross-Entropy loss (`BCEWithLogitsLoss` with `pos_weight=3`) during training, aiming to improve recall for the minority (pneumonia) class.
*   **Evaluation:** Calculates and reports standard classification metrics (Accuracy, Precision, Recall, Confusion Matrix) using `torchmetrics`.
*   **Logging:** Integrates with TensorBoard for monitoring training progress.

## Dataset

This project uses the **RSNA Pneumonia Detection Challenge** dataset available on Kaggle:
[https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge)

**Important:** Due to its size, the dataset is **not included** in this repository.
1.  Download the dataset from the Kaggle link above.
2.  Place the `stage_2_train_labels.csv` and the `stage_2_train_images` folder in a specific location (e.g., create a `./rsna-pneumonia-detection-challenge/` directory in the project root).
3.  Update the `ROOT_PATH` variable in the `preprocessing.ipynb` notebook (or script) to point to the location of `stage_2_train_images`.
4.  Run the `preprocessing.ipynb` notebook to process the DICOM files and generate the `processed/` directory containing `.npy` files used for training.

## Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd pneumonia_classification
    ```
2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download Dataset:** Follow the instructions in the "Dataset" section above.

## Usage

1.  **Preprocessing:** Run the `preprocessing.ipynb` notebook to process the raw DICOM images into `.npy` files located in the `processed/train` and `processed/val` directories.
2.  **Training:** Run the `train.ipynb` notebook (or associated `.py` script). Training logs and model checkpoints will be saved under the `logs/lightning_logs/` directory. The specific model trained in the provided example uses weighted loss.
3.  **Evaluation:** The `train.ipynb` notebook contains a section for evaluating a trained checkpoint.
    *   Update the `ckpt_path` variable to point to the desired `.ckpt` file (e.g., one saved during your training run in `logs/lightning_logs/version_X/checkpoints/`).
    *   Run the evaluation cells to calculate and display metrics (Accuracy, Precision, Recall, Confusion Matrix).
    *   **(Optional: If you didn't commit checkpoints)** Provide instructions here to download your best checkpoint (e.g., from Google Drive/Dropbox) and where to place it. Example: `Download the best checkpoint from [Link] and place it in the 'logs/checkpoints/' directory.`

## Results & Discussion

The model trained with weighted loss (`pos_weight=3`) achieved the following results on the validation set (example from one run):

*   **Accuracy:** ~80.5%
*   **Precision:** ~54.5%
*   **Recall:** ~83.5%
*   **Confusion Matrix:** `[[1657 TN, 422 FP], [100 FN, 505 TP]]`

**Analysis:**
The dataset exhibits a significant class imbalance. By implementing weighted loss, the model was trained to prioritize detecting the minority pneumonia class. This is reflected in the **high recall (83.5%)**, indicating that the model successfully identified most of the actual pneumonia cases, resulting in a low number of False Negatives (100). This is often crucial in medical applications where missing a positive case can have serious consequences.

However, this focus on recall comes at the cost of **lower precision (54.5%)**. The model is more sensitive and thus generates a higher number of False Positives (422), classifying non-pneumonia cases as pneumonia more often. This project demonstrates the practical precision-recall trade-off when dealing with imbalanced data and highlights how adjusting loss weighting can tune the model's behavior based on application priorities. While overall accuracy is a useful metric, precision and recall provide a more nuanced understanding of the model's performance on this specific task.

## Future Work

*   Perform more extensive hyperparameter tuning (learning rate, weight decay, optimizer settings).
*   Experiment with different data augmentation strategies.
*   Explore alternative methods for handling class imbalance (e.g., oversampling, undersampling, Focal Loss).
*   Train and evaluate different model architectures (e.g., ResNet-34, DenseNet).
*   Implement k-fold cross-validation for more robust performance metrics.
*   **(If applicable)** Integrate and evaluate Class Activation Maps (CAMs) for model interpretability.

## Skills Demonstrated

*   **Programming:** Python
*   **Libraries:** PyTorch, PyTorch Lightning, Torchvision, Torchmetrics, NumPy, Pandas, Matplotlib, Pydicom, OpenCV (cv2)
*   **ML Concepts:** Deep Learning, Medical Image Analysis, Image Classification, Convolutional Neural Networks (CNNs), Transfer Learning (ResNet), Data Preprocessing & Augmentation, Handling Class Imbalance (Weighted Loss), Model Training & Evaluation (Accuracy, Precision, Recall, Confusion Matrix), BCE Loss, Adam Optimizer
*   **Tools:** Jupyter Notebooks, Git/GitHub, TensorBoard