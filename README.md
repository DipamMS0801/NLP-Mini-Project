# 📰 CyberAware — Fake News Detection Using NLP & Machine Learning

<p align="center">
  <strong>Detect fake and real news articles using Natural Language Processing (NLP) and Machine Learning.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Machine%20Learning-SVM-orange" alt="Machine Learning">
  <img src="https://img.shields.io/badge/NLP-TF--IDF-purple" alt="NLP">
  <img src="https://img.shields.io/badge/Interface-Tkinter-green" alt="Tkinter">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="MIT License">
</p>

---

## 📌 Overview

**CyberAware** is a Natural Language Processing (NLP) and Machine Learning mini-project designed to classify news articles as **Fake** or **Real** based on their textual content.

The project uses TF-IDF (Term Frequency–Inverse Document Frequency) to convert text into numerical features and a Linear Support Vector Machine (SVM) classifier to make predictions.

It includes a desktop GUI built with Tkinter, allowing users to enter news text, view the predicted classification, and check the model's reported accuracy.

The project demonstrates a practical machine learning workflow, from dataset preparation and text vectorization to model training and interactive prediction.

## ✨ Features

- 📰 **Fake News Classification:** Predicts whether an article is fake or real.
- 🧠 **NLP-Based Text Processing:** Converts textual content into numerical features using TF-IDF.
- 🤖 **Linear SVM Classifier:** Uses a supervised machine learning model for binary classification.
- 🖥️ **Interactive Desktop GUI:** Provides a simple interface built with Python Tkinter.
- 🔎 **Keyword Display:** Shows selected words from the input that are represented in the TF-IDF vocabulary.
- 📊 **Model Evaluation:** Calculates accuracy and displays a confusion matrix during training.
- 💾 **Model Persistence:** Saves the trained model, vectorizer, and accuracy using Python Pickle.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Dataset loading and preparation |
| Scikit-learn | TF-IDF, SVM training, data splitting and evaluation |
| Matplotlib | Confusion matrix and accuracy visualization |
| Tkinter | Desktop graphical user interface |
| Pickle | Saving and loading trained model artifacts |

## 📂 Project Structure

```text
NLP-Mini-Project/
│
├── app.py               # Tkinter desktop application
├── prepare_dataset.py   # Combines and labels the datasets
├── train_model.py       # TF-IDF, SVM training and evaluation
├── README.md            # Project documentation
├── LICENSE              # MIT License
│
├── Fake.csv             # Fake news dataset (download separately)
├── True.csv             # Real news dataset (download separately)
├── final_dataset.csv    # Generated combined dataset
│
├── model.pkl            # Generated trained SVM model
├── vectorizer.pkl       # Generated TF-IDF vectorizer
└── accuracy.pkl         # Generated model accuracy
```

*The CSV files and generated model artifacts may not be included in the repository. Follow the setup instructions below to generate them locally.*

## 📊 Dataset

This project uses the **Fake and Real News Dataset**, commonly known as the ISOT Fake News Dataset.

**Source:** [Download the Fake and Real News Dataset from Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

The dataset contains two primary files:

| File | Description |
|---|---|
| `Fake.csv` | News articles labelled as fake |
| `True.csv` | News articles labelled as real |

### Dataset preparation

The `prepare_dataset.py` script:

1. Loads `Fake.csv` and `True.csv`.
2. Assigns the corresponding `fake` and `real` labels.
3. Combines the two datasets.
4. Retains the `text` and `label` columns.
5. Exports the combined data to `final_dataset.csv`.

## ⚙️ How It Works

The project follows a simple machine learning pipeline:

```text
News Dataset
     │
     ▼
Dataset Preparation
(Fake.csv + True.csv)
     │
     ▼
Combined Dataset
(final_dataset.csv)
     │
     ▼
Sample 10,000 Articles
     │
     ▼
Train-Test Split
(80% Training / 20% Testing)
     │
     ▼
TF-IDF Vectorization
(Up to 3,000 Features)
     │
     ▼
Linear SVM Classifier
     │
     ▼
Model Evaluation
(Accuracy + Confusion Matrix)
     │
     ▼
Save Model Artifacts
     │
     ▼
CyberAware Desktop GUI
     │
     ▼
Fake News / Real News Prediction
```

### 1. Dataset preparation

The two source files are combined into a single dataset with text and binary classification labels.

### 2. Feature extraction with TF-IDF

TF-IDF transforms news text into numerical feature vectors. The implementation uses English stop-word removal, a maximum of 3,000 features, and unigrams and bigrams.

### 3. Model training with SVM

A Linear SVM classifier is trained on the vectorized text. The training script samples 10,000 records using a fixed sampling seed and uses an 80:20 train-test split.

### 4. Model evaluation

The model predicts labels for the test set. Accuracy and a confusion matrix are calculated to evaluate its performance.

### 5. Desktop application

The Tkinter interface loads the saved model artifacts and lets users submit news text for classification.

## 🚀 Installation & Setup

Follow these steps to run CyberAware locally.

### Prerequisites

- Python 3.x
- pip
- Git
- The dataset files downloaded from Kaggle

### Step 1: Clone the repository

```bash
git clone https://github.com/DipamMS0801/NLP-Mini-Project.git
cd NLP-Mini-Project
```

### Step 2: Create a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can run the commands below instead:

```powershell
.\venv\Scripts\python.exe -m pip install --upgrade pip
```

**macOS/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install dependencies

```bash
python -m pip install pandas scikit-learn matplotlib
```

Tkinter is included with many standard Python installations. On some Linux distributions, it must be installed separately.

### Step 4: Download the dataset

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset).
2. Extract the archive.
3. Place `Fake.csv` and `True.csv` in the project root directory.

Your directory should contain:

```text
NLP-Mini-Project/
├── app.py
├── prepare_dataset.py
├── train_model.py
├── Fake.csv
└── True.csv
```

### Step 5: Prepare the dataset

```bash
python prepare_dataset.py
```

This generates `final_dataset.csv`.

### Step 6: Train the model

```bash
python train_model.py
```

This script trains the model, prints the evaluation results, displays the confusion matrix, and saves the required model artifacts.

### Step 7: Launch CyberAware

```bash
python app.py
```

The desktop application should open, allowing you to enter news text and view the prediction.

**Important:** Run the preparation and training scripts before launching the GUI, because the application requires the generated `.pkl` files.

## 🖥️ Using the Application

1. Launch the application.
2. Enter a news article or a relevant piece of news text.
3. Click **Check News** to receive the model's prediction.
4. Review the displayed classification and selected keywords.
5. Use **Clear** to reset the input.
6. Use **Show Accuracy Graph** to display the application's accuracy visualization.

## 📈 Evaluation

The model is evaluated using a held-out test set.

The training script reports:

- **Accuracy:** The proportion of test examples classified correctly.
- **Confusion Matrix:** A breakdown of actual versus predicted labels.

The measured accuracy is generated when you train the model. Results can vary with the data split and environment.

*Note:* The GUI's accuracy graph is illustrative; it constructs a sequence of values from the final accuracy rather than recording actual accuracy across training epochs.

## 🎯 Learning Outcomes

This project provides hands-on experience with:

- Text classification using NLP.
- TF-IDF feature extraction.
- Supervised machine learning with SVM.
- Dataset preparation and label encoding.
- Train-test splitting and model evaluation.
- Saving and reusing trained ML models.
- Building desktop applications with Tkinter.
- Integrating a machine learning pipeline into a user interface.

## ⚠️ Limitations

- Predictions depend on the quality and representativeness of the training dataset.
- The classifier evaluates textual patterns, not the factual accuracy of claims against verified sources.
- The displayed keywords are selected from the input's TF-IDF vocabulary; they should not be interpreted as formal model explanations.
- Accuracy on a held-out dataset does not guarantee reliable performance on new or evolving news sources.
- This project is an educational demonstration and should not be used as the sole basis for determining whether a news article is trustworthy.

## 🔮 Future Improvements

Potential enhancements include:

- Advanced text preprocessing and linguistic normalization.
- Comparison with Logistic Regression, Naive Bayes and other classifiers.
- Hyperparameter tuning and cross-validation.
- Precision, recall, F1-score and ROC-AUC reporting.
- Improved interpretability using model explanation techniques.
- A web interface built with Streamlit or Flask.
- Integration with trusted fact-checking sources.
- Automated tests and dependency management.

## 🤝 Contributing

Contributions and suggestions are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Open a pull request describing your improvements.

## 📜 License

This project is distributed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Dipam M. Shendurkar**

- GitHub: [@DipamMS0801](https://github.com/DipamMS0801)
- Project Repository: [NLP-Mini-Project](https://github.com/DipamMS0801/NLP-Mini-Project)

---

<p align="center">
  <strong>Built with Python, NLP, and Machine Learning.</strong>
  <br>
  If you find this project useful, consider giving the repository a ⭐ on GitHub!
</p>
