# Movie Sentiment Classification Using SimpleRNN

This project predicts the sentiment of movie reviews using a **Simple Recurrent Neural Network (SimpleRNN)**. Sentiment classification helps understand audience reactions and can be applied to movie recommendations, review analysis, and social media monitoring.

---

## 📁 Dataset

The dataset used is the **IMDB Movie Reviews Dataset** containing:

- `review` → The text of the movie review  
- `sentiment` → Target variable: 1 = Positive, 0 = Negative  

> Only the top 10,000 most frequent words are considered. Reviews are preprocessed into integer sequences using the IMDB word index.

---

## 🛠 Tech Stack

- Python  
- NumPy, Pandas  
- TensorFlow / Keras  
- Scikit-learn  
- Streamlit  
- Matplotlib  
- TensorBoard  

---

## ⚙️ Features

- **Sequential Text Data:** Movie reviews converted to integer sequences  
- **Target:** Sentiment (binary classification: Positive / Negative)  
- **Max Review Length:** 500 words (padded sequences)

---

## 🔧 Preprocessing Steps

1. Loaded IMDB dataset with top 10,000 words.  
2. Padded all reviews to a fixed length of 500 using `pad_sequences`.  
3. Converted user input reviews to integer sequences using IMDB word index.  
4. Ensured input is preprocessed before prediction in the Streamlit app.

---

## 🧠 Model Architecture

A **SimpleRNN** model using Keras:

- **Embedding Layer:** Converts word integers to dense vectors (Vocabulary size = 10,000, Output dim = 128)  
- **SimpleRNN Layer:** 128 units, ReLU activation  
- **Dense Layer:** 1 neuron, Sigmoid activation  

**Loss Function:** Binary Crossentropy  
**Optimizer:** Adam  
**Metrics:** Accuracy  

**Callbacks:**  
- EarlyStopping (monitors `val_loss`, restores best weights)  
- TensorBoard for training visualization  

---

## 🚀 Training

- Training/Test split: 80% / 20%  
- Epochs: 10 (EarlyStopping used to avoid overfitting)  
- Batch size: 32  
- Validation split: 20%  

## Model Performance
-Training Accuracy: ~87%
-Validation Accuracy: ~86.5%
-Early stopping used to avoid overfitting

---

## 🎥 Project Demo
🎬 **Watch the demo video below:**  



https://github.com/user-attachments/assets/4536d055-c738-4b46-94b1-ab002fca41eb


git clone <your-repo-link>
cd Movie-Sentiment-Classification
