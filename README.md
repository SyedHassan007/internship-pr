# 🍽️ Smart Spoon - AI-Powered Food Recognition App

Smart Spoon is a machine learning-powered web application that recognizes food from images, estimates salt content, and collects user feedback for analysis. Built with TensorFlow, Streamlit, and NLP tools.


## 🧰 Tech Stack

- **Frontend**: Streamlit  
- **Model**: MobileNetV2 (fine-tuned)  
- **Backend**: Python (Keras, TensorFlow, Numpy, Pandas)  
- **Data Handling**: CSV + Excel  
- **Visualization**: Matplotlib, Seaborn  
- **Sentiment Analysis**: NLTK VADER  

---

## 🧠 Model Summary

- **Base Model**: MobileNetV2  
- **Input Size**: 224x224x3  
- **Output**: Softmax (5 food categories)  
- **Custom Layers**:  
  - GlobalAveragePooling2D  
  - Dense (256, ReLU)  
  - Dropout (0.5 + 0.3)  
  - Dense (NUM_CLASSES, softmax)  
- **Training Enhancements**:  
  - Data Augmentation  
  - Class Weighting  
  - EarlyStopping, ReduceLROnPlateau  

---

## 🚀 Usage Guide

### Step 1: Install Requirements

```bash
pip install -r requirements.txt
```

### Step 2: Train the Model

```bash
python train_food_model.py
```

### Step 3: Run the Web App

```bash
streamlit run streamlit_app.py
```




## 📁 Directory Structure

```
smart_spoon_project/
├── data/
├── food_recognition/
├── market_analysis/
├── sentiment_analysis/
├── streamlit_app.py
├── train_food_model.py
├── evaluate_model.py
├── requirements.txt
└── README.md
```

---

## 📬 Feedback & Contributions

Feel free to fork or submit pull requests for improvements.  
Smart Spoon is always learning—just like you!

---

# Smart-Spoon
