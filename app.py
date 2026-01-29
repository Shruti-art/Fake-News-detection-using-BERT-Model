import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, Bidirectional
import pickle
import os

# Function to load data
@st.cache_data
def load_data():
    true_df = pd.read_csv('true.csv')
    fake_df = pd.read_csv('fake.csv')
    true_df['true'] = 1
    fake_df['true'] = 0
    return true_df, fake_df

# Global variable for maximum length of sequences
MAX_LEN = 20

# Function to preprocess data
def preprocess_data(true_df, fake_df):
    df = pd.concat([true_df, fake_df])
    df = shuffle(df).reset_index(drop=True)

    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(df['title'])

    sequences = tokenizer.texts_to_sequences(df['title'])
    padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')

    train_size = int(len(df) * 0.8)
    val_size = int(train_size * 0.8)
    train_sequences = padded_sequences[:val_size]
    val_sequences = padded_sequences[val_size:train_size]
    test_sequences = padded_sequences[train_size:]

    y_train = df['true'][:val_size]
    y_val = df['true'][val_size:train_size]
    y_test = df['true'][train_size:]

    return train_sequences, y_train, val_sequences, y_val, test_sequences, y_test, tokenizer

# Function to build the model
def build_model():
    model = Sequential()
    model.add(Embedding(input_dim=5000, output_dim=40))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(100)))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Load data
true_df, fake_df = load_data()

# Sidebar for navigation
st.sidebar.title("Fake News Detection")
option = st.sidebar.selectbox("Choose an option", ["EDA", "Model Training", "Prediction", "News Detection"])

# EDA
if option == "EDA":
    st.title("Exploratory Data Analysis")

    st.subheader("True News Subject Distribution")
    fig_true = plt.figure()
    sns.countplot(y="subject", palette="coolwarm", data=true_df).set_title('True News Subject Distribution')
    st.pyplot(fig_true)

    st.subheader("Fake News Subject Distribution")
    fig_fake = plt.figure()
    sns.countplot(y="subject", palette="coolwarm", data=fake_df).set_title('Fake News Subject Distribution')
    st.pyplot(fig_fake)

    st.subheader("Word Cloud for True News Titles")
    real_all_words = ' '.join(true_df.title)
    wordcloud_real = WordCloud(background_color='white', width=800, height=500, max_font_size=180, collocations=False).generate(real_all_words)
    fig_wordcloud_true = plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud_real, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(fig_wordcloud_true)

    st.subheader("Word Cloud for Fake News Titles")
    fake_all_words = ' '.join(fake_df.title)
    wordcloud_fake = WordCloud(background_color='white', width=800, height=500, max_font_size=180, collocations=False).generate(fake_all_words)
    fig_wordcloud_fake = plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud_fake, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(fig_wordcloud_fake)

# Model Training
elif option == "Model Training":
    st.title("Model Training")

    # Preprocess data
    train_sequences, y_train, val_sequences, y_val, test_sequences, y_test, tokenizer = preprocess_data(true_df, fake_df)

    # Save the tokenizer for later use
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)

    # Build model
    model = build_model()
    model.summary(print_fn=lambda x: st.text(x))

    # Train model
    epochs = st.slider("Number of epochs", min_value=1, max_value=10, value=3)
    if st.button("Train Model"):
        history = model.fit(train_sequences, y_train, batch_size=64, validation_data=(val_sequences, y_val), epochs=epochs)
        st.success("Model trained successfully")

        # Save the model in Keras format
        model.save('fake_news_model.keras')

        # Plot training history
        st.subheader("Training and Validation Loss")
        fig_loss, ax_loss = plt.subplots()
        ax_loss.plot(history.history['loss'], label='Training Loss')
        ax_loss.plot(history.history['val_loss'], label='Validation Loss')
        ax_loss.legend()
        st.pyplot(fig_loss)

        st.subheader("Training and Validation Accuracy")
        fig_accuracy, ax_accuracy = plt.subplots()
        ax_accuracy.plot(history.history['accuracy'], label='Training Accuracy')
        ax_accuracy.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax_accuracy.legend()
        st.pyplot(fig_accuracy)

# Prediction
elif option == "Prediction":
    st.title("Model Prediction")

    # Preprocess data
    train_sequences, y_train, val_sequences, y_val, test_sequences, y_test, tokenizer = preprocess_data(true_df, fake_df)

    # Build and load trained model
    model = build_model()
    if st.button("Load Trained Model"):
        if os.path.exists('fake_news_model.keras'):
            model = load_model('fake_news_model.keras')
            st.success("Model loaded successfully")
        else:
            st.error("Model file not found. Please train the model first.")

    # Make predictions
    if st.button("Predict"):
        if os.path.exists('fake_news_model.keras'):
            predictions = (model.predict(test_sequences) > 0.5).astype("int32")
            accuracy = accuracy_score(y_test, predictions)
            st.write("Model Accuracy:", accuracy)

            # Confusion matrix
            cm = confusion_matrix(y_test, predictions)
            st.subheader("Confusion Matrix")
            sns.heatmap(cm, annot=True, fmt='d')
            st.pyplot()

            # Classification report
            st.subheader("Classification Report")
            report = classification_report(y_test, predictions, output_dict=True)
            st.write(report)
        else:
            st.error("Model file not found. Please train the model first.")

# News Detection
elif option == "News Detection":
    st.title("Detect Fake News")

    user_input = st.text_area("Enter the news title:")
    if st.button("Detect"):
        if user_input:
            # Load the tokenizer
            if os.path.exists('tokenizer.pkl'):
                with open('tokenizer.pkl', 'rb') as f:
                    tokenizer = pickle.load(f)

                # Preprocess user input
                sequence = tokenizer.texts_to_sequences([user_input])
                padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')

                # Load and predict
                if os.path.exists('fake_news_model.keras'):
                    model = load_model('fake_news_model.keras')
                    prediction = model.predict(padded_sequence)
                    if prediction > 0.5:
                        st.write("The news is True")
                    else:
                        st.write("The news is Fake")
                else:
                    st.error("Model file not found. Please train the model first.")
            else:
                st.error("Tokenizer file not found. Please train the model first.")
        else:
            st.warning("Please enter a news title")
