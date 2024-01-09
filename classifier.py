import pandas as pd 
import numpy as np 
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense







df = pd.read_csv('Cleaned Review.csv')

tokenizer = Tokenizer(num_words=1000)  

tokenizer.fit_on_texts(df['Review'])

sequences = tokenizer.texts_to_sequences(df['Review'])

max_sequence_length = 45 # You might need to adjust this based on the length of your reviews

padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

labels = df['Liked'].values

model = Sequential()
model.add(Embedding(input_dim=1000, output_dim=16, input_length=max_sequence_length))
model.add(Bidirectional(LSTM(32)))  # Using a Bidirectional LSTM layer
model.add(Dense(32, activation='relu'))  # You can adjust the number of neurons
model.add(Dense(1, activation='sigmoid'))  # Output layer

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

history = model.fit(X_train, y_train, epochs=20, batch_size=10, validation_split=0.2)  


loss, accuracy = model.evaluate(X_test, y_test)

print(f"Test Accuracy: {accuracy * 100}%")

model.save('restaurant_review_model.h5')




def predict_review_sentiment(review, model, tokenizer, max_sequence_length):
    # Tokenize and preprocess the review
    sequence = tokenizer.texts_to_sequences([review])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length)
    
    # Make a prediction
    prediction = model.predict(padded_sequence)
    
    # Interpret the prediction
    if prediction[0][0] >= 0.5:
        return "Positive review (1)"
    else:
        return "Negative review (0)"

# Example review
input_review = "I love your pizzagit "
# Use the prediction function
result = predict_review_sentiment(input_review, model, tokenizer, max_sequence_length)
print(result)





