from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.utils import to_categorical
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Embedding, LSTM, Dense
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
import numpy as np


corpus = "the cat sat on the mat the cat ate the mouse"
tokens = corpus.lower().split()
tokenizer = Tokenizer()
tokenizer.fit_on_texts([corpus])
word_index = tokenizer.word_index


sequences = []
for i in range(2, len(tokens)):
    seq = tokens[i-2:i+1]
    encoded = tokenizer.texts_to_sequences([' '.join(seq)])[0]
    sequences.append(encoded)

sequences = np.array(sequences)
X, y = sequences[:, :-1], sequences[:, -1]
y = to_categorical(y, num_classes=len(word_index)+1)

# Building the LSTM model
model = Sequential([
    Embedding(input_dim=len(word_index)+1, output_dim=10, input_length=2),
    LSTM(50),
    Dense(len(word_index)+1, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training the model
model.fit(X, y, epochs=200, verbose=0)

def generate(seed_text):
    encoded = tokenizer.texts_to_sequences([seed_text])[0]
    encoded = pad_sequences([encoded], maxlen=2)
    pred_index = model.predict(encoded, verbose=0).argmax()
    for word, idx in word_index.items():
        if idx == pred_index:
            return word

# Evaluate the model
def evaluate_model(model, corpus, tokenizer, n=3):
    tokens = corpus.lower().split()
    correct_predictions = 0
    total_predictions = 0

    for i in range(2, len(tokens)-1):
        context = ' '.join(tokens[i-2:i])
        actual_next_word = tokens[i]
        predicted_next_word = generate(context)
        
        if predicted_next_word == actual_next_word:
            correct_predictions += 1
        total_predictions += 1

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return accuracy


accuracy = evaluate_model(model, corpus, tokenizer)
print(f"Model accuracy: {accuracy:.2f}")
print("Prediction for 'cat sat':", generate("cat sat"))
