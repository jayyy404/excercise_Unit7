from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import time
from datetime import datetime

corpus = """
        the cat sat on the mat while the dog ran past
        the mouse hid from the cat but the cat found it
        the dog and the cat played on the mat together
        """
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
    start_time = time.time()
    tokens = corpus.lower().split()
    correct_predictions = 0
    total_predictions = 0
    prediction_times = []

    for i in range(2, len(tokens)-1):
        pred_start_time = time.time()
        context = ' '.join(tokens[i-2:i])
        actual_next_word = tokens[i]
        predicted_next_word = generate(context)
        prediction_times.append(time.time() - pred_start_time)
        
        if predicted_next_word == actual_next_word:
            correct_predictions += 1
        total_predictions += 1

    total_time = time.time() - start_time
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    print("\nEvaluation Metrics:")
    print(f"Total Evaluation Time: {total_time:.2f} seconds")
    print(f"Average Prediction Time: {(sum(prediction_times)/len(prediction_times))*1000:.2f} ms")
    print(f"Number of Predictions: {total_predictions}")
    
    return accuracy, prediction_times

def measure_performance(model, corpus, tokenizer, num_predictions=100):
    metrics = {
        'training_time': 0,
        'inference_time': [],
        'total_parameters': model.count_params()
    }
    
    # Measure training time
    start_time = time.time()
    model.fit(X, y, epochs=200, verbose=0)
    metrics['training_time'] = time.time() - start_time
    
    # Measure inference time
    test_phrase = "the cat"
    for _ in range(num_predictions):
        start_time = time.time()
        _ = generate(test_phrase)
        inference_time = time.time() - start_time
        metrics['inference_time'].append(inference_time)
    
    # Calculate statistics
    avg_inference_time = sum(metrics['inference_time']) / len(metrics['inference_time'])
    max_inference_time = max(metrics['inference_time'])
    min_inference_time = min(metrics['inference_time'])
    
    print("\nPerformance Metrics:")
    print(f"Total Training Time: {metrics['training_time']:.2f} seconds")
    print(f"Average Inference Time: {avg_inference_time*1000:.2f} ms")
    print(f"Min Inference Time: {min_inference_time*1000:.2f} ms")
    print(f"Max Inference Time: {max_inference_time*1000:.2f} ms")
    print(f"Model Parameters: {metrics['total_parameters']:,}")
    
    return metrics

# Run the performance measurements
perf_metrics = measure_performance(model, corpus, tokenizer)
accuracy, pred_times = evaluate_model(model, corpus, tokenizer)

print(f"Model accuracy: {accuracy:.2f}")
print("Prediction for 'the cat':", generate("the cat"))


