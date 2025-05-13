from collections import defaultdict, Counter
import random
import time
from datetime import datetime

def train_ngram_model(text, n=3):
    start_time = time.time()
    model = defaultdict(Counter)
    tokens = text.split()
    for i in range(len(tokens)-n+1):
        context = tuple(tokens[i:i+n-1])
        next_word = tokens[i+n-1]
        model[context][next_word] += 1
    training_time = time.time() - start_time
    return model, training_time

def predict_next_word(model, context):
    context = tuple(context)
    if context in model:
        return model[context].most_common(1)[0][0]
    else:
        return "<UNK>"

def measure_performance(model, text, num_predictions=100):
    metrics = {
        'training_time': 0,
        'inference_time': [],
        'model_size': len(model)
    }
    
    # Measure inference time
    test_context = ["the", "cat"]
    for _ in range(num_predictions):
        start_time = time.time()
        _ = predict_next_word(model, test_context)
        inference_time = time.time() - start_time
        metrics['inference_time'].append(inference_time)
    
    # Calculate statistics
    avg_inference_time = sum(metrics['inference_time']) / len(metrics['inference_time'])
    max_inference_time = max(metrics['inference_time'])
    min_inference_time = min(metrics['inference_time'])
    
    print("\nPerformance Metrics:")
    print(f"Average Inference Time: {avg_inference_time*1000:.2f} ms")
    print(f"Min Inference Time: {min_inference_time*1000:.2f} ms")
    print(f"Max Inference Time: {max_inference_time*1000:.2f} ms")
    print(f"Model Size (unique contexts): {metrics['model_size']}")
    
    return metrics

def evaluate_ngram_model(model, text, n=3):
    start_time = time.time()
    tokens = text.split()
    correct_predictions = 0
    total_predictions = 0
    prediction_times = []

    for i in range(len(tokens)-n):
        pred_start_time = time.time()
        context = tuple(tokens[i:i+n-1])
        actual_next_word = tokens[i+n-1]
        predicted_next_word = predict_next_word(model, context)
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

# Run the model with performance measurements
text = """
        the cat sat on the mat while the dog ran past
        the mouse hid from the cat but the cat found it
        the dog and the cat played on the mat together
        """
model, training_time = train_ngram_model(text, n=3)
print(f"\nTraining Time: {training_time:.2f} seconds")

# Measure performance
perf_metrics = measure_performance(model, text)
accuracy, pred_times = evaluate_ngram_model(model, text, n=3)

print(f"\nModel accuracy: {accuracy:.2f}")
print("Prediction for 'the cat':", predict_next_word(model, ["the", "cat"]))


