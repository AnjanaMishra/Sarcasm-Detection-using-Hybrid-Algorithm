print("\nEvaluating on test set...")
test_loss, test_accuracy = final_model.evaluate(test_data, test_labels)
test_predictions = final_model.predict(test_data)
test_predictions_binary = (test_predictions > 0.5).astype(int)

from sklearn.metrics import precision_score, recall_score, f1_score
precision = precision_score(y_test, test_predictions_binary, average='weighted')
recall = recall_score(y_test, test_predictions_binary, average='weighted')
f1 = f1_score(y_test, test_predictions_binary, average='weighted')

print("\nPerformance of OPTIMIZED-TL-BERT-MHA-B-LSTM MODEL")
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
