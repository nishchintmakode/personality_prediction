import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dense, GlobalMaxPooling1D
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Load the myPersonality dataset with sentiment scores
myp_data = pd.read_csv('myp_dataset_with_sentiment.csv', encoding='latin-1')

# Check for missing values in the 'STATUS' column
print(myp_data['STATUS'].isnull().sum())

# Convert 'STATUS' column to string
myp_data['STATUS'] = myp_data['STATUS'].astype(str)

# Tokenize the text data and create sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(myp_data['STATUS'])
sequences = tokenizer.texts_to_sequences(myp_data['STATUS'])

# Pad sequences to have a fixed length
max_seq_length = 100
X_text = pad_sequences(sequences, maxlen=max_seq_length)

# Prepare the target variables (personality traits)
y = myp_data[['sEXT', 'sNEU', 'sAGR', 'sCON', 'sOPN']].values

# Prepare the sentiment scores as additional features for personality detection
X_sentiment = myp_data['SENTIMENT'].values

# Combine text and sentiment features
X_combined = np.column_stack((X_text, X_sentiment))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Create the model architecture
embedding_dim = 100
num_filters = 128
kernel_size = 5
lstm_units = 64

model = Sequential()
model.add(Embedding(len(tokenizer.word_index) + 1, embedding_dim, input_length=max_seq_length + 1))  # +1 for the additional sentiment feature
model.add(Conv1D(num_filters, kernel_size, activation='relu'))
model.add(MaxPooling1D())
model.add(LSTM(lstm_units))
model.add(Dense(5))  # 5 outputs for the Big Five personality traits

# Compile the model
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# Train the model
epochs = 20
batch_size = 64
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

# Save the trained model to a file
model.save('final_personality_detection_model.keras')

# Load the saved model for evaluation (you can skip this if the model is already loaded)
saved_model = load_model('final_personality_detection_model.keras')

# Convert X_test back to a list of strings
X_test_text = [str(text) for text in X_test]

# Make predictions on the test data
test_sequences = tokenizer.texts_to_sequences(X_test_text)
max_length = max(len(sequence) for sequence in test_sequences)
X_test_padded = pad_sequences(test_sequences, maxlen=max_length)
y_pred = saved_model.predict(X_test_padded)

# Convert the predicted scores back to the original scale (if needed)
y_pred = (y_pred * y_train.std(axis=0)) + y_train.mean(axis=0)

# Round the predicted scores to get binary predictions (if needed)
# For example, if you want to classify personality traits as high/low based on a threshold:
threshold = 3.5
y_pred_binary = np.where(y_pred >= threshold, 1, 0)

# Convert y_test to binary format using the same threshold
y_test_binary = np.where(y_test >= threshold, 1, 0)

# Calculate evaluation metrics
precision = precision_score(y_test_binary, y_pred_binary, average='macro')
recall = recall_score(y_test_binary, y_pred_binary, average='macro')
f_measure = f1_score(y_test_binary, y_pred_binary, average='macro')
accuracy = accuracy_score(y_test_binary, y_pred_binary)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F-measure: {f_measure:.2f}")