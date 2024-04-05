import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dense, GlobalMaxPooling1D
from sklearn.model_selection import train_test_split

# Load the STS dataset
sts_data = pd.read_csv('preprocessed_sts_dataset.csv', encoding='latin-1')

# Check for missing values in the 'text' column
print(sts_data['text'].isnull().sum())

# Convert 'text' column to string
sts_data['text'] = sts_data['text'].astype(str)

# Tokenize the text data and create sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sts_data['text'])
sequences = tokenizer.texts_to_sequences(sts_data['text'])

# Pad sequences to have a fixed length
max_seq_length = 100
X_text = pad_sequences(sequences, maxlen=max_seq_length)

# Prepare the target variable (sentiment label, 0=negative, 1=positive)
y = sts_data['sentiment'].apply(lambda score: 1 if score >= 0 else 0).values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_text, y, test_size=0.2, random_state=42)

# Create the sentiment analysis model
embedding_dim = 100
num_filters = 128
kernel_size = 5
lstm_units = 64

sentiment_model = Sequential()
sentiment_model.add(Embedding(len(tokenizer.word_index) + 1, embedding_dim, input_length=max_seq_length))
sentiment_model.add(Conv1D(num_filters, kernel_size, activation='relu'))
sentiment_model.add(MaxPooling1D())
sentiment_model.add(LSTM(lstm_units))
sentiment_model.add(Dense(1, activation='sigmoid'))

# Compile the model
sentiment_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
epochs = 5
batch_size = 64
sentiment_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

# Save the trained sentiment analysis model to a file
sentiment_model.save('sentiment_analysis_model.h5')