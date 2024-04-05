import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Load myPersonality dataset
myp_df = pd.read_csv('myp_dataset.csv', encoding='latin-1')

# Load STS dataset
sts_df = pd.read_csv('sts_dataset-train.csv', encoding='latin-1')

# Drop unnecessary columns from the myPersonality dataset
myp_df = myp_df.drop(columns=['#AUTHID', 'cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN', 'DATE', 'NETWORKSIZE', 'BETWEENNESS', 'NBETWEENNESS', 'DENSITY', 'BROKERAGE', 'NBROKERAGE', 'TRANSITIVITY'])

# Drop unnecessary columns from the STS dataset
sts_df = sts_df.drop(columns=['date', 'user', 'query'])

# Text cleaning function
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters and punctuation
    text = text.lower()  # Convert text to lowercase
    return text

# Text normalization functions
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def normalize_text(text):
    # Expand contractions
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"doesn\'t", "does not", text)
    text = re.sub(r"isn\'t", "is not", text)
    text = re.sub(r"can\'t", "can not", text)
    # Remove stopwords
    text = " ".join([word for word in text.split() if word.lower() not in stop_words])
    # Text stemming
    text = " ".join([stemmer.stem(word) for word in text.split()])
    return text

# Apply text cleaning and normalization to myPersonality dataset
myp_df['STATUS'] = myp_df['STATUS'].apply(clean_text)
myp_df['STATUS'] = myp_df['STATUS'].apply(normalize_text)

# Apply text cleaning and normalization to STS dataset
sts_df['text'] = sts_df['text'].apply(clean_text)
sts_df['text'] = sts_df['text'].apply(normalize_text)

print(sts_df.head())
print(sts_df.info())

print(myp_df.head())
print(myp_df.info())

# Save preprocessed myPersonality data to a new CSV file
myp_df.to_csv('preprocessed_myp_dataset.csv', index=False)

# Save preprocessed STS data to a new CSV file
sts_df.to_csv('preprocessed_sts_dataset.csv', index=False)