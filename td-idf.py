import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import pearsonr

# Load your data
data = pd.read_csv('resource/asnlib/publicdata/dev/train_data1.csv')
labels = pd.read_csv('resource/asnlib/publicdata/dev/train_label1.csv')['BEDS']  # Assuming column name is 'BEDS'

# List of your open-ended text columns
#open_ended_columns = ['BROKERTITLE', 'ADDRESS', 'STATE', 'MAIN_ADDRESS', 'STREET_NAME', 'LONG_NAME', 'FORMATTED_ADDRESS']
#BROKERTITLE,TYPE,PRICE,BATH,PROPERTYSQFT,ADDRESS,STATE,MAIN_ADDRESS,ADMINISTRATIVE_AREA_LEVEL_2,LOCALITY,SUBLOCALITY,STREET_NAME,LONG_NAME,FORMATTED_ADDRESS,LATITUDE,LONGITUDE
open_ended_columns = ['TYPE']
# Function to preprocess text data similarly to your C++ code
def preprocess_text(text):
    # Tokenize by space and convert to lowercase
    tokens = text.lower().split()
    return ' '.join(tokens)

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(min_df=1)

# Dictionary to store correlations for each word
word_correlations = {}

for column in open_ended_columns:
    # Apply preprocessing to each open-ended text entry in the current column
    preprocessed_text = data[column].astype(str).apply(preprocess_text)
    
    # Fit and transform the preprocessed text data
    tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_text)
    
    # Get feature names (words)
    feature_names = tfidf_vectorizer.get_feature_names_out()
    
    # Iterate over each word
    for word in feature_names:
        # Calculate TF-IDF scores for the current word
        tfidf_scores = tfidf_matrix[:, tfidf_vectorizer.vocabulary_[word]].toarray().flatten()
        
        # Calculate correlation with number of beds
        correlation, _ = pearsonr(labels, tfidf_scores)
        
        # Store correlation for the word
        if word in word_correlations:
            word_correlations[word].append(correlation)
        else:
            word_correlations[word] = [correlation]

# Average correlation for each word
average_word_correlations = {word: np.mean(correlations) for word, correlations in word_correlations.items()}

# Sort words based on average correlation
sorted_words = sorted(average_word_correlations.items(), key=lambda x: x[1], reverse=True)

# Print top N words and their average correlations
top_n = 10  # Adjust as needed
for word, correlation in sorted_words[-top_n:]:
    print(f"Word: {word}, Average correlation with number of beds: {correlation}")
    
print(" switch ")
for word, correlation in sorted_words[:top_n]:
    print(f"Word: {word}, Average correlation with number of beds: {correlation}")
