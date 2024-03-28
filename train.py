import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer


## Function to preprocess the text like- tokenize the text, removing the stop words, lowering the case, removing URLs etc .
def preprocess_text(text):
    # Step 1: Convert to lowercase
    text = text.lower()
    
    pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    # Step 2: replace matched URLs with an empty string
    text = re.sub(pattern, '', text)
    
    # Step 3: Remove hyperlinks
    text = re.sub(r'http\S+|www\S+|https\S+|\d+|[^A-Za-z\s]+', '', text)

    # Step 4: Tokenization
    tokens = word_tokenize(text)

    # Step 5: Remove Punctuation and Special Characters
    tokens = [re.sub(r'[^a-zA-Z0-9]', '', token) for token in tokens]

    # Step 6: Remove Stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Step 7: Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]

    # Step 8: Remove Numerical Values
    tokens = [token for token in tokens if not token.isdigit()]

    # Step 9: Remove Single Character Tokens
    tokens = [token for token in tokens if len(token) > 1]

    # Step 10: Join the tokens back into a single string
    preprocessed_text = ' '.join(tokens)
    
    preprocessed_text=re.sub(r'^.*?subject','',preprocessed_text)

    return preprocessed_text


train_data = pd.read_csv('train.csv')
X_train_sent, y_train = train_data['Processed_text'], train_data['spam']

# Initialize a CountVectorizer object
vectorizer = CountVectorizer()

# Fitting the CountVectorizer to the training text data 
vectorizer.fit(X_train_sent)

# Define a function named data_prep that takes a single argument msg and prepare the input for the model to work on
def data_prep(msg):

    msg = preprocess_text(msg)  # Preprocess the input message
    k = [msg]  # Create a list containing the preprocessed message
    testing_data = pd.Series(k) # Convert the list k into a Pandas Series object named testing_data
    
    # Use the fitted CountVectorizer to transform the preprocessed message into a sparse matrix of token counts
    text_vector = vectorizer.transform(testing_data)
    
    # Return the transformed text data as a sparse matrix
    return text_vector
