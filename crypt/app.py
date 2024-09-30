from flask import Flask, render_template, request
import joblib

# Load the saved model and vectorizer
model = joblib.load('cipher_classification_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

app = Flask(__name__)

# Algorithm descriptions
ALGORITHM_DESCRIPTIONS = {
    'AES': "The Advanced Encryption Standard (AES) is an algorithm that uses the same key to encrypt and decrypt protected data. Instead of a single round of encryption, data is put through several rounds of substitution, transposition, and mixing to make it harder to compromise.",
    'DES': "The Data Encryption Standard (DES) is a symmetric-key algorithm for the encryption of digital data. DES uses the same key for both encryption and decryption and has been widely used in the past.",
    'ECC': "Elliptic Curve Cryptography (ECC) is used to create faster, smaller, and more efficient cryptographic keys. It provides the same level of security as other algorithms but with smaller key sizes.",
    'RSA': "RSA is a public-key cryptosystem that is widely used for secure data transmission. It is based on the mathematical problem of factoring large numbers, making it hard to crack.",
    'SHA-256': "SHA-256 is a cryptographic hash function that generates a unique 256-bit hash value for data. It is used in various security applications and protocols, including SSL/TLS and blockchain."
}

# Function to predict the likelihood of algorithms
def predict_cipher_algorithm(ciphertext):
    # Transform the input ciphertext using the saved TF-IDF vectorizer
    ciphertext_tfidf = tfidf_vectorizer.transform([ciphertext])
    
    # Predict the algorithm probabilities
    probabilities = model.predict_proba(ciphertext_tfidf)[0]
    
    # Get the class labels (algorithms) from the model
    classes = model.classes_
    
    # Create a dictionary of algorithm names and their probabilities
    results = {classes[i]: round(prob * 100, 2) for i, prob in enumerate(probabilities)}
    
    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the input ciphertext from the form
        ciphertext = request.form['ciphertext']
        
        # Get the prediction results
        results = predict_cipher_algorithm(ciphertext)
        
        # Determine the highest likelihood algorithm
        predicted_algorithm = max(results, key=results.get)
        
        # Get the description of the predicted algorithm
        algorithm_description = ALGORITHM_DESCRIPTIONS.get(predicted_algorithm, "")

        # Render the results, predicted algorithm, and description on the page
        return render_template('index.html', results=results, ciphertext=ciphertext,
                               predicted_algorithm=predicted_algorithm, algorithm_description=algorithm_description)

if __name__ == '__main__':
    app.run(debug=True)
