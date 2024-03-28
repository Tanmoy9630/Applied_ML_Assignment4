from flask import Flask, request, render_template, url_for, redirect
import joblib
import score

app = Flask(__name__)

# Load the pre-trained model
filename = r"best_model.joblib"
best_model = joblib.load(filename)

# Set the threshold for classification
threshold = 0.5

@app.route('/') 
def home():
    # Render the home page
    return render_template('spam_page.html')

@app.route('/spam', methods=['POST'])
def spam():
    # Get the input text from the form
    txt = request.form['sent']
    
    # Get the prediction and propensity score using the pre-trained model
    pred,prop = score.score(txt, best_model, threshold)
    
    # Determine the label based on the prediction
    label = "Spam" if pred == 1 else "Not spam"
    
    # Generate the response message
    ans = f"""The sentence "{txt}" is {label} with propensity {prop}."""
    
    # Render the result page with the response message
    return render_template('result_page.html', ans = ans)

if __name__ == '__main__': 
    # Run the Flask app
    app.run(debug=True, use_reloader=False)
