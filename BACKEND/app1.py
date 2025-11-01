from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from fuzzywuzzy import process  # Import fuzzy matching library

app = Flask(__name__)
CORS(app)

# MySQL Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://Thiraj:Thiraj_05@localhost/elephant_data'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Define a model for storing support queries and responses
class SupportQuery(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_message = db.Column(db.String(500), nullable=False)
    bot_reply = db.Column(db.String(500), nullable=False)

@app.route("/support_chat", methods=["POST"])
def support_chat():
    data = request.get_json()
    user_message = data.get("user_message", "").lower().strip()

    # Fetch all possible queries from DB
    all_queries = SupportQuery.query.all()
    query_dict = {q.user_message: q.bot_reply for q in all_queries}

    # Check for an exact match first
    if user_message in query_dict:
        bot_reply = query_dict[user_message]
    else:
        # Use fuzzy matching to find the best match
        best_match, score = process.extractOne(user_message, query_dict.keys()) if query_dict else (None, 0)

        # Set a confidence threshold for matching
        if best_match and score > 80:  # Adjusted from 70 to 80 for better accuracy
            bot_reply = query_dict[best_match]
        else:
            # Fallback response based on keywords
            if "mobile" in user_message or "phone" in user_message:
                bot_reply = "Yes! ImageBot is fully compatible with mobile devices."
            elif "signup" in user_message or "login" in user_message:
                bot_reply = "To sign up, go to the 'Login' page and click 'Sign Up'."
            elif "image" in user_message or "upload" in user_message:
                bot_reply = "To upload an image, click on the 'Upload' button in the features section."
            else:
                bot_reply = "I'm not sure about that. Try rephrasing your question!"

    return jsonify({"reply": bot_reply})



if __name__ == "__main__":
    with app.app_context():
        db.create_all()  # Ensure tables are created
    app.run(debug=True, port=5000)
