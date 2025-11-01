from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import random
from flask_cors import CORS
import mysql.connector
import urllib.parse  # For URL encoding/decoding

app = Flask(__name__)
CORS(app, resources={r"/search": {"origins": "*"}})  # Allow frontend requests

# Database Connection
try:
    db = mysql.connector.connect(
        host="localhost",
        user="Thiraj",  # Change if needed
        password="Thiraj_05",  # Change if needed
        database="elephant_data"
    )
    print("✅ Database Connected Successfully!")
except mysql.connector.Error as err:
    print(f"❌ Database Connection Error: {err}")
    db = None

# Base directory for elephant images
BASE_DIR = r"E:\IMAGERECOGNITIONCHATBOT\BACKEND\elephant_data"

# Function to get a random image from a species folder (Partial Matching)
def get_random_image(species):
    """
    Allows partial matches: e.g., typing 'asian' will match 'asian elephant'
    so you don't need the exact phrase 'asian elephant'.
    """
    species_lower = species.lower()
    # Map (in lowercase) -> actual folder name on disk
    species_folder_map = {
        "african bush elephant": "African Bush Elephant",
        "african forest elephant": "African Forest Elephant",
        "asian elephant": "Asian Elephant"
    }

    folder_path = None
    # Check if user's query is contained in any known species key
    for key, folder_name in species_folder_map.items():
        if species_lower in key:  # partial match
            folder_path = os.path.join(BASE_DIR, folder_name)
            break

    if not folder_path or not os.path.exists(folder_path):
        return None

    # Filter for valid images
    images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        return None

    random_image = random.choice(images)
    return os.path.join(folder_path, random_image)

# API to handle search request
@app.route("/search", methods=["GET"])
def search_elephant():
    query = request.args.get("query", "").strip().lower()

    if not query:
        return jsonify({"error": "No search query provided"}), 400

    if not db:
        return jsonify({"error": "Database connection failed"}), 500

    try:
        cursor = db.cursor(dictionary=True)
        sql = "SELECT * FROM elephant_species WHERE species_name LIKE %s"
        print(f"🔍 Executing SQL: {sql} with value: %{query}%")  # Debugging print
        cursor.execute(sql, (f"%{query}%",))
        results = cursor.fetchall()
        cursor.close()

        if not results:
            return jsonify({"error": "No results found"}), 404

        # Get random image for the species and URL-encode the path
        image_path = get_random_image(query)
        if image_path:
            encoded_path = urllib.parse.quote(image_path)
            image_url = f"/get_image?path={encoded_path}"
        else:
            image_url = None

        # Add image URL to the first result
        results[0]["image_url"] = image_url

        return jsonify(results[0])  # Send first record only

    except mysql.connector.Error as err:
        print(f"❌ Database Query Error: {err}")  # Print error in logs
        return jsonify({"error": f"Database query failed: {err}"}), 500

# Endpoint to serve images
@app.route("/get_image")
def get_image():
    from urllib.parse import unquote
    path = unquote(request.args.get("path", ""))
    if os.path.exists(path):
        return send_from_directory(os.path.dirname(path), os.path.basename(path))
    return "Image not found", 404

if __name__ == "__main__":
    app.run(debug=True)
