from flask import Flask, render_template, request, redirect, url_for, jsonify
from deepface import DeepFace
import cv2
import numpy as np
import pandas as pd
import os
import json
import requests

import os
print(os.getcwd())


app = Flask(__name__)

# Cancelable biometric key (length = Facenet embedding size)
KEY = np.random.rand(128)  # keep this consistent; save/load if needed

FACES_CSV = "faces.csv"
FARES_CSV = "fares.csv"
MAPBOX_TOKEN = "ADD_YOUR_TOKEN"

# Ensure CSV exists
if not os.path.exists(FACES_CSV):
    pd.DataFrame(columns=["name", "embedding"]).to_csv(FACES_CSV, index=False)
if not os.path.exists(FARES_CSV):
    pd.DataFrame(columns=["name", "balance", "rides"]).to_csv(FARES_CSV, index=False)

# Utility: convert np.array <-> json string
def encode_to_str(encoding):
    return json.dumps(encoding)

def decode_to_np(s):
    return np.array(json.loads(s))


# ---------- ROUTES ----------

@app.route("/")
def home():
    return "<h2>Smart Fare System</h2><a href='/enroll'>Enroll</a> | <a href='/recognise'>Recognise</a>"




# Apply cancelable transformation
def cancelable_transform(embedding):
    emb = np.array(embedding)
    transformed = np.sign(emb * KEY)  # +1/-1 vector
    return transformed

# Compare transformed embeddings using Hamming similarity
def compare_cancelable(known, candidate):
    known = np.array(known)
    candidate = np.array(candidate)
    return np.sum(known == candidate) / len(known)  # 0.0 to 1.0





# ----- ENROLL -----
@app.route("/enroll", methods=["GET", "POST"])
def enroll():
    if request.method == "POST":
        name = request.form["name"]

        # Capture webcam frame
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return "Camera failed!"

        # Get embedding with DeepFace
        try:
            emb = DeepFace.represent(frame, model_name="Facenet")[0]['embedding']
        except Exception as e:
            return f"No face detected! {e}"

        # Save to CSV
        # Transform embedding for cancelable biometric
        trans_emb = cancelable_transform(emb)

        # Save transformed embedding instead of raw
        df = pd.read_csv(FACES_CSV)
        df = pd.concat([df, pd.DataFrame([[name, encode_to_str(trans_emb.tolist())]], columns=df.columns)])
        df.to_csv(FACES_CSV, index=False)


        # Also add to fares if new
        fares = pd.read_csv(FARES_CSV)
        if name not in fares["name"].values:
            fares = pd.concat([fares, pd.DataFrame([[name, 50000, 0]], columns=fares.columns)])
            fares.to_csv(FARES_CSV, index=False)

        return f"Face enrolled for {name}!"

    return render_template("enroll.html")


# ----- RECOGNISE -----
@app.route("/recognise", methods=["GET", "POST"])
def recognise():
    if request.method == "POST":
        # Load known faces
        df = pd.read_csv(FACES_CSV)
        if df.empty:
            return "No faces enrolled!"

        names = df["name"].tolist()
        encodings = [decode_to_np(e) for e in df["embedding"].tolist()]

        # Capture frame
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return "Camera failed!"

        # Current embedding
        try:
            current = DeepFace.represent(frame, model_name="Facenet")[0]['embedding']
        except Exception as e:
            return f"No face detected! {e}"

        # Transform current embedding
        current_trans = cancelable_transform(current)

        # Compare using cancelable embeddings
        best_match, best_score = None, 0  # similarity
        for name, emb in zip(names, encodings):
            score = compare_cancelable(emb, current_trans)
            if score > best_score:
                best_match, best_score = name, score

        threshold = 0.8  # 80% similarity
        if best_score > threshold:
            # Update rides etc.

            # Update rides
            fares = pd.read_csv(FARES_CSV)
            fares.loc[fares["name"] == best_match, "rides"] += 1
            fares.to_csv(FARES_CSV, index=False)

            return redirect(url_for("fare", user=best_match))
        else:
            return "Face not recognised!"

    return render_template("recognise.html")


# ----- FARE -----
@app.route("/fare", methods=["GET", "POST"])
def fare():
    user = request.args.get("user", "")
    if request.method == "POST":
        start = request.form["start"]
        end = request.form["end"]

        # Call Mapbox
        url = f"https://api.mapbox.com/directions/v5/mapbox/driving/{start};{end}?geometries=geojson&access_token={MAPBOX_TOKEN}"
        r = requests.get(url).json()

        if "routes" not in r or len(r["routes"]) == 0:
            return "No route found!"

        dist_km = r["routes"][0]["distance"] / 1000
        fare_amount = dist_km * 10  # Rs 10 per km

        fares = pd.read_csv(FARES_CSV)
        balance = fares.loc[fares["name"] == user, "balance"].values[0]
        if balance < fare_amount:
            return f"Insufficient balance! Current balance: ₹{balance:.2f}"
        
        fares.loc[fares["name"] == user, "balance"] -= fare_amount
        fares.loc[fares["name"] == user, "rides"] += 1
        fares.to_csv(FARES_CSV, index=False)

        return f"User {user}: Distance {dist_km:.2f} km → Fare ₹{fare_amount:.2f} deducted! New balance: ₹{balance - fare_amount:.2f}"


    return render_template("fare.html", user=user)


# ----- ANALYTICS -----
@app.route("/analytics")
def analytics():
    fares = pd.read_csv(FARES_CSV)
    total_rides = fares["rides"].sum()
    return f"Total passenger entries recorded: {total_rides}"


if __name__ == "__main__":
    app.run(debug=True)
