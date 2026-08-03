from flask import Flask, request, jsonify, render_template
import joblib
import sqlite3
from datetime import datetime

app = Flask(__name__)

model = joblib.load("model.pkl")

DATABASE = "database.db"


def init_database():

    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sensor_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            voltage REAL,
            current REAL,
            temperature REAL,
            humidity REAL,
            power REAL,
            prediction INTEGER,
            status TEXT,
            confidence REAL
        )
    """)

    connection.commit()
    connection.close()

    print("Database initialized.")


@app.route("/")
def dashboard():

    return render_template("dashboard.html")


@app.route("/api/sensor", methods=["POST"])
def sensor():

    try:

        data = request.get_json()

        voltage = float(data["voltage"])
        current = float(data["current"])
        temperature = float(data["temperature"])
        humidity = float(data["humidity"])
        power = float(data["power"])

        features = [[
            voltage,
            current,
            temperature,
            humidity,
            power
        ]]

        prediction = int(model.predict(features)[0])

        probability = model.predict_proba(features)[0]

        confidence = float(max(probability))

        if prediction == 0:
            status = "NORMAL"
        else:
            status = "FAULT"

        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        connection = sqlite3.connect(DATABASE)
        cursor = connection.cursor()

        cursor.execute("""
            INSERT INTO sensor_data (
                timestamp,
                voltage,
                current,
                temperature,
                humidity,
                power,
                prediction,
                status,
                confidence
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp,
            voltage,
            current,
            temperature,
            humidity,
            power,
            prediction,
            status,
            confidence
        ))

        connection.commit()
        connection.close()

        print()
        print("========== SENSOR DATA ==========")
        print("Voltage:", voltage, "V")
        print("Current:", current, "A")
        print("Temperature:", temperature, "C")
        print("Humidity:", humidity, "%")
        print("Power:", power, "W")
        print("Status:", status)
        print("Confidence:", round(confidence * 100, 2), "%")
        print("=================================")

        return jsonify({
            "status": status,
            "prediction": prediction,
            "confidence": confidence,
            "voltage": voltage,
            "current": current,
            "temperature": temperature,
            "humidity": humidity,
            "power": power,
            "timestamp": timestamp
        })

    except Exception as error:

        print("ERROR:", error)

        return jsonify({
            "error": str(error)
        }), 500


@app.route("/api/latest")
def latest():

    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row

    cursor = connection.cursor()

    cursor.execute("""
        SELECT *
        FROM sensor_data
        ORDER BY id DESC
        LIMIT 1
    """)

    row = cursor.fetchone()

    connection.close()

    if row is None:

        return jsonify({
            "message": "No sensor data available"
        })

    return jsonify(dict(row))


@app.route("/api/history")
def history():

    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row

    cursor = connection.cursor()

    cursor.execute("""
        SELECT *
        FROM sensor_data
        ORDER BY id DESC
        LIMIT 30
    """)

    rows = cursor.fetchall()

    connection.close()

    return jsonify([
        dict(row)
        for row in rows
    ])


if __name__ == "__main__":

    init_database()

    import os

    port = int(os.environ.get("PORT", 5000))

    print()
    print("==========================================")
    print(" SMART GRID FAULT DETECTION SYSTEM")
    print("==========================================")
    print()
    print("Server starting...")
    print("Port:", port)
    print()
    print("Waiting for sensor data...")
    print()

    app.run(
        host="0.0.0.0",
        port=port
    )

