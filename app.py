import os
import numpy as np
import joblib
from flask import Flask, request, render_template_string

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "final_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "rfm_scaler.pkl"))

HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Customer Segmentation</title>
</head>
<body style="font-family:Arial;background:#f1f5f9;padding:30px;">
    <div style="max-width:650px;margin:30px auto;background:white;
                padding:30px;border-radius:16px;">
        <h1>Customer Segmentation</h1>
        <p>Understand your customers using Machine Learning.</p>

        <form method="POST">
            <label>Recency (days)</label><br>
            <input type="number" name="recency" min="0" required
                   style="box-sizing:border-box;width:100%;padding:12px;margin:10px 0 20px;">

            <label>Frequency (number of orders)</label><br>
            <input type="number" name="frequency" min="1" required
                   style="box-sizing:border-box;width:100%;padding:12px;margin:10px 0 20px;">

            <label>Monetary (total spending)</label><br>
            <input type="number" name="monetary" min="0.01" step="any" required
                   style="box-sizing:border-box;width:100%;padding:12px;margin:10px 0 20px;">

            <button type="submit"
                    style="width:100%;padding:14px;background:#2563eb;
                           color:white;border:0;border-radius:8px;">
                Predict Customer Segment
            </button>
        </form>

        {% if segment %}
            <hr>
            <h2 style="color:{{ color }};">{{ segment }}</h2>
            <p>{{ description }}</p>
            <h3>Recommended Action</h3>
            <p>{{ recommendation }}</p>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    segment = description = recommendation = color = None

    if request.method == "POST":
        try:
            recency = float(request.form["recency"])
            frequency = int(request.form["frequency"])
            monetary = float(request.form["monetary"])

            if recency < 0 or frequency < 1 or monetary <= 0:
                raise ValueError("Please enter valid positive values.")

            customer = np.array([[recency, frequency, monetary]])
            customer_scaled = scaler.transform(np.log1p(customer))
            cluster = int(model.predict(customer_scaled)[0])

            if cluster == 0:
                segment = "Loyal / High-Value"
                description = "This customer belongs to cluster 0."
                recommendation = "Consider loyalty rewards and relevant product recommendations."
                color = "#15803d"
            else:
                segment = "At-Risk / Low-Engagement"
                description = "This customer belongs to cluster 1."
                recommendation = "Consider a targeted re-engagement campaign."
                color = "#b45309"

        except (ValueError, TypeError):
            segment = "Invalid input"
            description = "Please enter valid numeric RFM values."
            color = "#b91c1c"

    return render_template_string(
        HTML,
        segment=segment,
        description=description,
        recommendation=recommendation,
        color=color
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
