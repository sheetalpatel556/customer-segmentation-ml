# Customer Segmentation Using Machine Learning

A machine learning project that segments customers based on their purchasing behaviour using **RFM Analysis and K-Means Clustering**.

## Live Demo
[Try the Customer Segmentation App](https://customer-segmentation-ml-kfu7.onrender.com)

## Project Overview
This project analyses customer transactions and groups customers into meaningful segments. Businesses can use these insights to improve customer retention, marketing strategies, and loyalty programs.

## RFM Analysis
Three metrics are used to understand customer behaviour:

- **Recency:** Number of days since the customer's last purchase.
- **Frequency:** Number of unique orders placed by the customer.
- **Monetary:** Total amount spent by the customer.

## Machine Learning Workflow
1. Load and clean the Online Retail dataset.
2. Handle missing values, duplicate records, cancelled orders, and invalid transactions.
3. Calculate customer-level RFM metrics.
4. Apply log transformation and feature scaling.
5. Train a K-Means clustering model with two clusters.
6. Save the trained model and scaler.
7. Deploy the prediction app using Flask and Render.

## Customer Segments
- **Loyal / High-Value:** Customers showing stronger engagement and higher spending.
- **At-Risk / Low-Engagement:** Customers showing lower engagement and spending.

These labels are based on the characteristics of the resulting clusters.

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- Flask
- Joblib
- GitHub
- Render

## Dataset
[UCI Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online+retail)

## Run Locally
Install the required libraries:

```bash
pip install -r requirements.txt
```

Start the Flask application:

```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

## Project Files
- `app.py` — Flask web application
- `final_model.pkl` — Trained K-Means model
- `rfm_scaler.pkl` — Feature scaler
- `requirements.txt` — Python dependencies

## Future Improvements
- Interactive customer segment visualisations
- More detailed customer segments
- Automated marketing recommendations

---
Developed as a Machine Learning project.
