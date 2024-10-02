from flask import Flask, request, jsonify
import pandas as pd
import requests
import os
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
import time

app = Flask(__name__)

# Replace this with your actual Telegram Bot Token
BOT_TOKEN = 'YOUR_BOT_TOKEN'

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        data = request.get_json()

        if data is None:
            return jsonify({"error": "No JSON data provided"}), 400

        user_id = data['message']['from']['id']
        text = data['message'].get('text', '')

        if text.lower() == '/start':
            response_text = "Welcome! Please upload your transactional data in CSV format."
            send_telegram_message(user_id, response_text)
            return jsonify({"status": "success"}), 200

        elif 'document' in data['message']:
            file_id = data['message']['document']['file_id']
            file_response = requests.get(f'https://api.telegram.org/bot{BOT_TOKEN}/getFile?file_id={file_id}')
            file_data = file_response.json()
            file_path = file_data['result']['file_path']
            file_url = f'https://api.telegram.org/file/bot{BOT_TOKEN}/{file_path}'

            try:
                df = pd.read_csv(file_url)
                if df.empty:
                    raise ValueError("The uploaded CSV is empty.")

            except Exception as e:
                send_telegram_message(user_id, f"There was an error reading the CSV file: {str(e)}")
                return jsonify({"error": "Error loading CSV file"}), 500

            # Process the data and generate CSV files
            insights, csv_paths = process_data(df, user_id)

            # Send insights and CSVs to the user
            send_telegram_message(user_id, insights)
            for csv_path in csv_paths:
                send_telegram_file(user_id, csv_path)

            return jsonify({"status": "success"}), 200

        else:
            response_text = "Please upload a CSV file with your transactional data."
            send_telegram_message(user_id, response_text)
            return jsonify({"status": "success"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def process_data(df, user_id):
    try:
        # Determine if the CSV is in one-hot encoded format or needs preprocessing
        df_cleaned = preprocess_transaction_data(df)

        # Apriori algorithm
        start_time_apriori = time.time()
        frequent_itemsets_apriori = apriori(df_cleaned, min_support=0.05, use_colnames=True)
        rules_apriori = association_rules(frequent_itemsets_apriori, metric="confidence", min_threshold=0.7)
        time_apriori = time.time() - start_time_apriori

        # FPGrowth algorithm
        start_time_fpgrowth = time.time()
        frequent_itemsets_fpgrowth = fpgrowth(df_cleaned, min_support=0.05, use_colnames=True)
        rules_fpgrowth = association_rules(frequent_itemsets_fpgrowth, metric="confidence", min_threshold=0.7)
        time_fpgrowth = time.time() - start_time_fpgrowth

        # Save results to CSV files
        csv_paths = []
        csv_paths.append(save_table_to_csv(frequent_itemsets_apriori, f"frequent_itemsets_apriori_{user_id}.csv"))
        csv_paths.append(save_table_to_csv(rules_apriori, f"rules_apriori_{user_id}.csv"))
        csv_paths.append(save_table_to_csv(frequent_itemsets_fpgrowth, f"frequent_itemsets_fpgrowth_{user_id}.csv"))
        csv_paths.append(save_table_to_csv(rules_fpgrowth, f"rules_fpgrowth_{user_id}.csv"))

        # Generate customer insights
        insights = generate_customer_insights(rules_apriori)

        return insights, csv_paths

    except Exception as e:
        return f"Error processing data: {str(e)}", []

def preprocess_transaction_data(df):
    """Preprocesses the CSV to a format usable by apriori/fpgrowth algorithms.
       If it's not already one-hot encoded, it will be transformed.
    """
    if not is_one_hot_encoded(df):
        # Assuming the transactions are in a single column, need to pivot the table
        df = df.stack().groupby(level=0).apply(lambda x: pd.Series(1, index=x)).unstack(fill_value=0)
    
    return df

def is_one_hot_encoded(df):
    """Check if the DataFrame is one-hot encoded (binary transaction matrix)."""
    return df.applymap(lambda x: x in [0, 1]).all().all()

def generate_customer_insights(rules):
    insights = "### Customer Insights:\n\n"
    for _, rule in rules.iterrows():
        antecedents = ', '.join(list(rule['antecedents']))
        consequents = ', '.join(list(rule['consequents']))
        support = rule['support']
        confidence = rule['confidence']
        
        insights += f"If customers buy **{antecedents}**, they are likely to also buy **{consequents}** (Confidence: {confidence:.2f}, Support: {support:.2f}).\n"

    if insights == "### Customer Insights:\n\n":
        insights += "No actionable insights available."

    return insights

def save_table_to_csv(table, filename):
    csv_path = os.path.join(os.getcwd(), filename)
    table.to_csv(csv_path, index=False)
    return csv_path

def send_telegram_message(chat_id, text):
    requests.post(f'https://api.telegram.org/bot{BOT_TOKEN}/sendMessage', data={'chat_id': chat_id, 'text': text, 'parse_mode': 'Markdown'})

def send_telegram_file(chat_id, file_path):
    with open(file_path, 'rb') as f:
        requests.post(f'https://api.telegram.org/bot{BOT_TOKEN}/sendDocument', data={'chat_id': chat_id}, files={'document': f})

if __name__ == '__main__':
    app.run(port=5004)
