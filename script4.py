from flask import Flask, request, jsonify
import pandas as pd
import requests
import os
import subprocess
from mlxtend.frequent_patterns import apriori, association_rules

app = Flask(__name__)

# Replace this with your actual Telegram Bot Token
BOT_TOKEN = '7769785094:AAFOY5_AxNHxBYWoG6EQMPg_8MMSNvcO9TY'  # Update this to your bot token

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        data = request.get_json()

        if data is None:
            return jsonify({"error": "Invalid request"}), 400

        user_id = data['message']['from']['id']
        text = data['message'].get('text', '').lower()

        if text == '/start':
            send_telegram_message(user_id, "Welcome! Please upload your CSV data.")
            return jsonify({"status": "success"}), 200

        elif 'document' in data['message']:
            file_id = data['message']['document']['file_id']
            file_url = get_file_url(file_id)

            try:
                df = pd.read_csv(file_url)
                if df.empty:
                    raise ValueError("Empty CSV file.")

            except Exception as e:  # Catch specific exceptions for better error reporting
                send_telegram_message(user_id, f"Error reading the CSV file: {str(e)}")
                return jsonify({"error": "Error loading CSV"}), 500

            insights, csv_paths = process_data(df, user_id)
            send_telegram_message(user_id, insights)

            # Call Gemma to generate suggestions
            suggestions = run_gemma_model(insights)
            send_telegram_message(user_id, suggestions)

            for csv_path in csv_paths:
                send_telegram_file(user_id, csv_path)

            return jsonify({"status": "success"}), 200

        else:
            send_telegram_message(user_id, "Please upload a CSV file.")
            return jsonify({"status": "success"}), 200

    except Exception as e:  # Provide more specific error feedback
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

def get_file_url(file_id):
    file_response = requests.get(f'https://api.telegram.org/bot{BOT_TOKEN}/getFile?file_id={file_id}')
    file_data = file_response.json()
    if not file_data.get('ok'):
        raise ValueError("Error fetching file from Telegram.")
    file_path = file_data['result']['file_path']
    return f'https://api.telegram.org/file/bot{BOT_TOKEN}/{file_path}'

def process_data(df, user_id):
    try:
        df_cleaned = preprocess_transaction_data(df)
        frequent_itemsets = apriori(df_cleaned, min_support=0.05, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

        csv_paths = [
            save_table_to_csv(frequent_itemsets, f"frequent_itemsets_{user_id}.csv"),
            save_table_to_csv(rules, f"rules_{user_id}.csv"),
            save_insights_to_csv(rules, f"insights_{user_id}.csv")  # Added insights CSV
        ]

        return "Customer insights generated and saved successfully.", csv_paths

    except Exception as e:  # Catch specific exceptions for better error reporting
        return f"Data processing failed: {str(e)}", []

def save_insights_to_csv(rules, filename):
    insights_data = []

    for _, rule in rules.iterrows():
        antecedents = ', '.join(list(rule['antecedents']))
        consequents = ', '.join(list(rule['consequents']))
        support = rule['support']
        confidence = rule['confidence']
        
        insights_data.append({
            "Antecedents": antecedents,
            "Consequents": consequents,
            "Confidence": f"{confidence:.2f}",
            "Support": f"{support:.2f}",
            "Insight": f"If customers buy {antecedents}, they are likely to also buy {consequents} (Confidence: {confidence:.2f}, Support: {support:.2f})."
        })

    insights_df = pd.DataFrame(insights_data)
    insights_df.to_csv(filename, index=False)
    return filename

def preprocess_transaction_data(df):
    if not is_one_hot_encoded(df):
        df = df.stack().groupby(level=0).apply(lambda x: pd.Series(1, index=x)).unstack(fill_value=0)
    return df

def is_one_hot_encoded(df):
    return df.applymap(lambda x: x in [0, 1]).all().all()

def save_table_to_csv(table, filename):
    csv_path = os.path.join(os.getcwd(), filename)
    table.to_csv(csv_path, index=False)
    return csv_path

def run_gemma_model(insights):
    # Call the Gemma model (assuming it's available as a subprocess)
    try:
        result = subprocess.run(['ollama', 'run', 'gemma2:2b', insights], capture_output=True, text=True)
        if result.returncode != 0:
            return "Error running Gemma model."
        
        gemma_output = result.stdout.strip()
        
        # Generate a market tip based on insights
        market_tip = generate_market_tip(insights)
        
        return f"{gemma_output}\n\n**Market Tip:** {market_tip}"
    
    except Exception as e:
        return f"Error: {str(e)}"

def generate_market_tip(insights):
    # Logic to create a market tip based on the insights
    # This is a simple example; you may want to use more complex logic or data
    if "high value" in insights.lower():
        return "Consider targeting high-value customers with personalized offers."
    elif "frequent" in insights.lower():
        return "Leverage frequent purchases to create loyalty programs."
    else:
        return "Stay updated with market trends to enhance your strategies."


def send_telegram_message(chat_id, text):
    requests.post(f'https://api.telegram.org/bot{BOT_TOKEN}/sendMessage', data={'chat_id': chat_id, 'text': text, 'parse_mode': 'Markdown'})

def send_telegram_file(chat_id, file_path):
    with open(file_path, 'rb') as f:
        requests.post(f'https://api.telegram.org/bot{BOT_TOKEN}/sendDocument', data={'chat_id': chat_id}, files={'document': f})

if __name__ == '__main__':
    app.run(port=5004)
