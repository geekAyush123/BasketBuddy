from flask import Flask, request, jsonify
import pandas as pd
import requests
import os
import subprocess
import time
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

app = Flask(__name__)

# Replace this with your actual Telegram Bot Token
BOT_TOKEN = 'YOUR_BOT_TOKEN'  # Update this to your bot token

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

        elif text == '/preprocess':
            help_message = (
                "To preprocess your data for analysis, please ensure your CSV is formatted correctly.\n\n"
                "### Expected Format:\n"
                "Each row should represent a transaction, and each column should represent an item.\n"
                "\n### Example:\n"
                "```\n"
                "Bread,Butter,Milk,Eggs,Cheese\n"
                "1,1,0,1,0\n"
                "1,0,1,1,0\n"
                "0,1,1,0,1\n"
                "1,1,1,0,1\n"
                "0,0,1,1,0\n"
                "```\n"
                "### Preprocessing Notes:\n"
                "- The data should be in a one-hot encoded format (1 for presence, 0 for absence).\n"
                "- Ensure that the first row contains the item names as headers.\n"
                "- There should be no missing values in the CSV.\n"
                "- You can send the CSV after following these guidelines."
            )
            send_telegram_message(user_id, help_message)
            return jsonify({"status": "success"}), 200

        elif 'document' in data['message']:
            file_id = data['message']['document']['file_id']
            file_url = get_file_url(file_id)

            try:
                # Initialize an empty DataFrame for chunk processing
                df_cleaned = pd.DataFrame()

                # Read CSV in chunks
                for chunk in pd.read_csv(file_url, chunksize=1000):  # Adjust chunksize as needed
                    if chunk.empty:
                        raise ValueError("Empty chunk in CSV file.")
                    df_cleaned = pd.concat([df_cleaned, preprocess_transaction_data(chunk)])

                if df_cleaned.empty:
                    raise ValueError("Empty CSV file after processing.")

            except Exception as e:
                send_telegram_message(user_id, f"Error reading the CSV file: {str(e)}")
                return jsonify({"error": "Error loading CSV"}), 500

            insights, csv_paths = process_data(df_cleaned, user_id)
            send_telegram_message(user_id, insights)

            # Call Gemma to generate suggestions
            suggestions = run_gemma_model(insights)
            send_telegram_message(user_id, suggestions)

            for csv_path in csv_paths:
                send_telegram_file(user_id, csv_path)

            return jsonify({"status": "success"}), 200

        else:
            send_telegram_message(user_id, "Please upload a CSV file or type /help for assistance.")
            return jsonify({"status": "success"}), 200

    except Exception as e:
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
        # Timing Apriori
        start_time = time.time()
        frequent_itemsets_apriori = apriori(df, min_support=0.05, use_colnames=True)
        apriori_time = time.time() - start_time
        
        rules_apriori = association_rules(frequent_itemsets_apriori, metric="confidence", min_threshold=0.7)

        # Timing FP-Growth
        start_time = time.time()
        frequent_itemsets_fp = fpgrowth(df, min_support=0.05, use_colnames=True)
        fp_growth_time = time.time() - start_time
        
        rules_fp = association_rules(frequent_itemsets_fp, metric="confidence", min_threshold=0.7)

        # Prepare insights for Telegram
        insights = generate_insights(rules_apriori, rules_fp, apriori_time, fp_growth_time)
        
        csv_paths = [
            save_table_to_csv(frequent_itemsets_apriori, f"frequent_itemsets_apriori_{user_id}.csv"),
            save_table_to_csv(rules_apriori, f"rules_apriori_{user_id}.csv"),
            save_table_to_csv(frequent_itemsets_fp, f"frequent_itemsets_fp_{user_id}.csv"),
            save_table_to_csv(rules_fp, f"rules_fp_{user_id}.csv"),
            save_insights_to_csv(rules_apriori, f"insights_{user_id}.csv")  # Added insights CSV
        ]

        return insights, csv_paths

    except Exception as e:
        return f"Data processing failed: {str(e)}", []

def generate_insights(rules_apriori, rules_fp, apriori_time, fp_growth_time):
    insights = []

    insights.append(f"Apriori Algorithm:\n- Execution Time: {apriori_time:.4f} seconds\n- Rules Generated: {len(rules_apriori)}")
    insights.append(f"FP-Growth Algorithm:\n- Execution Time: {fp_growth_time:.4f} seconds\n- Rules Generated: {len(rules_fp)}")

    if apriori_time < fp_growth_time:
        insights.append("Suggestion: Use the Apriori algorithm for better performance.")
    else:
        insights.append("Suggestion: Use the FP-Growth algorithm for better performance.")

    return "\n".join(insights)

def save_table_to_csv(table, filename):
    csv_path = os.path.join(os.getcwd(), filename)
    table.to_csv(csv_path, index=False)
    return csv_path

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

def run_gemma_model(insights):
    try:
        with open('gemma_output.txt', 'w') as output_file:
            result = subprocess.run(['ollama', 'run', 'gemma2:2b', insights], stdout=output_file, stderr=subprocess.STDOUT, text=True)
        
        if result.returncode != 0:
            return "Error running Gemma model. Check 'gemma_output.txt' for details."
        
        with open('gemma_output.txt', 'r') as output_file:
            gemma_output = output_file.read().strip()
        
        return gemma_output
    
    except Exception as e:
        return f"Error: {str(e)}"

def send_telegram_message(chat_id, text):
    requests.post(f'https://api.telegram.org/bot{BOT_TOKEN}/sendMessage', data={'chat_id': chat_id, 'text': text, 'parse_mode': 'Markdown'})

def send_telegram_file(chat_id, file_path):
    with open(file_path, 'rb') as f:
        requests.post(f'https://api.telegram.org/bot{BOT_TOKEN}/sendDocument', data={'chat_id': chat_id}, files={'document': f})

if __name__ == '__main__':
    app.run(port=5004)
