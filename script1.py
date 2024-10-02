from flask import Flask, request, jsonify
import pandas as pd
import requests
from mlxtend.frequent_patterns import apriori, association_rules

app = Flask(__name__)

# Replace this with your actual Telegram Bot Token
BOT_TOKEN = '7769785094:AAFOY5_AxNHxBYWoG6EQMPg_8MMSNvcO9TY'

@app.route('/webhook', methods=['POST'])
def webhook():
    print("Webhook endpoint hit!")  # Debugging line to confirm request hit
    try:
        # Retrieve JSON data from the request
        data = request.get_json()
        print(f"Received data: {data}")  # Log the incoming data for debugging

        # Check if data is provided
        if data is None:
            print("No JSON data received")  # Debugging line
            return jsonify({"error": "No JSON data provided"}), 400

        # Extracting parameters from the JSON data
        user_id = data['message']['from']['id']
        text = data['message'].get('text', '')

        print(f"User ID: {user_id}, Message: {text}")  # Log user ID and text

        if text.lower() == '/start':
            response_text = "Welcome! Please upload your transactional data in CSV format."
            print("Sending welcome message")  # Debugging line
            send_telegram_message(user_id, response_text)
            return jsonify({"status": "success"}), 200

        elif 'document' in data['message']:
            file_id = data['message']['document']['file_id']
            print(f"File ID received: {file_id}")  # Debugging line

            # Fetching file info from Telegram
            file_response = requests.get(f'https://api.telegram.org/bot{BOT_TOKEN}/getFile?file_id={file_id}')
            file_data = file_response.json()
            print(f"File data: {file_data}")  # Debugging line

            file_path = file_data['result']['file_path']
            file_url = f'https://api.telegram.org/file/bot{BOT_TOKEN}/{file_path}'
            print(f"File URL: {file_url}")  # Debugging line

            # Read the CSV data
            try:
                df = pd.read_csv(file_url)
                print("CSV data successfully loaded")  # Debugging line
                print(df.head())  # Log a preview of the DataFrame for verification
            except Exception as e:
                print(f"Error loading CSV: {str(e)}")  # Debugging line
                send_telegram_message(user_id, "There was an error reading the CSV file.")
                return jsonify({"error": "Error loading CSV file"}), 500

            # Proceed with analysis
            results = process_data(df)
            print(f"Analysis results: {results}")  # Debugging line

            # Send results back to the user
            send_telegram_message(user_id, results)
            return jsonify({"status": "success"}), 200

        else:
            print("No document detected, asking for CSV")  # Debugging line
            response_text = "Please upload a CSV file with your transactional data."
            send_telegram_message(user_id, response_text)
            return jsonify({"status": "success"}), 200

    except Exception as e:
        print(f"Error: {str(e)}")  # Log the exception for debugging
        return jsonify({"error": str(e)}), 500

def process_data(df):
    try:
        # Perform data cleaning and processing
        print("Running Apriori algorithm...")  # Debugging line
        
        frequent_itemsets = apriori(df, min_support=0.05, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

        # Generate a concise response
        results = "Top 5 Frequent Itemsets:\n"
        results += frequent_itemsets.head(5).to_string(index=False)  # Return only top 5 itemsets

        results += "\n\nTop 5 Association Rules:\n"
        rules_summary = rules[['antecedents', 'consequents', 'support', 'confidence']].head(5)  # Only top 5 rules
        results += rules_summary.to_string(index=False)

        return results[:4000]  # Telegram messages are limited to 4096 characters, ensure it's within limit

    except Exception as e:
        print(f"Error processing data: {str(e)}")  # Log the exception for debugging
        return f"Error processing data: {str(e)}"


def send_telegram_message(chat_id, text):
    print(f"Sending message to chat ID {chat_id}: {text}")  # Debugging line
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown"
    }
    response = requests.post(url, json=payload)
    print(f"Message send response: {response.status_code}, {response.text}")  # Debugging line

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5004, debug=True)