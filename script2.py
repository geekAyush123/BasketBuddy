from flask import Flask, request, jsonify
import pandas as pd
import requests
import matplotlib.pyplot as plt
import plotly.express as px
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
import time
import os

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

            # Proceed with analysis and generate graphs
            results, graph_path, report_path = process_data(df, user_id)
            print(f"Analysis results: {results}")  # Debugging line

            # Send results back to the user
            send_telegram_message(user_id, results)

            # Send the generated graph to the user
            if graph_path:
                send_telegram_file(user_id, graph_path)

            # Send the report to the user
            if report_path:
                send_telegram_file(user_id, report_path)

            return jsonify({"status": "success"}), 200

        else:
            print("No document detected, asking for CSV")  # Debugging line
            response_text = "Please upload a CSV file with your transactional data."
            send_telegram_message(user_id, response_text)
            return jsonify({"status": "success"}), 200

    except Exception as e:
        print(f"Error: {str(e)}")  # Log the exception for debugging
        return jsonify({"error": str(e)}), 500

def process_data(df, user_id):
    try:
        # Perform data cleaning and processing
        print("Running Apriori algorithm...")  # Debugging line
        
        # Ensure that the DataFrame is in the correct format for apriori
        start_time_apriori = time.time()
        frequent_itemsets_apriori = apriori(df, min_support=0.05, use_colnames=True)
        rules_apriori = association_rules(frequent_itemsets_apriori, metric="confidence", min_threshold=0.7)
        time_apriori = time.time() - start_time_apriori

        print("Running FPGrowth algorithm...")  # Debugging line
        start_time_fpgrowth = time.time()
        frequent_itemsets_fpgrowth = fpgrowth(df, min_support=0.05, use_colnames=True)
        rules_fpgrowth = association_rules(frequent_itemsets_fpgrowth, metric="confidence", min_threshold=0.7)
        time_fpgrowth = time.time() - start_time_fpgrowth

        # Generate a concise response
        results = "### Analysis Results\n\n"
        results += f"**Apriori Algorithm** (Time: {time_apriori:.4f} seconds):\n"
        results += "Top 5 Frequent Itemsets:\n"
        results += frequent_itemsets_apriori.head(5).to_string(index=False) + "\n\n"
        results += "Top 5 Association Rules:\n"
        results += rules_apriori[['antecedents', 'consequents', 'support', 'confidence']].head(5).to_string(index=False) + "\n\n"

        results += f"**FPGrowth Algorithm** (Time: {time_fpgrowth:.4f} seconds):\n"
        results += "Top 5 Frequent Itemsets:\n"
        results += frequent_itemsets_fpgrowth.head(5).to_string(index=False) + "\n\n"
        results += "Top 5 Association Rules:\n"
        results += rules_fpgrowth[['antecedents', 'consequents', 'support', 'confidence']].head(5).to_string(index=False)

        # Generate actionable insights for shopkeepers
        actionable_insights = generate_actionable_insights(rules_apriori)
        results += "\n### Actionable Insights for Your Store:\n"
        results += actionable_insights

        # Generate graph of the frequent itemsets for both algorithms
        graph_path = generate_graph(frequent_itemsets_apriori, frequent_itemsets_fpgrowth)

        # Generate detailed report
        report_path = generate_report(results)

        return results[:4000], graph_path, report_path  # Telegram messages are limited to 4096 characters

    except Exception as e:
        print(f"Error processing data: {str(e)}")  # Log the exception for debugging
        return f"Error processing data: {str(e)}", None, None

def generate_actionable_insights(rules):
    insights = ""
    for _, rule in rules.iterrows():
        antecedents = ', '.join(list(rule['antecedents']))
        consequents = ', '.join(list(rule['consequents']))
        support = rule['support']
        confidence = rule['confidence']
        
        # Formulate actionable insights
        insights += f"If customers buy **{antecedents}**, they are likely to also buy **{consequents}** (Confidence: {confidence:.2f}, Support: {support:.2f}). Consider placing these items together!\n"

    return insights or "No actionable insights available."


def generate_graph(frequent_itemsets_apriori, frequent_itemsets_fpgrowth):
    try:
        plt.figure(figsize=(14, 6))

        # Apriori graph
        plt.subplot(1, 2, 1)
        plt.bar(frequent_itemsets_apriori['itemsets'].astype(str), frequent_itemsets_apriori['support'])
        plt.title('Apriori Frequent Itemsets Support')
        plt.xlabel('Itemsets')
        plt.ylabel('Support')
        plt.xticks(rotation=45)

        # FPGrowth graph
        plt.subplot(1, 2, 2)
        plt.bar(frequent_itemsets_fpgrowth['itemsets'].astype(str), frequent_itemsets_fpgrowth['support'])
        plt.title('FPGrowth Frequent Itemsets Support')
        plt.xlabel('Itemsets')
        plt.ylabel('Support')
        plt.xticks(rotation=45)

        graph_path = 'frequent_itemsets_graph.png'
        plt.tight_layout()
        plt.savefig(graph_path)
        plt.close()
        print(f"Graph saved at: {graph_path}")  # Debugging line
        return graph_path
    except Exception as e:
        print(f"Error generating graph: {str(e)}")
        return None

def generate_report(results):
    try:
        report_path = 'analysis_report.txt'
        with open(report_path, 'w') as f:
            f.write(results)
        print(f"Report saved at: {report_path}")  # Debugging line
        return report_path
    except Exception as e:
        print(f"Error generating report: {str(e)}")
        return None

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

def send_telegram_file(chat_id, file_path):
    print(f"Sending file to chat ID {chat_id}: {file_path}")  # Debugging line
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendDocument"
    with open(file_path, 'rb') as file:
        response = requests.post(url, data={"chat_id": chat_id}, files={"document": file})
    print(f"File send response: {response.status_code}, {response.text}")  # Debugging line

if __name__ == '__main__':
    app.run(port=5004)
