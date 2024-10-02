from flask import Flask, request, jsonify
import pandas as pd
import requests
import os
import time
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx  # Import NetworkX for creating network graphs
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

app = Flask(__name__)

# Replace this with your actual Telegram Bot Token
BOT_TOKEN = 'Your_bot_token'  # Update this to your bot token

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        data = request.get_json()

        if data is None or 'message' not in data:
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
        # Initialize lists to store the run times
        thresholds = []
        apriori_times = []
        fp_growth_times = []

        # Define different minimum support thresholds to test
        min_support_thresholds = [0.01, 0.05, 0.1, 0.15, 0.2]

        for threshold in min_support_thresholds:
            # Timing Apriori
            start_time = time.time()
            frequent_itemsets_apriori = apriori(df, min_support=threshold, use_colnames=True)
            apriori_time = time.time() - start_time
            
            # Timing FP-Growth
            start_time = time.time()
            frequent_itemsets_fp = fpgrowth(df, min_support=threshold, use_colnames=True)
            fp_growth_time = time.time() - start_time

            # Store the run times and thresholds
            thresholds.append(threshold)
            apriori_times.append(apriori_time * 1000)  # Convert to milliseconds
            fp_growth_times.append(fp_growth_time * 1000)  # Convert to milliseconds

        # Prepare insights for Telegram
        insights = generate_insights(frequent_itemsets_apriori, frequent_itemsets_fp, apriori_times, fp_growth_times)

        # Create and save network model
        network_plot_path = create_network_model(frequent_itemsets_apriori, user_id)

        # Visualization code
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=thresholds, y=fp_growth_times, label="FP-Growth", marker='o')
        sns.lineplot(x=thresholds, y=apriori_times, label="Apriori", marker='o')
        plt.xlabel("Min Support Threshold")
        plt.ylabel("Run Time (ms)")
        plt.title("Run Time of Apriori and FP-Growth Algorithms vs. Min Support Threshold")
        plt.legend()
        plt.grid()
        plt.tight_layout()

        # Save the plot
        plot_path = os.path.join(os.getcwd(), f'runtime_comparison_{user_id}.png')
        plt.savefig(plot_path)
        plt.close()  # Close the plot to avoid display

        # Save CSVs and send insights
        csv_paths = [
            save_table_to_csv(frequent_itemsets_apriori, f"frequent_itemsets_apriori_{user_id}.csv"),
            save_table_to_csv(association_rules(frequent_itemsets_apriori, metric="confidence", min_threshold=0.7), f"rules_apriori_{user_id}.csv"),
            save_table_to_csv(frequent_itemsets_fp, f"frequent_itemsets_fp_{user_id}.csv"),
            save_table_to_csv(association_rules(frequent_itemsets_fp, metric="confidence", min_threshold=0.7), f"rules_fp_{user_id}.csv"),
            plot_path,  # Include the plot in the CSV paths
            network_plot_path  # Include the network plot in the CSV paths
        ]

        return insights, csv_paths

    except Exception as e:
        return f"Data processing failed: {str(e)}", []

def generate_insights(rules_apriori, rules_fp, apriori_times, fp_growth_times):
    insights = []
    
    # Calculate average times
    avg_apriori_time = sum(apriori_times) / len(apriori_times)
    avg_fp_growth_time = sum(fp_growth_times) / len(fp_growth_times)

    insights.append(f"Apriori Algorithm:\n- Average Execution Time: {avg_apriori_time:.4f} ms\n- Rules Generated: {len(rules_apriori)}")
    insights.append(f"FP-Growth Algorithm:\n- Average Execution Time: {avg_fp_growth_time:.4f} ms\n- Rules Generated: {len(rules_fp)}")

    if avg_apriori_time < avg_fp_growth_time:
        insights.append("Suggestion: Use the Apriori algorithm for better performance.")
    else:
        insights.append("Suggestion: Use the FP-Growth algorithm for better performance.")

    return "\n".join(insights)

def create_network_model(frequent_itemsets, user_id):
    # Generate rules from frequent itemsets
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

    # Create a directed graph
    G = nx.DiGraph()

    for _, rule in rules.iterrows():
        G.add_edge(tuple(rule['antecedents']), tuple(rule['consequents']), weight=rule['confidence'])

    # Draw the graph
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)  # positions for all nodes
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=10, font_weight='bold', arrows=True)
    
    # Draw edge labels
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title('Association Rule Network')
    
    # Save the network plot
    network_plot_path = os.path.join(os.getcwd(), f'network_model_{user_id}.png')
    plt.savefig(network_plot_path)
    plt.close()  # Close the plot to avoid display

    return network_plot_path

def preprocess_transaction_data(df):
    # Assuming the CSV is in a suitable format, you might want to perform any additional preprocessing here
    # For example, handling missing values or filtering certain columns
    return df

def save_table_to_csv(df, filename):
    # Ensure the CSV is saved in the correct location
    path = os.path.join(os.getcwd(), filename)
    df.to_csv(path, index=False)
    return path

def send_telegram_message(user_id, text):
    requests.post(f'https://api.telegram.org/bot{BOT_TOKEN}/sendMessage', data={'chat_id': user_id, 'text': text})

def send_telegram_file(user_id, file_path):
    with open(file_path, 'rb') as file:
        requests.post(f'https://api.telegram.org/bot{BOT_TOKEN}/sendDocument', data={'chat_id': user_id}, files={'document': file})

def run_gemma_model(insights):
    # Placeholder for running Gemma
    return "Here are some analysis like association rules, network graph , comparision curve etc."

if __name__ == '__main__':
    app.run(port=5004)
