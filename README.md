
# BasketBuddy

**BasketBuddy** is a product recommendation system designed to help retailers maximize sales by suggesting complementary products based on customer shopping behaviors. Utilizing advanced algorithms and data analysis techniques, BasketBuddy provides actionable insights that enhance the shopping experience and drive conversions.



## Features

- **Product Recommendations**: Suggests complementary products based on association rules derived from transaction data.
- **Performance Analysis**: Provides insights into algorithm performance using metrics like execution time and rule generation.
- **Visualization**: Generates visualizations of product associations and algorithm performance for better understanding.
- **Webhook Integration**: Connects with Telegram for seamless user interactions, allowing users to upload data and receive recommendations.

## Technologies Used

- **Python**: Core programming language for implementing algorithms and data processing.
- **Flask**: Framework for creating the web application and handling requests.
- **Pandas**: Library for data manipulation and analysis.
- **MLxtend**: Library for implementing the Apriori and FP-Growth algorithms.
- **Seaborn/Matplotlib**: Libraries for data visualization.
- **NetworkX**: Library for creating and visualizing network graphs of product associations.
- **Telegram Bot API**: For user interactions via Telegram.

## Installation

To get started with BasketBuddy, clone this repository and install the required packages:

```bash
git clone https://github.com/yourusername/basketbuddy.git
cd basketbuddy
pip install -r requirements.txt
```

## Usage

1. Start the Flask application:

   ```bash
   python app.py
   ```

2. Interact with the BasketBuddy bot on Telegram:
   - Use `/start` to initiate the bot.
   - Upload your CSV file containing transaction data.
   - Use `/preprocess` to get guidance on formatting your data correctly.
   - Receive product recommendations based on your data.

## API Endpoints

- **POST /webhook**: This endpoint receives messages from Telegram and processes CSV uploads.
- **GET /status**: (optional) Check the status of the BasketBuddy application.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue to discuss improvements.

1. Fork the repository.
2. Create your feature branch: `git checkout -b feature/YourFeature`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/YourFeature`
5. Open a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to modify any sections to better fit your project's specifics or personal preferences!
