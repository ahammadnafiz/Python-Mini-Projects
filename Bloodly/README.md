# Bangladesh Blood Donation Bot

A Telegram bot designed to connect blood donors with those in need across Bangladesh. The bot facilitates blood donation requests, emergency blood needs, and allows users to manage their donor profiles.

## Features

- **Start Interaction**: Provides a welcoming message and options to donate blood, find blood, or make an emergency request.
- **Location Sharing**: Accepts and processes user location data through coordinates, Google Maps links, or addresses.
- **Blood Type Management**: Collects and stores donor blood type, location, and contact information.
- **Emergency Requests**: Finds and displays nearest donors based on the requested blood type.
- **Profile Management**: Allows users to view and update their donor profile information.
- **Error Handling**: Provides user-friendly error messages and logs for debugging.

## Installation

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/ahammadnafiz/Python-Mini-Projects.git
    cd Bloodly
    ```

2. **Create a Virtual Environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Set Up Environment Variables**:
    Create a `.env` file in the root directory and add your Telegram Bot API token:
    ```
    API=your-telegram-bot-api-token
    ```

4. **Set Up the Database**:
    The bot will automatically set up the SQLite database upon first run.

## Usage

1. **Run the Bot**:
    ```bash
    python blood_donation_bot.py
    ```

2. **Interact with the Bot**:
    - Start the bot by sending `/start`.
    - Use the provided options to donate blood, find blood, make an emergency request, or view/update your profile.
    - Follow the prompts to provide your location, blood type, and contact information.

## Commands

- `/start` - Start the bot and view the main menu.
- `/help` - Display the help message.
- `/cancel` - Cancel the current operation.

## Development

- **Logging**: Logs are configured to capture bot activities and errors.
- **Geocoding**: Uses `geopy` for location services.
- **Database**: Utilizes `aiosqlite` for asynchronous database interactions.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on GitHub.
