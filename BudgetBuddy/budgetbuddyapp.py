import streamlit as st
import os
import json
import pandas as pd
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
import googleapiclient.discovery
from datetime import datetime, timezone, timedelta
import plotly.express as px
import numpy as np


#Core functions

def load_data():
    """
    Load existing financial data from a JSON file if it exists.
    """
    data = {'transactions': [], 'expense_breakdown': {}, 'main_income_balance': 0, 'budgets': {}}
    if os.path.exists('budgetbuddy.json'):
        with open('budgetbuddy.json', 'r') as file:
            loaded_data = json.load(file)
            data.update(loaded_data)
    return data


def save_data(data):
    """
    Save financial data to a JSON file.
    """
    with open('budgetbuddy.json', 'w') as file:
        json.dump(data, file, indent=2)


#Data manipulation functions

def add_transaction(data, amount, category, description, transaction_type, timestamp):
    """
    Add a new financial transaction to the data.
    """
    
    if transaction_type.lower() == 'expense':
        amount *= -1  # Expenses are represented as negative amounts
        data['main_income_balance'] += amount  # Update main income balance for expenses

    transaction = {'amount': amount, 'category': category, 'description': description, 'timestamp': timestamp,
                   'transaction_type': transaction_type, 'source': None}

    data['transactions'].append(transaction)

    # Update expense breakdown
    if transaction_type.lower() == 'expense':
        if 'expense_breakdown' not in data:
            data['expense_breakdown'] = {}

        if category not in data['expense_breakdown']:
            data['expense_breakdown'][category] = {'total': 0, 'transactions': []}

        data['expense_breakdown'][category]['total'] += amount
        data['expense_breakdown'][category]['transactions'].append(transaction)
        
    # Automatically add to Google Calendar
    add_to_google_calendar(timestamp, description)
    save_data(data)


def delete_all_data():
    # Open the JSON file in write mode
    with open('budgetbuddy.json', 'w') as file:
        # Write an empty dictionary to the file
        json.dump({}, file)

# Call the function to delete all data
#delete_all_data()


#Viewing functions

def view_balance(financial_data):
    st.header("View Balance")

    # Set background color and padding
    st.markdown(
        """
        <style>
            .view-balance-container {
                background-color: #e6f7ff; 
                padding: 2em;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }

            .balance-text {
                color: #3399cc; 
                font-size: 1.5em;
                font-weight: bold;
                text-align: center;
                margin-bottom: 20px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    with st.container():
        
        main_income_balance = financial_data.get("main_income_balance", 0)

        st.write(f"<p class='balance-text'>Main Income Balance: {main_income_balance}</p>", unsafe_allow_html=True)


def view_transactions(data):
    """
    Display all transactions, including a breakdown of expenses by category.
    """
    print("\nAll Transactions:")
    for idx, transaction in enumerate(data['transactions']):
        timestamp = transaction['timestamp']
        description = transaction['description']
        amount = transaction['amount']
        category = transaction['category']
        
        # Check if 'transaction_type' key exists before accessing it
        if 'transaction_type' in transaction:
            transaction_type = transaction['transaction_type']
        else:
            transaction_type = "Unknown"

        source = transaction.get('source', '')

        if transaction_type.lower() == 'expense':
            print(f"{idx + 1}. {timestamp} - {description}: {amount} ({category})")
        elif transaction_type.lower() == 'income':
            print(f"{idx + 1}. {timestamp} - {description} from {source}: {amount}")

    if not data['transactions']:
        print("No transactions found.")


def view_expense_breakdown(data):
    """
    Display a breakdown of expenses by category.
    """
    print("\nExpense Breakdown:")
    for category, info in data['expense_breakdown'].items():
        print(f"{category}: {info['total']}")
        for transaction in info['transactions']:
            print(f"  {transaction['timestamp']} - {transaction['description']}: {transaction['amount']}")


def view_transactions_by_category(data, category):
    """
    Display transactions for a specific category.
    """
    print(f"\nTransactions for Category '{category}':")
    for transaction in data['transactions']:
        if transaction['category'] == category:
            print(f"{transaction['timestamp']} - {transaction['description']}: {transaction['amount']}")


def view_budget(data):
    """
    Display the budgets set for each expense category.
    """
    if 'budgets' not in data or not data['budgets']:
        print("No budgets set.")
        return

    print("\nExpense Budgets:")
    for category, budget_amount in data['budgets'].items():
        print(f"{category}: {budget_amount}")


def view_savings(data):
    """
    Display all savings transactions.
    """

    # Set background color and padding
    st.markdown(
        """
        <style>
            .view-savings-container {
                background-color: #d3ffd3; 
                padding: 2em;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }

            .header-text {
                color: #4caf50; 
                font-size: 1.5em;
                font-weight: bold;
                text-align: center;
                margin-bottom: 20px;
            }

            .savings-text {
                color: #4caf50; 
                text-align: center;
                margin-top: 20px;
            }

            .savings-table {
                margin-top: 20px;
                text-align: center;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    with st.container():
        # Display HTML content with custom styles for a centered box with space after the text
        st.markdown(
            """
            <div class="add-income-container" style="display: flex; flex-direction: column; align-items: center; justify-content: center; background-color: #eaeaea; padding: 10px; border-radius: 5px; margin-bottom: 50px;">
                <label class="header-text" style="color: #100b09; font-size: 20px;">View Savings</label>
                <!-- Your input fields go here -->
            </div>
            """,
            unsafe_allow_html=True
        )

        savings_transactions = [transaction for transaction in data.get('transactions', []) if transaction.get('category') == 'Savings']

        if savings_transactions:
            # Create a list of dictionaries for each savings transaction
            table_data = [
                {
                    "Index": idx,
                    "Timestamp": transaction.get('timestamp', ''),
                    "Description": transaction.get('description', ''),
                    "Amount": transaction.get('amount', ''),
                }
                for idx, transaction in enumerate(savings_transactions, 1)
            ]

            # Display the savings transactions in a table
            st.table(pd.DataFrame(table_data).set_index("Index").style.set_table_styles([{'selector': 'td', 'props': [('text-align', 'center')]}, {'selector': 'th', 'props': [('text-align', 'center')]}]))
        else:
            st.markdown("<div class='savings-text'>No savings transactions found.</div>", unsafe_allow_html=True)


#Category management

def add_expense_category(data, category):
    """
    Add a new expense category to the data.
    """
    if 'expense_categories' not in data:
        data['expense_categories'] = []

    data['expense_categories'].append(category)


#Budget management

def check_budget(data, amount, category):
    """
    Check if a transaction exceeds the budget for its category.
    Display a warning if the budget is exceeded.
    """
    if 'budgets' in data and category in data['budgets']:
        budget = data['budgets'][category]
        if amount < 0 and abs(amount) > budget:
            print(f"Warning: Transaction exceeds budget for category '{category}'!")
            

def set_budget(data, category, budget_amount):
    """
    Set a budget for a specific expense category.
    """
    if 'budgets' not in data:
        data['budgets'] = {}
    data['budgets'][category] = budget_amount


#Income management

def add_income(data, amount, source, date, time):
    """
    Add income to the data.
    """
    timestamp = f"{date} {time}"
    data['main_income_balance'] += amount
    transaction = {'amount': amount, 'category': 'Income', 'description': f"Income from {source}", 'timestamp': timestamp, 'transaction_type': 'income', 'source': source}
    data['transactions'].append(transaction)


#Goals management

def set_financial_goal(data, goal_name, goal_amount, goal_target_date):
    goal = {'name': goal_name, 'amount': goal_amount, 'target_date': goal_target_date, 'progress': 0}
    if 'financial_goals' not in data:
        data['financial_goals'] = []

    data['financial_goals'].append(goal)
    # You may want to save the data here or update it in another way
    save_data(data)
    st.success(f"Financial goal '{goal_name}' set successfully!")


def track_financial_goals(data):
    if 'financial_goals' in data and data['financial_goals']:
        st.markdown(
                """
                <div class="view-savings-container" style="display: flex; flex-direction: column; align-items: center; justify-content: center; background-color: #eaeaea; padding: 20px; border-radius: 10px; margin-bottom: 50px;">
                    <label class="header-text" style="color: #100b09; font-size: 20px; font-weight: bold; margin-bottom: 15px;">Track Financial Goal</label>
                    <!-- Your content goes here -->
                    <!-- For example, you can add a table to display savings data -->
                </div>
                """,
                unsafe_allow_html=True
            )
        # Display each financial goal
        for goal in data['financial_goals']:
            current_balance = calculate_current_balance(data)
            progress_percentage = (current_balance / goal['amount']) * 100

            # Ensure progress_percentage does not exceed 100
            progress_percentage = min(progress_percentage, 100)

            goal['progress'] = progress_percentage  # Update the progress in the goal dictionary

            # Format the output with two decimal places for progress and one decimal place for main income balance
            formatted_progress = "{:.2f}%".format(progress_percentage)
            formatted_main_income_balance = "{:.1f}".format(data.get('main_income_balance', 0.0))

            # Display goal details
            with st.container():
                st.subheader(goal['name'])
                st.write(f"Target Amount: BDT {goal['amount']:.2f}")
                st.write(f"Target Date: {goal['target_date']}")
                st.write(f"Main Income Balance: BDT {formatted_main_income_balance}")
                
                # Display progress bar
                st.progress(progress_percentage / 100)

                # Display progress percentage
                st.write(f"Progress: {formatted_progress}")

            # Add some space between goals
            st.markdown("<br>", unsafe_allow_html=True)
    else:
        st.info("No financial goals set.")


def view_financial_goals(data):
    """
    Display and interact with financial goals.
    """
    
    # Set background color and padding
    st.markdown(
        """
        <style>
            .financial-goals-container {
                background-color: #ffeeba;
                padding: 2em;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0.1, 0.1);
            }

            .header-text {
                color: #ff9800;
                font-size: 1.5em;
                font-weight: bold;
                text-align: center;
                margin-bottom: 20px;
            }

            .goal-item {
                margin-bottom: 10px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    with st.container():
        # Display HTML content with custom styles for a centered box with space after the text
        st.markdown(
            """
            <div class="add-income-container" style="display: flex; flex-direction: column; align-items: center; justify-content: center; background-color: #eaeaea; padding: 10px; border-radius: 5px; margin-bottom: 50px;">
                <label class="header-text" style="color: #100b09; font-size: 20px;">View Financial Goals</label>
                <!-- Your input fields go here -->
            </div>
            """,
            unsafe_allow_html=True
        )

        financial_goals = data.get('financial_goals', [])

        for idx, goal in enumerate(financial_goals, 1):
            goal_name = goal.get('name', '')
            goal_amount = goal.get('amount', '')
            goal_target_date = goal.get('target_date', '')

            st.markdown(
                f"<div class='goal-item'>{idx}. {goal_name} - Amount: {goal_amount}, Target Date: {goal_target_date}</div>",
                unsafe_allow_html=True
            )

        if not financial_goals:
            st.markdown("<div>No financial goals found.</div>", unsafe_allow_html=True)


#Calendar integration
          
def get_dhaka_timestamp():
    # Set the time zone for Dhaka (UTC+6)
    dhaka_timezone = timezone(timedelta(hours=6))
    
    # Get the current time in UTC
    current_time_utc = datetime.now(timezone.utc)
    
    # Convert UTC time to Dhaka time
    current_time_dhaka = current_time_utc.astimezone(dhaka_timezone)
    
    # Format the timestamp
    formatted_timestamp = current_time_dhaka.strftime('%Y-%m-%dT%H:%M:%S%z')
    
    return formatted_timestamp


def add_to_google_calendar(timestamp, description):
    """
    Add a transaction to Google Calendar.
    """
    credentials = None
    token = 'token.json'  # Change this to your token file name
    timestamp = get_dhaka_timestamp()
    
    
    if os.path.exists(token):
        credentials = Credentials.from_authorized_user_file(token)

    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', ['https://www.googleapis.com/auth/calendar.events'])
            credentials = flow.run_local_server(port=0)

        with open(token, 'w') as token_file:
            token_file.write(credentials.to_json())

    service = googleapiclient.discovery.build('calendar', 'v3', credentials=credentials)

    event = {
        'summary': description,
        'description': f'Transaction: {description}',
        'start': {
            'dateTime': timestamp,
            'timeZone': 'Asia/Dhaka',  # Use the correct time zone for Dhaka
        },
        'end': {
            'dateTime': timestamp,
            'timeZone': 'Asia/Dhaka',  # Use the correct time zone for Dhaka
        },
    }

    calendar_id = 'anafiz2330006@bsds.uiu.ac.bd'  # Change this to your calendar ID
    service.events().insert(calendarId=calendar_id, body=event).execute()


#Utils

def calculate_balance(data):
    """
    Calculate the current balance based on transactions.
    """
    income = sum(transaction['amount'] for transaction in data['transactions'] if transaction['amount'] > 0)
    expenses = sum(transaction['amount'] for transaction in data['transactions'] if transaction['amount'] < 0)
    balance = income + expenses
    return balance


def calculate_current_balance(data):

    return sum(transaction['amount'] for transaction in data.get('transactions', []))


def set_custom_theme():
    # Read theme settings from a configuration file or user input
    theme = """
        <style>
            body {
                background-color: #0C1618;
            }
            .streamlitFrame {
                background-color: #0C1618;
            }
            .streamlit-expanderHeader {
                color: #0C1618;
            }
            .streamlit-expanderContent {
                color: #ffffff;
            }
            .streamlit-button {
                background-color: #0C1618;
                color: #0a0e12;
            }
        </style>
    """

    # Apply the theme
    st.markdown(theme, unsafe_allow_html=True)


def add_icon_to_menu(menu_item, icon):
    return f"{icon} {menu_item}"


def json_to_dataframe(data):
    '''
    Convert JSON data to DataFrame
    '''
    # Assuming your JSON data has a 'transactions' key containing a list of transactions
    transactions = data.get('transactions', [])
    df = pd.DataFrame(transactions)
    return df


def prepare_expense_breakdown(data):
    expense_breakdown = data.get("expense_breakdown", {})
    breakdown_list = []

    for category, details in expense_breakdown.items():
        breakdown_list.append({
            "description": category,
            "amount": abs(details["total"])  # Make the amount positive
        })

    return pd.DataFrame(breakdown_list)

import plotly.graph_objects as go

def create_dashboard(financial_data):
    # Check if transactions are available
    if 'transactions' not in financial_data:
        st.warning("No transaction data available.")
        return

    # Create DataFrame from transactions
    transactions_df = pd.DataFrame(financial_data['transactions'])

    # Visualize data
    st.subheader("Budget Dashboard")

    # Set background color and padding
    st.markdown(
        """
        <style>
            .view-balance-container {
                background-color: #e6f7ff; 
                padding: 2em;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }

            .balance-text {
                color: #3399cc; 
                font-size: 1.5em;
                font-weight: bold;
                text-align: center;
                margin-bottom: 20px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    with st.container():
        main_income_balance = financial_data.get("main_income_balance", 0)

        # Display balance and check for 0 balance
        if main_income_balance == 0:
            st.warning("Your current balance is 0. Please review your financial data.")
            return
        else:
            st.write(f"<p class='balance-text'>Current Balance: {main_income_balance}</p>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

    
    # Assuming transactions_df is your DataFrame
    transactions_df['amount'] = np.abs(transactions_df['amount'])

    # Sort DataFrame by timestamp
    transactions_df = transactions_df.sort_values(by='timestamp')

    # Create a scatter plot with curved line
    fig = go.Figure()

    for category, data in transactions_df.groupby('category'):
        fig.add_trace(go.Scatter(
            x=data['timestamp'],
            y=data['amount'],
            mode='lines+markers',
            name=category,
            line=dict(shape='spline'),
        ))

    # Customize the layout if needed
    fig.update_layout(
        xaxis=dict(title="Timestamp"),
        yaxis=dict(title="Amount"),
        title="Transactions Over Time",
        template="plotly_dark",
    )

    st.plotly_chart(fig)


    # Group by category and sum the amounts
    category_amounts = transactions_df.groupby("category")["amount"].sum().reset_index()

    # Assuming 'amount' column contains both positive and negative values
    category_amounts['amount'] = np.abs(category_amounts['amount'])

    # Create an interactive bar chart with Plotly
    fig = px.bar(category_amounts, x='category', y='amount', color='category',
                labels={'amount': 'Total Amount', 'category': 'Category'},
                title='Transaction Amounts by Category',
                template='plotly_dark')  # Use a dark template for a dark background

    # Customize the layout if needed
    fig.update_layout(
        xaxis=dict(title='Category'),
        yaxis=dict(title='Total Amount'),
    )

    st.plotly_chart(fig)

    expense_breakdown_df = prepare_expense_breakdown(financial_data)

    # Check if 'description' column exists in expense_breakdown_df
    if 'description' not in expense_breakdown_df.columns:
        st.warning("The 'description' column is not present in the expense breakdown data.")
        return
    
    
    # Create an interactive pie chart with Plotly
    fig = px.pie(expense_breakdown_df, values='amount', names='description',
                title='Expense Breakdown', hole=0.3,
                template='plotly_dark')  # Use a dark template for a dark background

    st.plotly_chart(fig)
    
    st.markdown("<h4 style='text-align: center; color: #FAFAFA;'>Calender View</h4>", unsafe_allow_html=True)
    # Google Calendar embed code
    calendar_embed_code = """
    <iframe src="https://calendar.google.com/calendar/embed?height=500&wkst=7&bgcolor=%23ffffff&ctz=Asia%2FDhaka&showTitle=0&showPrint=0&showCalendars=1&showTabs=0&src=YW5hZml6MjMzMDAwNkBic2RzLnVpdS5hYy5iZA&src=YWRkcmVzc2Jvb2sjY29udGFjdHNAZ3JvdXAudi5jYWxlbmRhci5nb29nbGUuY29t&src=ZW4uYmQjaG9saWRheUBncm91cC52LmNhbGVuZGFyLmdvb2dsZS5jb20&color=%23039BE5&color=%2333B679&color=%230B8043" style="border:solid 1px #777" width="700" height="500" frameborder="0" scrolling="no"></iframe>
    """

    # Display the Google Calendar using an iframe
    st.markdown(calendar_embed_code, unsafe_allow_html=True)

    # Add some space after the calendar
    st.markdown("<br>", unsafe_allow_html=True)


#Savings

def add_savings(data, amount, description, date, time):
    """
    Add a savings transaction to the data and adjust the main income balance.
    """
    timestamp = f"{date} {time}"
    savings_transaction = {'amount': amount, 'category': 'Savings', 'description': description, 'timestamp': timestamp}
    data['transactions'].append(savings_transaction)
    
    # Adjust main income balance by subtracting the savings amount
    data['main_income_balance'] -= amount


#Sections

def view_transactions_section(data):

    # Set background color and padding
    st.markdown(
        """
        <style>
            .view-transactions-container {
                background-color: #f0f5f5;
                padding: 2em;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }

            .header-text {
                color: #009688;
                font-size: 1.5em;
                font-weight: bold;
                text-align: center;
                margin-bottom: 20px;
            }

            .expense-text {
                color: #e74c3c; /* Red */
            }

            .income-text {
                color: #2ecc71; /* Green */
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    with st.container():
        # Display HTML content with custom styles for a centered box with space after the text
        st.markdown(
            """
            <div class="add-income-container" style="display: flex; flex-direction: column; align-items: center; justify-content: center; background-color: #eaeaea; padding: 10px; border-radius: 5px; margin-bottom: 50px;">
                <label class="header-text" style="color: #100b09; font-size: 20px;">View Transactions</label>
                <!-- Your input fields go here -->
            </div>
            """,
            unsafe_allow_html=True
        )

        transactions = data.get('transactions', [])

        # Create a list of dictionaries for each transaction
        table_data = [
            {
                "Index": idx,
                "Timestamp": transaction['timestamp'],
                "Description": transaction['description'],
                "Amount": transaction['amount'],
                "Category": transaction['category'],
                "Transaction Type": transaction.get('transaction_type', 'Unknown'),
                "Source": transaction.get('source', ''),
            }
            for idx, transaction in enumerate(transactions, 1)
        ]

        # Display the table
        st.table(pd.DataFrame(table_data).set_index("Index"))

        if not transactions:
            st.markdown("<div>No transactions found.</div>", unsafe_allow_html=True)

            
def view_expense_breakdown_section(data):
    # Set background color and padding
    st.markdown(
        """
        <style>
            .expense-breakdown-container {
                background-color: #f0f5f5;
                padding: 2em;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }

            .header-text {
                color: #e74c3c; /* Red */
                font-size: 1.5em;
                font-weight: bold;
                text-align: center;
                margin-bottom: 20px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    with st.container():
        # Display HTML content with custom styles for a centered box with space after the text
        st.markdown(
            """
            <div class="add-income-container" style="display: flex; flex-direction: column; align-items: center; justify-content: center; background-color: #eaeaea; padding: 10px; border-radius: 5px; margin-bottom: 50px;">
                <label class="header-text" style="color: #100b09; font-size: 20px;">Expense Breakdown</label>
                <!-- Your input fields go here -->
            </div>
            """,
            unsafe_allow_html=True
        )

        expense_breakdown = data.get('expense_breakdown', {})

        if expense_breakdown:
            # Create a DataFrame for expense breakdown
            df_expense_breakdown = pd.DataFrame(expense_breakdown.items(), columns=['Category', 'Info'])
            df_expense_breakdown['Total Expense'] = df_expense_breakdown['Info'].apply(lambda x: x['total'])
            df_expense_breakdown.drop('Info', axis=1, inplace=True)

            # Display the table
            st.table(df_expense_breakdown.set_index("Category"))
        else:
            st.markdown("<div>No expense breakdown found.</div>", unsafe_allow_html=True)


def view_transactions_by_category_section(data, category):
    st.header(f"Transactions for Category '{category}'")

    # Set background color and padding
    st.markdown(
        """
        <style>
            .transactions-by-category-container {
                background-color: #f0f5f5;
                padding: 2em;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }

            .header-text {
                color: #009688;
                font-size: 1.5em;
                font-weight: bold;
                text-align: center;
                margin-bottom: 20px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    with st.container():
        # Display HTML content with custom styles for a centered box with space after the text
        st.markdown(
            """
            <div class="add-income-container" style="display: flex; flex-direction: column; align-items: center; justify-content: center; background-color: #eaeaea; padding: 10px; border-radius: 5px; margin-bottom: 50px;">
                <label class="header-text" style="color: #100b09; font-size: 20px;">Transactions for Category</label>
                <!-- Your input fields go here -->
            </div>
            """,
            unsafe_allow_html=True
        )

        transactions = data.get('transactions', [])
        filtered_transactions = [t for t in transactions if t['category'] == category]

        if filtered_transactions:
            # Create a list of dictionaries for each transaction
            table_data = [
                {
                    "Timestamp": transaction['timestamp'],
                    "Description": transaction['description'],
                    "Amount": transaction['amount'],
                    "Category": transaction['category'],
                    "Transaction Type": transaction.get('transaction_type', 'Unknown'),
                    "Source": transaction.get('source', ''),
                }
                for transaction in filtered_transactions
            ]

            # Display the table
            st.table(pd.DataFrame(table_data))
        else:
            st.markdown(f"<div>No transactions found for category '{category}'.</div>", unsafe_allow_html=True)


def view_budget_section(data):

    # Set background color and padding
    st.markdown(
        """
        <style>
            .view-budget-container {
                background-color: #f0f5f5;
                padding: 2em;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }

            .header-text {
                color: #009688;
                font-size: 1.5em;
                font-weight: bold;
                text-align: center;
                margin-bottom: 20px;
            }

            .budget-table th {
                background-color: #add8e6; /* Light Blue */
            }

            .budget-table td {
                padding: 10px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    with st.container():
        # Display HTML content with custom styles for a centered box with space after the text
        st.markdown(
            """
            <div class="add-income-container" style="display: flex; flex-direction: column; align-items: center; justify-content: center; background-color: #eaeaea; padding: 10px; border-radius: 5px; margin-bottom: 50px;">
                <label class="header-text" style="color: #100b09; font-size: 20px;">View Budget</label>
                <!-- Your input fields go here -->
            </div>
            """,
            unsafe_allow_html=True
        )

        budgets = data.get('budgets', {})

        if budgets:
            # Create a list of dictionaries for each budget
            table_data = [
                {
                    "Category": category,
                    "Budget Amount": budget_amount,
                }
                for category, budget_amount in budgets.items()
            ]

            # Display the table
            st.table(pd.DataFrame(table_data).set_index("Category").style.set_table_styles([{'selector': 'thead', 'props': [('background-color', '#070f17')]}]))
        else:
            st.markdown("<div class='header-text'>No budgets set.</div>", unsafe_allow_html=True)


def add_expense_category_section(data):
   

    # Set background color and padding
    st.markdown(
        """
        <style>
            .add-expense-category-container {
                background-color: #f0f5f5;
                padding: 2em;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }

            .header-text {
                color: #009688;
                font-size: 1.5em;
                font-weight: bold;
                text-align: center;
                margin-bottom: 20px;
            }

            label {
                color: #666666;
                font-size: 1em;
                font-weight: bold;
                margin-bottom: 5px;
            }

            input, select {
                width: 100%;
                padding: 8px;
                margin-bottom: 15px;
                border: 1px solid #ccc;
                border-radius: 4px;
                box-sizing: border-box;
            }

            .button-container {
                text-align: center;
            }

            .success-message {
                color: #33cc33; /* Green */
                font-weight: bold;
                text-align: center;
                margin-top: 15px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    with st.container():
        # Display HTML content with custom styles for a centered box with space after the text
        st.markdown(
            """
            <div class="add-income-container" style="display: flex; flex-direction: column; align-items: center; justify-content: center; background-color: #eaeaea; padding: 10px; border-radius: 5px; margin-bottom: 50px;">
                <label class="header-text" style="color: #100b09; font-size: 20px;">Add Expense</label>
                <!-- Your input fields go here -->
            </div>
            """,
            unsafe_allow_html=True
        )

        category = st.text_input("Enter Expense Category:", help="Enter the expense category")

        if st.button("Add Category"):
            with st.spinner("Adding Category..."):
                add_expense_category(data, category)
                save_data(data)
                st.success("Category added successfully!")


def set_budget_section(data):
    
    # Set background color and padding
    st.markdown(
        """
        <style>
            .set-budget-container {
                background-color: #f0f5f5;
                padding: 2em;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }

            .header-text {
                color: #009688;
                font-size: 1.5em;
                font-weight: bold;
                text-align: center;
                margin-bottom: 20px;
            }

            label {
                color: #666666;
                font-size: 1em;
                font-weight: bold;
                margin-bottom: 5px;
            }

            input, select {
                width: 100%;
                padding: 8px;
                margin-bottom: 15px;
                border: 1px solid #ccc;
                border-radius: 4px;
                box-sizing: border-box;
            }

            .button-container {
                text-align: center;
            }

            .success-message {
                color: #33cc33; /* Green */
                font-weight: bold;
                text-align: center;
                margin-top: 15px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    with st.container():
        # Display HTML content with custom styles for a centered box with space after the text
        st.markdown(
            """
            <div class="add-income-container" style="display: flex; flex-direction: column; align-items: center; justify-content: center; background-color: #eaeaea; padding: 10px; border-radius: 5px; margin-bottom: 50px;">
                <label class="header-text" style="color: #100b09; font-size: 20px;">Set Budget</label>
                <!-- Your input fields go here -->
            </div>
            """,
            unsafe_allow_html=True
        )

        # Check if 'expense_categories' key exists in data
        if 'expense_categories' in data:
            categories = st.multiselect("Select Expense Categories:", data['expense_categories'])

            budget_amount = st.number_input("Enter Budget Amount:", step=0.01, help="Enter the budget amount")

            if st.button("Set Budget"):
                with st.spinner("Setting Budget..."):
                    for category in categories:
                        set_budget(data, category, budget_amount)

                    save_data(data)
                    st.success("Budget set successfully!")
        else:
            st.warning("Expense categories not found in the data. Please add some categories first.")


def add_income_section(financial_data):

    # Set background color and padding
    st.markdown(
        """
        <style>
            .add-income-container {
                background-color: #e6f7ff;
                padding: 2em;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }

            .header-text {
                color: #3399cc;
                font-size: 1.5em;
                font-weight: bold;
                text-align: center;
                margin-bottom: 20px;
            }

            label {
                color: #666666;
                font-size: 1em;
                font-weight: bold;
                margin-bottom: 5px;
            }

            input, select {
                width: 100%;
                padding: 8px;
                margin-bottom: 15px;
                border: 1px solid #ccc;
                border-radius: 4px;
                box-sizing: border-box;
            }

            .button-container {
                text-align: center;
            }

            .success-message {
                color: #33cc33;
                font-weight: bold;
                text-align: center;
                margin-top: 15px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    with st.container():
        # Display HTML content with custom styles for a centered box with space after the text
        st.markdown(
            """
            <div class="add-income-container" style="display: flex; flex-direction: column; align-items: center; justify-content: center; background-color: #eaeaea; padding: 10px; border-radius: 5px; margin-bottom: 50px;">
                <label class="header-text" style="color: #100b09; font-size: 20px;">Add Income</label>
                <!-- Your input fields go here -->
            </div>
            """,
            unsafe_allow_html=True
        )

        amount = st.number_input("Amount:", step=0.01, help="Enter the income amount")
        source = st.text_input("Source:", help="Enter the source of income")
        date = st.date_input("Date:", help="Select the income date")
        time = st.time_input("Time:", help="Select the income time")

        if st.button("Add Income"):
            with st.spinner("Adding Income..."):
                add_income(financial_data, amount, source, str(date), str(time))
                save_data(financial_data)
                st.success("Income added successfully!")


def add_transaction_section(financial_data):

    # Set background color and padding
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

            body {
                font-family: 'Poppins', sans-serif;
            }

            .main-container {
                background-color: #e6f7ff; 
                padding: 2em;
            }

            .add-transaction-container {
                background-color: #fff5e6; 
                padding: 2em;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }

            .header-text {
                color: #3399cc; 
                font-size: 1.5em;
                font-weight: bold;
                text-align: center;
                margin-bottom: 20px;
            }

            label {
                color: #666666;
                font-size: 1em;
                font-weight: bold;
                margin-bottom: 5px;
            }

            input, select {
                width: 100%;
                padding: 8px;
                margin-bottom: 15px;
                border: 1px solid #ccc;
                border-radius: 4px;
                box-sizing: border-box;
            }

            .button-container {
                text-align: center;
            }

            .success-message {
                color: #33cc33; /* Green */
                font-weight: bold;
                text-align: center;
                margin-top: 15px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    with st.container():
        # Display HTML content with custom styles for a centered box with space after the text
        st.markdown(
            """
            <div class="add-income-container" style="display: flex; flex-direction: column; align-items: center; justify-content: center; background-color: #eaeaea; padding: 10px; border-radius: 5px; margin-bottom: 50px;">
                <label class="header-text" style="color: #100b09; font-size: 20px;">Add Transaction</label>
                <!-- Your input fields go here -->
            </div>
            """,
            unsafe_allow_html=True
        )
        
        amount = st.number_input("Amount:", step=0.01, help="Enter the transaction amount")
        category = st.text_input("Category:", help="Enter the transaction category")
        description = st.text_input("Description:", help="Enter a brief description")
        transaction_type = st.selectbox("Type:", ["Income", "Expense"], help="Select the transaction type")
        date = st.date_input("Date:", help="Select the transaction date")
        time = st.time_input("Time:", help="Select the transaction time")
        timestamp = f"{date.strftime('%Y-%m-%d')} {time.strftime('%H:%M')}"
        
    if st.button("Add Transaction"):
        with st.spinner("Adding Transaction..."):
            
            # Pass formatted timestamp to add_transaction
            add_transaction(financial_data, amount, category, description, transaction_type.lower(), timestamp)
            check_budget(financial_data, amount, category)
            save_data(financial_data)
            st.success("Transaction added successfully!")


def add_savings_section(data):

    # Set background color and padding
    st.markdown(
        """
        <style>
            .add-savings-container {
                background-color: #d3ffd3; 
                padding: 2em;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }

            .header-text {
                color: #4caf50; 
                font-size: 1.5em;
                font-weight: bold;
                text-align: center;
                margin-bottom: 20px;
            }

            label {
                color: #666666;
                font-size: 1em;
                font-weight: bold;
                margin-bottom: 5px;
            }

            input, select {
                width: 100%;
                padding: 8px;
                margin-bottom: 15px;
                border: 1px solid #ccc;
                border-radius: 4px;
                box-sizing: border-box;
            }

            .button-container {
                text-align: center;
            }

            .success-message {
                color: #4caf50; 
                font-weight: bold;
                text-align: center;
                margin-top: 15px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    with st.container():
        # Display HTML content with custom styles for a centered box with space after the text
        st.markdown(
            """
            <div class="add-income-container" style="display: flex; flex-direction: column; align-items: center; justify-content: center; background-color: #eaeaea; padding: 10px; border-radius: 5px; margin-bottom: 50px;">
                <label class="header-text" style="color: #100b09; font-size: 20px;">Add Savings</label>
                <!-- Your input fields go here -->
            </div>
            """,
            unsafe_allow_html=True
        )

        amount = st.number_input("Amount:", step=0.01, help="Enter the savings amount")
        description = st.text_input("Description:", help="Enter a brief description")
        date = st.date_input("Date:", help="Select the transaction date")
        time = st.time_input("Time:", help="Select the transaction time")

    if st.button("Add Savings"):
        with st.spinner("Adding Savings..."):
            add_savings(data, amount, description, str(date), str(time))
            save_data(data)
            st.success("Savings added successfully!")
            
            
def main():
    # Load existing data or start with an empty dictionary
    financial_data = load_data()
    set_custom_theme()
    
    # Add a hero image
    hero_image = "budgetbuddy.png"  # Replace with the actual path to your image
    st.image(hero_image, use_column_width=True)
        
        
    # Center the text "BudgetBuddy" using an HTML h1 tag
    # st.markdown("<h1 style='text-align: center; color: #32FF72;'>BudgetBuddy</h1>", unsafe_allow_html=True)
    # st.markdown("<p style='text-align: center; color: #fdfcdc;'>Welcome to BudgetBuddy, your personal budget management tool!</p>", unsafe_allow_html=True)
    # st.markdown("---")  # Add a horizontal line for separation

    # Sidebar conten
    st.sidebar.image("letter-b.png", width=50)  # Add your application logo
    st.sidebar.title("BudgetBuddy")
    
    # Navigation bar
    st.sidebar.markdown("---")
    st.sidebar.header("Navigation")

    # Add icons to menu items
    menu_choices = ["Home", "Transactions", "Expenses", "Savings", "Budget", "Goals"]
    icons = ["🏠", "💰", "📉", "💸", "📊", "🎯"]
    selected_page = st.sidebar.radio("Go to", [add_icon_to_menu(menu_item, icon) for menu_item, icon in zip(menu_choices, icons)])

    if 'Home' in selected_page:
        # Display basic statistics from the transactions
        create_dashboard(financial_data)
        
    elif "Transactions" in selected_page:
        transaction_choice = st.sidebar.selectbox("Select an option", ["View Balance", "Add Income", "Add Transaction", "View Transactions", "View Transactions by Category"])
        if transaction_choice == "Add Transaction":
            add_transaction_section(financial_data)
        elif transaction_choice == "View Balance":
            view_balance(financial_data)
        elif transaction_choice == "Add Income":
            add_income_section(financial_data)
        elif transaction_choice == "View Transactions":
            view_transactions_section(financial_data)
        elif transaction_choice == "View Transactions by Category":
            # Custom CSS for text input box size
            text_input_style = """
                <style>
                    div[data-baseweb="input"] {
                        width: 300px; /* Adjust the width as needed */
                    }
                </style>
            """
            # Inject custom CSS for text input box size
            st.markdown(text_input_style, unsafe_allow_html=True)
            # Use st.text_input as usual
            selected_category = st.text_input("Enter Category:", help='Enter the category for you transaction history')
            # selected_category = st.text_input("Enter Category:")
            view_transactions_by_category_section(financial_data, selected_category)
    elif "Expenses" in selected_page:
        expense_choice = st.sidebar.selectbox("Select an option", ["Add Expense Category", "View Expense Breakdown"])
        if expense_choice == "Add Expense Category":
            add_expense_category_section(financial_data)
        elif expense_choice == "View Expense Breakdown":
            view_expense_breakdown_section(financial_data)
    elif "Savings" in selected_page:
        savings_choice = st.sidebar.selectbox("Select an option", ["Add Savings", "View Savings"])
        if savings_choice == "Add Savings":
            add_savings_section(financial_data)
        elif savings_choice == "View Savings":
            view_savings(financial_data)
    elif "Budget" in selected_page:
        budget_choice = st.sidebar.selectbox("Select an option", ["Set Budget", "View Budgets"])
        if budget_choice == "Set Budget":
            set_budget_section(financial_data)
        elif budget_choice == "View Budgets":
            view_budget_section(financial_data)
    elif "Goals" in selected_page:
        goals_choice = st.sidebar.selectbox("Select an option", ["Set Financial Goal", "View Financial Goals", "Track Financial Goals"])
        if goals_choice == "Set Financial Goal":
            st.markdown(
                """
                <div class="view-savings-container" style="display: flex; flex-direction: column; align-items: center; justify-content: center; background-color: #eaeaea; padding: 10px; border-radius: 10px; margin-bottom: 50px;">
                    <label class="header-text" style="color: #100b09; font-size: 20px; font-weight: bold; margin-bottom: 15px;">Set Financial Goal</label>
                    <!-- Your content goes here -->
                    <!-- For example, you can add a table to display savings data -->
                </div>
                """,
                unsafe_allow_html=True
            )
            goal_name = st.text_input("Enter the name of the financial goal:")
            goal_amount = st.number_input("Enter the target amount for the goal:", min_value=0.0)
            goal_target_date = st.date_input("Enter the target date for the goal:")
            if st.button("Set Goal"):
                set_financial_goal(financial_data, goal_name, goal_amount, str(goal_target_date))
            
            # Additional style enhancements
            st.markdown("---")
            st.write("📊 Keep track of your financial progress! 📈")
        elif goals_choice == "View Financial Goals":
            view_financial_goals(financial_data)
        elif goals_choice == "Track Financial Goals":
            track_financial_goals(financial_data)

    
    # Add a footer
    footer = """
    ---
    *copyright@ahammadnafiz*
    """

    st.markdown(footer)
    
    # Footer content with links and emojis
    footer_content = """
        <div class="footer">
            Follow me: &nbsp;&nbsp;&nbsp;
            <a href="https://github.com/ahammadnafiz" target="_blank">GitHub</a> 🚀 |
            <a href="https://twitter.com/ahammadnafi_z" target="_blank">Twitter</a> 🐦
        </div>
    """

    # Display the footer
    st.markdown(footer_content, unsafe_allow_html=True)
        
    
if __name__ == "__main__":
    main()
