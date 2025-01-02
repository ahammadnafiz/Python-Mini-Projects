import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import altair as alt
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Define CSV file paths
TASKS_FILE = 'tasks.csv'
NOTES_FILE = 'notes.csv'
GOALS_FILE = 'goals.csv'

def init_session_state():
    if 'show_completed' not in st.session_state:
        st.session_state.show_completed = False
    if 'filter_priority' not in st.session_state:
        st.session_state.filter_priority = "All"
    if 'edit_task_index' not in st.session_state:
        st.session_state.edit_task_index = None

def save_data(df, file_path):
    try:
        df.to_csv(file_path, index=False)
        return True
    except Exception as e:
        st.error(f"Error saving data: {str(e)}")
        return False

def load_data():
    # Initialize empty DataFrames with correct columns
    tasks_columns = ['title', 'description', 'due_date', 'priority', 'status', 'created_at', 'recurring', 'subtasks', 'time_estimate', 'category', 'auto_adjust']
    notes_columns = ['title', 'content', 'created_at']
    goals_columns = ['title', 'description', 'target_date', 'status', 'created_at', 'completed_at']

    try:
        if os.path.exists(TASKS_FILE) and os.path.getsize(TASKS_FILE) > 0:
            tasks_df = pd.read_csv(TASKS_FILE)
            tasks_df['subtasks'] = tasks_df['subtasks'].apply(eval)  # Convert subtasks from string to list
        else:
            tasks_df = pd.DataFrame(columns=tasks_columns)
            save_data(tasks_df, TASKS_FILE)  # Create the file with headers if it doesn't exist

        if os.path.exists(NOTES_FILE) and os.path.getsize(NOTES_FILE) > 0:
            notes_df = pd.read_csv(NOTES_FILE)
        else:
            notes_df = pd.DataFrame(columns=notes_columns)
            save_data(notes_df, NOTES_FILE)  # Create the file with headers if it doesn't exist

        if os.path.exists(GOALS_FILE) and os.path.getsize(GOALS_FILE) > 0:
            goals_df = pd.read_csv(GOALS_FILE)
        else:
            goals_df = pd.DataFrame(columns=goals_columns)
            save_data(goals_df, GOALS_FILE)  # Create the file with headers if it doesn't exist

        return tasks_df, notes_df, goals_df

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame(columns=tasks_columns), pd.DataFrame(columns=notes_columns), pd.DataFrame(columns=goals_columns)

def delete_row(df, index):
    return df.drop(index)

def tasks_page(tasks_df):
    st.markdown(
                "<div style='text-align: center; margin-top: 25px; margin-bottom: 25px; font-size: 30px; '>📝 Tasks</div>",
                unsafe_allow_html=True,
            )

    control_container = st.container()
    with control_container:
        col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
        with col1:
            st.session_state.filter_priority = st.selectbox(
                "Filter by Priority",
                ["All", "High", "Medium", "Low"]
            )
        with col2:
            st.session_state.show_completed = st.checkbox("Completed")
        with col3:
            if st.button("+ New Task", use_container_width=True):
                st.session_state.show_task_form = True
                st.session_state.edit_task_index = None

    if st.session_state.get('show_task_form', False):
        with st.form("task_form", clear_on_submit=True):
            st.subheader("New Task" if st.session_state.edit_task_index is None else "Edit Task")
            task_title = st.text_input("Title", value=tasks_df.at[st.session_state.edit_task_index, 'title'] if st.session_state.edit_task_index is not None else "")
            col1, col2 = st.columns(2)
            with col1:
                due_date = st.date_input("Due Date", value=pd.to_datetime(tasks_df.at[st.session_state.edit_task_index, 'due_date']) if st.session_state.edit_task_index is not None else datetime.now())
            with col2:
                priority = st.selectbox("Priority", ["High", "Medium", "Low"], index=["High", "Medium", "Low"].index(tasks_df.at[st.session_state.edit_task_index, 'priority']) if st.session_state.edit_task_index is not None else 0)
            task_description = st.text_area("Description", value=tasks_df.at[st.session_state.edit_task_index, 'description'] if st.session_state.edit_task_index is not None else "")

            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                recurring = st.checkbox("Recurring", value=tasks_df.at[st.session_state.edit_task_index, 'recurring'] if st.session_state.edit_task_index is not None else False)
                if recurring:
                    recurring_interval = st.number_input("Recurring Interval (days)", min_value=1, value=tasks_df.at[st.session_state.edit_task_index, 'recurring'] if st.session_state.edit_task_index is not None else 1)
                else:
                    recurring_interval = None
            with col2:
                subtasks = st.text_area("Subtasks (comma-separated)", value=", ".join(tasks_df.at[st.session_state.edit_task_index, 'subtasks']) if st.session_state.edit_task_index is not None else "")
            with col3:
                time_estimate = st.number_input("Time Estimate (hours)", min_value=0, value=tasks_df.at[st.session_state.edit_task_index, 'time_estimate'] if st.session_state.edit_task_index is not None else 0)
                category = st.text_input("Category/Tags", value=tasks_df.at[st.session_state.edit_task_index, 'category'] if st.session_state.edit_task_index is not None else "")
                auto_adjust = st.checkbox("Auto-Adjust Priority", value=tasks_df.at[st.session_state.edit_task_index, 'auto_adjust'] if st.session_state.edit_task_index is not None else False)

            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                if st.form_submit_button("Save Task", use_container_width=True):
                    if task_title:
                        if st.session_state.edit_task_index is not None:
                            tasks_df.at[st.session_state.edit_task_index, 'title'] = task_title
                            tasks_df.at[st.session_state.edit_task_index, 'description'] = task_description
                            tasks_df.at[st.session_state.edit_task_index, 'due_date'] = due_date.strftime("%Y-%m-%d")
                            tasks_df.at[st.session_state.edit_task_index, 'priority'] = priority
                            tasks_df.at[st.session_state.edit_task_index, 'recurring'] = recurring_interval
                            tasks_df.at[st.session_state.edit_task_index, 'subtasks'] = [subtask.strip() for subtask in subtasks.split(',')] if subtasks else []
                            tasks_df.at[st.session_state.edit_task_index, 'time_estimate'] = time_estimate
                            tasks_df.at[st.session_state.edit_task_index, 'category'] = category
                            tasks_df.at[st.session_state.edit_task_index, 'auto_adjust'] = auto_adjust
                        else:
                            new_task = pd.DataFrame([{
                                'title': task_title,
                                'description': task_description,
                                'due_date': due_date.strftime("%Y-%m-%d"),
                                'priority': priority,
                                'status': "Pending",
                                'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'recurring': recurring_interval,
                                'subtasks': [subtask.strip() for subtask in subtasks.split(',')] if subtasks else [],
                                'time_estimate': time_estimate,
                                'category': category,
                                'auto_adjust': auto_adjust
                            }])
                            tasks_df = pd.concat([tasks_df, new_task], ignore_index=True)
                        save_data(tasks_df, TASKS_FILE)
                        st.session_state.show_task_form = False
                        st.session_state.edit_task_index = None
                        st.success("Task saved successfully!")
                        st.rerun()
            with col2:
                if st.form_submit_button("Cancel", use_container_width=True):
                    st.session_state.show_task_form = False
                    st.session_state.edit_task_index = None
                    st.rerun()

    # Filter tasks
    filtered_tasks = tasks_df.copy()
    if not st.session_state.show_completed:
        filtered_tasks = filtered_tasks[filtered_tasks['status'] != "Completed"]
    if st.session_state.filter_priority != "All":
        filtered_tasks = filtered_tasks[filtered_tasks['priority'] == st.session_state.filter_priority]

    # Display tasks
    for idx, task in filtered_tasks.iterrows():
        with st.container():
            col1, col2, col3 = st.columns([6, 3, 1])

            with col1:
                task_status = "✅ " if task['status'] == "Completed" else "🔲 "
                st.write(f"{task_status} **{task['title']}**")
                st.write(task['description'])
                if 'subtasks' in task and task['subtasks']:
                    st.write("Subtasks:")
                    for subtask in task['subtasks']:
                        st.write(f"- {subtask}")

            with col2:
                st.write(f"📅 {task['due_date']}")
                st.write(f"🎯 {task['priority']}")
                st.write(f"⏰ {task['time_estimate']} hours")
                st.write(f"🏷️ {task['category']}")

            with col3:
                button_col1, button_col2, button_col3 = st.columns(3)
                with button_col1:
                    if task['status'] != "Completed":
                        if st.button("✓", key=f"complete_{idx}", use_container_width=True):
                            tasks_df.at[idx, 'status'] = "Completed"
                            save_data(tasks_df, TASKS_FILE)
                            st.rerun()
                with button_col2:
                    if st.button("🗑", key=f"delete_{idx}", use_container_width=True):
                        tasks_df = delete_row(tasks_df, idx)
                        save_data(tasks_df, TASKS_FILE)
                        st.rerun()
                with button_col3:
                    if st.button("✏️", key=f"edit_{idx}", use_container_width=True):
                        st.session_state.show_task_form = True
                        st.session_state.edit_task_index = idx
                        st.rerun()

    return tasks_df

def notes_page(notes_df):
    st.markdown(
                "<div style='text-align: center; margin-top: 25px; margin-bottom: 25px; font-size: 30px; '>📓 Notes</div>",
                unsafe_allow_html=True,
            )

    if st.button("+ New Note"):
        st.session_state.show_note_form = True

    if st.session_state.get('show_note_form', False):
        with st.form("new_note_form", clear_on_submit=True):
            st.subheader("New Note")
            note_title = st.text_input("Title")
            note_content = st.text_area("Content")

            col1, col2 = st.columns(2)
            with col1:
                if st.form_submit_button("Save Note"):
                    if note_title and note_content:
                        new_note = pd.DataFrame([{
                            'title': note_title,
                            'content': note_content,
                            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }])
                        notes_df = pd.concat([notes_df, new_note], ignore_index=True)
                        save_data(notes_df, NOTES_FILE)
                        st.session_state.show_note_form = False
                        st.success("Note saved successfully!")
                        st.rerun()
            with col2:
                if st.form_submit_button("Cancel"):
                    st.session_state.show_note_form = False
                    st.rerun()

    for idx, note in notes_df.iterrows():
        with st.expander(note['title']):
            st.write(note['content'])
            st.caption(f"Created: {note['created_at']}")
            if st.button("Delete Note", key=f"delete_note_{idx}"):
                notes_df = delete_row(notes_df, idx)
                save_data(notes_df, NOTES_FILE)
                st.rerun()

    return notes_df

def goals_page(goals_df):
    st.markdown(
                "<div style='text-align: center; margin-top: 25px; margin-bottom: 25px; font-size: 30px; '>🎯 Goals</div>",
                unsafe_allow_html=True,
            )

    button_container = st.container()
    with button_container:
        col1, col2, col3 = st.columns([1, 4, 1])
        with col1:
            if st.button("+ New Goal", use_container_width=True):
                st.session_state.show_goal_form = True

    if st.session_state.get('show_goal_form', False):
        with st.form("new_goal_form", clear_on_submit=True):
            st.subheader("New Goal")
            goal_title = st.text_input("Title")
            col1, col2 = st.columns(2)
            with col1:
                target_date = st.date_input("Target Date")
            with col2:
                goal_status = st.selectbox("Initial Status", ["Not Started", "In Progress", "Completed", "Abandoned"])
            goal_description = st.text_area("Description")

            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                if st.form_submit_button("Set Goal", use_container_width=True):
                    if goal_title:
                        new_goal = pd.DataFrame([{
                            'title': goal_title,
                            'description': goal_description,
                            'target_date': target_date.strftime("%Y-%m-%d"),
                            'status': goal_status,
                            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'completed_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S") if goal_status == "Completed" else None
                        }])
                        goals_df = pd.concat([goals_df, new_goal], ignore_index=True)
                        save_data(goals_df, GOALS_FILE)
                        st.session_state.show_goal_form = False
                        st.success("Goal set successfully!")
                        st.rerun()
            with col2:
                if st.form_submit_button("Cancel", use_container_width=True):
                    st.session_state.show_goal_form = False
                    st.rerun()

    for idx, goal in goals_df.iterrows():
        with st.expander(goal['title']):
            st.write(goal['description'])
            st.write(f"Target Date: {goal['target_date']}")
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                previous_status = goal['status']
                new_status = st.selectbox(
                    "Status",
                    ["Not Started", "In Progress", "Completed", "Abandoned"],
                    index=["Not Started", "In Progress", "Completed", "Abandoned"].index(goal['status']),
                    key=f"goal_status_{idx}"
                )
                if new_status != previous_status:
                    goals_df.at[idx, 'status'] = new_status
                    if new_status == "Completed":
                        goals_df.at[idx, 'completed_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    elif previous_status == "Completed":
                        goals_df.at[idx, 'completed_at'] = None
                    save_data(goals_df, GOALS_FILE)
            with col3:
                if st.button("Delete", key=f"delete_goal_{idx}", use_container_width=True):
                    goals_df = delete_row(goals_df, idx)
                    save_data(goals_df, GOALS_FILE)
                    st.rerun()

    return goals_df

def create_goal_status_chart(df):
    status_counts = df['status'].value_counts().reset_index()
    status_counts.columns = ['status', 'count']

    return alt.Chart(status_counts).mark_arc().encode(
        theta=alt.Theta(field='count', type='quantitative'),
        color=alt.Color(field='status', type='nominal'),
        tooltip=['status', 'count']
    ).properties(
        title='Goal Status Distribution'
    )

def create_goal_timeline_chart(df):
    timeline_data = df[['target_date', 'status', 'title']].copy()
    timeline_data['target_date'] = pd.to_datetime(timeline_data['target_date'])

    return alt.Chart(timeline_data).mark_circle(size=100).encode(
        x=alt.X('target_date:T', title='Target Date'),
        y=alt.Y('status:N', title='Status'),
        color='status:N',
        tooltip=['title', 'status', 'target_date']
    ).properties(
        title='Goal Timeline'
    )

def create_goal_completion_trend(df):
    # First check if completed_at column exists and has any non-null values
    if 'completed_at' not in df.columns:
        return None

    completed_goals = df[df['completed_at'].notna()].copy()
    if not completed_goals.empty:
        completed_goals['completed_at'] = pd.to_datetime(completed_goals['completed_at'])
        daily_completions = completed_goals.groupby(completed_goals['completed_at'].dt.date).size().reset_index()
        daily_completions.columns = ['date', 'count']

        return alt.Chart(daily_completions).mark_line(point=True).encode(
            x=alt.X('date:T', title='Date'),
            y=alt.Y('count:Q', title='Completed Goals'),
            tooltip=['date', 'count']
        ).properties(
            title='Goal Completion Trend'
        )
    return None

def create_status_chart(df):
    status_counts = df['status'].value_counts().reset_index()
    status_counts.columns = ['status', 'count']

    return alt.Chart(status_counts).mark_arc().encode(
        theta=alt.Theta(field='count', type='quantitative'),
        color=alt.Color(field='status', type='nominal'),
        tooltip=['status', 'count']
    ).properties(
        title='Task Status Distribution'
    )

def create_priority_chart(df):
    priority_counts = df['priority'].value_counts().reset_index()
    priority_counts.columns = ['priority', 'count']

    return alt.Chart(priority_counts).mark_bar().encode(
        x=alt.X('priority:N', sort=['High', 'Medium', 'Low']),
        y='count:Q',
        color=alt.Color('priority:N'),
        tooltip=['priority', 'count']
    ).properties(
        title='Tasks by Priority'
    )

def create_timeline_chart(df):
    daily_tasks = df.groupby(df['created_at'].dt.date).size().reset_index()
    daily_tasks.columns = ['date', 'count']

    return alt.Chart(daily_tasks).mark_line(point=True).encode(
        x=alt.X('date:T', title='Date'),
        y=alt.Y('count:Q', title='Number of Tasks'),
        tooltip=['date', 'count']
    ).properties(
        title='Task Creation Timeline'
    )

def create_productivity_chart(df):
    completed_tasks = df[df['status'] == 'Completed'].copy()
    completed_tasks['day_of_week'] = completed_tasks['created_at'].dt.day_name()

    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_completion = completed_tasks['day_of_week'].value_counts().reindex(day_order).reset_index()
    daily_completion.columns = ['day', 'count']

    return alt.Chart(daily_completion).mark_bar().encode(
        x=alt.X('day:N', sort=day_order),
        y=alt.Y('count:Q', title='Completed Tasks'),
        tooltip=['day', 'count']
    ).properties(
        title='Task Completion by Day of Week'
    )

def create_burndown_chart(df):
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['due_date'] = pd.to_datetime(df['due_date'])
    df = df.sort_values(by='due_date')

    # Ensure we are applying cumsum on a numeric column
    df['cumulative_count'] = df.groupby('due_date').cumcount() + 1

    return alt.Chart(df).mark_line().encode(
        x=alt.X('due_date:T', title='Due Date'),
        y=alt.Y('cumulative_count:Q', title='Cumulative Tasks'),
        tooltip=['due_date', 'cumulative_count']
    ).properties(
        title='Burndown Chart'
    )

def create_velocity_chart(df):
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['due_date'] = pd.to_datetime(df['due_date'])
    df = df.sort_values(by='due_date')

    velocity_data = df.groupby('due_date').size().reset_index()
    velocity_data.columns = ['due_date', 'count']

    return alt.Chart(velocity_data).mark_line().encode(
        x=alt.X('due_date:T', title='Due Date'),
        y=alt.Y('count:Q', title='Tasks Completed'),
        tooltip=['due_date', 'count']
    ).properties(
        title='Velocity Chart'
    )

def create_progress_forecast_chart(df):
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['due_date'] = pd.to_datetime(df['due_date'])
    df = df.sort_values(by='due_date')

    forecast_data = df.groupby('due_date').size().reset_index()
    forecast_data.columns = ['due_date', 'count']

    return alt.Chart(forecast_data).mark_line().encode(
        x=alt.X('due_date:T', title='Due Date'),
        y=alt.Y('count:Q', title='Tasks Completed'),
        tooltip=['due_date', 'count']
    ).properties(
        title='Progress Forecast Chart'
    )

def create_category_chart(df):
    category_counts = df['category'].value_counts().reset_index()
    category_counts.columns = ['category', 'count']

    return alt.Chart(category_counts).mark_bar().encode(
        x=alt.X('category:N', sort='-y'),
        y='count:Q',
        color=alt.Color('category:N'),
        tooltip=['category', 'count']
    ).properties(
        title='Tasks by Category'
    )

def create_time_estimate_chart(df):
    time_estimate_counts = df['time_estimate'].value_counts().reset_index()
    time_estimate_counts.columns = ['time_estimate', 'count']

    return alt.Chart(time_estimate_counts).mark_bar().encode(
        x=alt.X('time_estimate:Q', bin=True),
        y='count:Q',
        tooltip=['time_estimate', 'count']
    ).properties(
        title='Tasks by Time Estimate'
    )

def train_task_completion_model(df):
    # Check if we have enough data to train
    if len(df) < 5:  # Arbitrary minimum threshold
        st.warning("Not enough data to train the model. Need at least 5 tasks.")
        return None

    # Feature engineering
    df['due_date'] = pd.to_datetime(df['due_date'])
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['days_to_due'] = (df['due_date'] - df['created_at']).dt.days
    df['is_completed'] = df['status'].apply(lambda x: 1 if x == 'Completed' else 0)
    
    # Convert priority to numeric
    priority_map = {'High': 3, 'Medium': 2, 'Low': 1}
    df['priority_numeric'] = df['priority'].map(priority_map)

    # Select features and target
    features = ['priority_numeric', 'time_estimate', 'days_to_due']
    X = df[features].fillna(0)  # Handle any missing values
    y = df['is_completed']

    # Adjust test size based on dataset size
    test_size = min(0.2, 1/len(df))
    
    # Train the model
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Task Completion Model Accuracy: {accuracy:.2f}")
        
        return model
    except Exception as e:
        st.warning(f"Could not train model: {str(e)}")
        return None

def predict_task_completions(model, df):
    if model is None:
        return df
        
    # Feature engineering
    df['due_date'] = pd.to_datetime(df['due_date'])
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['days_to_due'] = (df['due_date'] - df['created_at']).dt.days
    
    # Convert priority to numeric
    priority_map = {'High': 3, 'Medium': 2, 'Low': 1}
    df['priority_numeric'] = df['priority'].map(priority_map)

    # Select features
    features = ['priority_numeric', 'time_estimate', 'days_to_due']
    X = df[features].fillna(0)  # Handle any missing values

    try:
        # Make predictions
        predictions = model.predict(X)
        df['predicted_completion'] = predictions
        df['predicted_completion'] = df['predicted_completion'].map({1: 'Likely to complete', 0: 'May need attention'})
    except Exception as e:
        st.warning(f"Could not make predictions: {str(e)}")
        df['predicted_completion'] = 'Unable to predict'

    return df

def train_goal_achievement_model(df):
    # Check if we have enough data to train
    if len(df) < 5:  # Arbitrary minimum threshold
        st.warning("Not enough data to train the model. Need at least 5 goals.")
        return None

    # Feature engineering
    df['target_date'] = pd.to_datetime(df['target_date'])
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['days_to_target'] = (df['target_date'] - df['created_at']).dt.days
    df['is_completed'] = df['status'].apply(lambda x: 1 if x == 'Completed' else 0)

    # Select features and target
    features = ['days_to_target']
    X = df[features].fillna(0)  # Handle any missing values
    y = df['is_completed']

    # Adjust test size based on dataset size
    test_size = min(0.2, 1/len(df))

    try:
        # Train the model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Goal Achievement Model Accuracy: {accuracy:.2f}")
        
        return model
    except Exception as e:
        st.warning(f"Could not train model: {str(e)}")
        return None

def predict_goal_achievements(model, df):
    if model is None:
        return df
        
    # Feature engineering
    df['target_date'] = pd.to_datetime(df['target_date'])
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['days_to_target'] = (df['target_date'] - df['created_at']).dt.days

    # Select features
    features = ['days_to_target']
    X = df[features].fillna(0)  # Handle any missing values

    try:
        # Make predictions
        predictions = model.predict(X)
        df['predicted_achievement'] = predictions
        df['predicted_achievement'] = df['predicted_achievement'].map({1: 'Likely to achieve', 0: 'May need support'})
    except Exception as e:
        st.warning(f"Could not make predictions: {str(e)}")
        df['predicted_achievement'] = 'Unable to predict'

    return df

def analytics_page(tasks_df, goals_df):
    st.markdown(
                "<div style='text-align: center; margin-top: 25px; margin-bottom: 25px; font-size: 30px; '>📊 Analytics Dashboard</div>",
                unsafe_allow_html=True,
            )

    # Date Range Selection
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        preset = st.selectbox(
            "Date Range",
            ["Last 7 Days", "Last 30 Days", "Last 90 Days", "Custom"],
            index=1
        )

    end_date = datetime.now().date()
    if preset == "Last 7 Days":
        start_date = end_date - timedelta(days=7)
    elif preset == "Last 30 Days":
        start_date = end_date - timedelta(days=30)
    elif preset == "Last 90 Days":
        start_date = end_date - timedelta(days=90)
    else:
        with col2:
            start_date = st.date_input("Start Date", value=end_date - timedelta(days=30))
        with col3:
            end_date = st.date_input("End Date", value=end_date)

    # Task Analysis Section
    if not tasks_df.empty:
        # Convert date columns to datetime
        tasks_df['due_date'] = pd.to_datetime(tasks_df['due_date'])
        tasks_df['created_at'] = pd.to_datetime(tasks_df['created_at'])

        mask = (tasks_df['created_at'].dt.date >= start_date) & (tasks_df['created_at'].dt.date <= end_date)
        filtered_tasks = tasks_df[mask]

        st.subheader("📈 Task Metrics")
        m1, m2, m3, m4 = st.columns(4)

        total_tasks = len(filtered_tasks)
        completed_tasks = len(filtered_tasks[filtered_tasks['status'] == 'Completed'])
        completion_rate = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        overdue_tasks = len(filtered_tasks[
            (filtered_tasks['status'] != 'Completed') &
            (filtered_tasks['due_date'].dt.date < datetime.now().date())
        ])

        m1.metric("Total Tasks", total_tasks)
        m2.metric("Completion Rate", f"{completion_rate:.1f}%")
        m3.metric("High Priority", len(filtered_tasks[filtered_tasks['priority'] == 'High']))
        m4.metric("Overdue", overdue_tasks,
                delta=None if overdue_tasks == 0 else "needs attention",
                delta_color="inverse")

        st.subheader("📊 Task Analysis")
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Status & Priority", "Timeline", "Productivity", "Burndown", "Velocity", "Category & Time", "Predictions"])

        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.altair_chart(create_status_chart(filtered_tasks), use_container_width=True)
            with col2:
                st.altair_chart(create_priority_chart(filtered_tasks), use_container_width=True)

        with tab2:
            st.altair_chart(create_timeline_chart(filtered_tasks), use_container_width=True)

        with tab3:
            st.altair_chart(create_productivity_chart(filtered_tasks), use_container_width=True)

        with tab4:
            st.altair_chart(create_burndown_chart(filtered_tasks), use_container_width=True)

        with tab5:
            st.altair_chart(create_velocity_chart(filtered_tasks), use_container_width=True)

        with tab6:
            col1, col2 = st.columns(2)
            with col1:
                st.altair_chart(create_category_chart(filtered_tasks), use_container_width=True)
            with col2:
                st.altair_chart(create_time_estimate_chart(filtered_tasks), use_container_width=True)

        with tab7:
            st.subheader("Future Predictions")
            task_model = train_task_completion_model(filtered_tasks)
            if task_model is not None:
                predicted_tasks = predict_task_completions(task_model, filtered_tasks)
                display_cols = ['title']
                if 'predicted_completion' in predicted_tasks.columns:
                    display_cols.append('predicted_completion')
                st.write(predicted_tasks[display_cols])
            else:
                st.info("Add more tasks to see completion predictions")

    # Goals Analysis Section
    if not goals_df.empty:
        # Convert date columns to datetime
        goals_df['target_date'] = pd.to_datetime(goals_df['target_date'])
        goals_df['created_at'] = pd.to_datetime(goals_df['created_at'])
        if 'completed_at' in goals_df.columns:
            goals_df['completed_at'] = pd.to_datetime(goals_df['completed_at'])

        # Filter goals based on date range
        goals_mask = (goals_df['created_at'].dt.date >= start_date) & (goals_df['created_at'].dt.date <= end_date)
        filtered_goals = goals_df[goals_mask]

        st.subheader("🎯 Goal Metrics")
        g1, g2, g3, g4 = st.columns(4)

        total_goals = len(filtered_goals)
        completed_goals = len(filtered_goals[filtered_goals['status'] == 'Completed'])
        in_progress_goals = len(filtered_goals[filtered_goals['status'] == 'In Progress'])
        goal_completion_rate = (completed_goals / total_goals * 100) if total_goals > 0 else 0

        g1.metric("Total Goals", total_goals)
        g2.metric("Completion Rate", f"{goal_completion_rate:.1f}%")
        g3.metric("In Progress", in_progress_goals)
        g4.metric("Completed", completed_goals)

        st.subheader("📈 Goal Analysis")
        goal_tab1, goal_tab2, goal_tab3, goal_tab4 = st.tabs(["Status Overview", "Timeline", "Completion Trend", "Predictions"])

        with goal_tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.altair_chart(create_goal_status_chart(filtered_goals), use_container_width=True)
            with col2:
                # Calculate and display goal achievement rate over time
                if 'completed_at' in filtered_goals.columns:
                    completed_over_time = filtered_goals[filtered_goals['completed_at'].notna()].copy()
                    if not completed_over_time.empty:
                        completed_over_time['completion_month'] = completed_over_time['completed_at'].dt.to_period('M')
                        monthly_completion = completed_over_time.groupby('completion_month').size().reset_index()
                        monthly_completion.columns = ['month', 'completed']

                        monthly_chart = alt.Chart(monthly_completion).mark_bar().encode(
                            x='month:T',
                            y='completed:Q',
                            tooltip=['month', 'completed']
                        ).properties(title='Monthly Goal Completion')

                        st.altair_chart(monthly_chart, use_container_width=True)
                    else:
                        st.info("No completed goals data available yet.")

        with goal_tab2:
            st.altair_chart(create_goal_timeline_chart(filtered_goals), use_container_width=True)

        with goal_tab3:
            trend_chart = create_goal_completion_trend(filtered_goals)
            if trend_chart:
                st.altair_chart(trend_chart, use_container_width=True)
            else:
                st.info("No goal completion trend data available yet.")

        with goal_tab4:
            st.subheader("Future Predictions")
            goal_model = train_goal_achievement_model(filtered_goals)
            if goal_model is not None:
                predicted_goals = predict_goal_achievements(goal_model, filtered_goals)
                display_cols = ['title']
                if 'predicted_achievement' in predicted_goals.columns:
                    display_cols.append('predicted_achievement')
                st.write(predicted_goals[display_cols])
            else:
                st.info("Add more goals to see achievement predictions")

    # Show message if no data is available
    if tasks_df.empty and goals_df.empty:
        st.info("No data available for analysis. Start by adding some tasks or goals!")

def main():
    st.set_page_config(page_title="Basto", page_icon="📝")
    init_session_state()
    st.image('basto.png')

    # Load data from CSV files
    tasks_df, notes_df, goals_df = load_data()

    # Navigation sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Tasks", "Notes", "Goals", "Analytics"])

    # Page routing with DataFrame updates
    if page == "Tasks":
        tasks_df = tasks_page(tasks_df)
    elif page == "Notes":
        notes_df = notes_page(notes_df)
    elif page == "Goals":
        goals_df = goals_page(goals_df)
    else:
        analytics_page(tasks_df, goals_df)  # Pass the DataFrames to analytics page

if __name__ == "__main__":
    main()
