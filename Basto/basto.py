import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import altair as alt
import json
import os

def init_session_state():
    if 'tasks' not in st.session_state:
        st.session_state.tasks = []
    if 'notes' not in st.session_state:
        st.session_state.notes = []
    if 'goals' not in st.session_state:
        st.session_state.goals = []
    if 'show_completed' not in st.session_state:
        st.session_state.show_completed = False
    if 'filter_priority' not in st.session_state:
        st.session_state.filter_priority = "All"

def save_data():
    try:
        data = {
            'tasks': st.session_state.tasks,
            'notes': st.session_state.notes,
            'goals': st.session_state.goals
        }
        with open('basto_data.json', 'w') as f:
            json.dump(data, f)
        return True
    except Exception as e:
        st.error(f"Error saving data: {str(e)}")
        return False

def load_data():
    try:
        if os.path.exists('notion_mini_data.json'):
            with open('notion_mini_data.json', 'r') as f:
                data = json.load(f)
                st.session_state.tasks = data.get('tasks', [])
                st.session_state.notes = data.get('notes', [])
                st.session_state.goals = data.get('goals', [])
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")

def delete_item(item_list, index):
    if 0 <= index < len(item_list):
        item_list.pop(index)
        save_data()

def tasks_page():
    st.title("📝 Tasks")
    
    # Create a container for consistent button and filter layout
    control_container = st.container()
    with control_container:
        col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
        with col1:
            st.session_state.filter_priority = st.selectbox(
                "Filter by Priority",
                ["All", "High", "Medium", "Low"]
            )
        with col2:
            st.session_state.show_completed = st.checkbox("Show Completed Tasks")
        with col3:
            if st.button("+ New Task", use_container_width=True):
                st.session_state.show_task_form = True

    if st.session_state.get('show_task_form', False):
        with st.form("new_task_form", clear_on_submit=True):
            st.subheader("New Task")
            task_title = st.text_input("Title")
            col1, col2 = st.columns(2)
            with col1:
                due_date = st.date_input("Due Date")
            with col2:
                priority = st.selectbox("Priority", ["High", "Medium", "Low"])
            task_description = st.text_area("Description")
            
            # Align form buttons consistently
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                if st.form_submit_button("Add Task", use_container_width=True):
                    if task_title:
                        new_task = {
                            'title': task_title,
                            'description': task_description,
                            'due_date': due_date.strftime("%Y-%m-%d"),
                            'priority': priority,
                            'status': "Pending",
                            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        st.session_state.tasks.append(new_task)
                        save_data()
                        st.session_state.show_task_form = False
                        st.success("Task added successfully!")
                        st.rerun()
            with col2:
                if st.form_submit_button("Cancel", use_container_width=True):
                    st.session_state.show_task_form = False
                    st.rerun()

    # Filter tasks
    filtered_tasks = st.session_state.tasks.copy()
    if not st.session_state.show_completed:
        filtered_tasks = [t for t in filtered_tasks if t['status'] != "Completed"]
    if st.session_state.filter_priority != "All":
        filtered_tasks = [t for t in filtered_tasks if t['priority'] == st.session_state.filter_priority]

    # Display tasks with consistent layout
    for idx, task in enumerate(filtered_tasks):
        with st.container():
            col1, col2, col3 = st.columns([6, 3, 1])
            
            with col1:
                task_status = "✅ " if task['status'] == "Completed" else "🔲 "
                st.write(f"{task_status} **{task['title']}**")
                st.write(task['description'])
            
            with col2:
                st.write(f"📅 {task['due_date']}")
                st.write(f"🎯 {task['priority']}")
            
            with col3:
                button_col1, button_col2 = st.columns(2)
                with button_col1:
                    if task['status'] != "Completed":
                        if st.button("✓", key=f"complete_{idx}", use_container_width=True):
                            task['status'] = "Completed"
                            save_data()
                            st.rerun()
                with button_col2:
                    if st.button("🗑", key=f"delete_{idx}", use_container_width=True):
                        delete_item(st.session_state.tasks, idx)
                        st.rerun()

def notes_page():
    st.title("📓 Notes")
    
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
                        new_note = {
                            'title': note_title,
                            'content': note_content,
                            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        st.session_state.notes.append(new_note)
                        save_data()
                        st.session_state.show_note_form = False
                        st.success("Note saved successfully!")
                        st.rerun()
            with col2:
                if st.form_submit_button("Cancel"):
                    st.session_state.show_note_form = False
                    st.rerun()

    for idx, note in enumerate(st.session_state.notes):
        with st.expander(note['title']):
            st.write(note['content'])
            st.caption(f"Created: {note['created_at']}")
            if st.button("Delete Note", key=f"delete_note_{idx}"):
                delete_item(st.session_state.notes, idx)
                st.rerun()

def goals_page():
    st.title("🎯 Goals")
    
    # Create a container for consistent button alignment
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
            
            # Align form buttons
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                if st.form_submit_button("Set Goal", use_container_width=True):
                    if goal_title:
                        new_goal = {
                            'title': goal_title,
                            'description': goal_description,
                            'target_date': target_date.strftime("%Y-%m-%d"),
                            'status': goal_status,
                            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'completed_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S") if goal_status == "Completed" else None
                        }
                        st.session_state.goals.append(new_goal)
                        save_data()
                        st.session_state.show_goal_form = False
                        st.success("Goal set successfully!")
                        st.rerun()
            with col2:
                if st.form_submit_button("Cancel", use_container_width=True):
                    st.session_state.show_goal_form = False
                    st.rerun()

    # Goals list with consistent layout
    for idx, goal in enumerate(st.session_state.goals):
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
                    goal['status'] = new_status
                    # Update completed_at timestamp when status changes to or from Completed
                    if new_status == "Completed":
                        goal['completed_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    elif previous_status == "Completed":
                        goal['completed_at'] = None
                    save_data()
            with col3:
                if st.button("Delete", key=f"delete_goal_{idx}", use_container_width=True):
                    delete_item(st.session_state.goals, idx)
                    st.rerun()

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

def analytics_page():
    st.title("📊 Analytics Dashboard")
    
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
    tasks_df = pd.DataFrame(st.session_state.tasks)
    
    if not tasks_df.empty:
        tasks_df['due_date'] = pd.to_datetime(tasks_df['due_date'])
        tasks_df['created_at'] = pd.to_datetime(tasks_df['created_at'])
        
        mask = (tasks_df['created_at'].dt.date >= start_date) & (tasks_df['created_at'].dt.date <= end_date)
        filtered_tasks = tasks_df[mask]
        
        st.header("📈 Task Metrics")
        m1, m2, m3, m4 = st.columns(4)
        
        total_tasks = len(filtered_tasks)
        completed_tasks = len(filtered_tasks[filtered_tasks['status'] == 'Completed'])
        completion_rate = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        overdue_tasks = len(filtered_tasks[
            (filtered_tasks['status'] != 'Completed') & 
            (filtered_tasks['due_date'] < datetime.now())
        ])
        
        m1.metric("Total Tasks", total_tasks)
        m2.metric("Completion Rate", f"{completion_rate:.1f}%")
        m3.metric("High Priority", len(filtered_tasks[filtered_tasks['priority'] == 'High']))
        m4.metric("Overdue", overdue_tasks, 
                delta=None if overdue_tasks == 0 else "needs attention",
                delta_color="inverse")

        st.header("📊 Task Analysis")
        tab1, tab2, tab3 = st.tabs(["Status & Priority", "Timeline", "Productivity"])
        
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

    # Goals Analysis Section
    goals_df = pd.DataFrame(st.session_state.goals)
    
    if not goals_df.empty:
        goals_df['target_date'] = pd.to_datetime(goals_df['target_date'])
        goals_df['created_at'] = pd.to_datetime(goals_df['created_at'])
        if 'completed_at' in goals_df.columns:
            goals_df['completed_at'] = pd.to_datetime(goals_df['completed_at'])
        
        # Filter goals based on date range
        goals_mask = (goals_df['created_at'].dt.date >= start_date) & (goals_df['created_at'].dt.date <= end_date)
        filtered_goals = goals_df[goals_mask]
        
        st.header("🎯 Goal Metrics")
        g1, g2, g3, g4 = st.columns(4)
        
        total_goals = len(filtered_goals)
        completed_goals = len(filtered_goals[filtered_goals['status'] == 'Completed'])
        in_progress_goals = len(filtered_goals[filtered_goals['status'] == 'In Progress'])
        goal_completion_rate = (completed_goals / total_goals * 100) if total_goals > 0 else 0
        
        g1.metric("Total Goals", total_goals)
        g2.metric("Completion Rate", f"{goal_completion_rate:.1f}%")
        g3.metric("In Progress", in_progress_goals)
        g4.metric("Completed", completed_goals)

        st.header("📈 Goal Analysis")
        goal_tab1, goal_tab2, goal_tab3 = st.tabs(["Status Overview", "Timeline", "Completion Trend"])
        
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

    # Show message if no data is available
    if tasks_df.empty and goals_df.empty:
        st.info("No data available for analysis. Start by adding some tasks or goals!")

def main():
    st.set_page_config(page_title="Notion Mini", page_icon="📝")
    init_session_state()
    # load_css()
    load_data()
    
    # Navigation sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Tasks", "Notes", "Goals", "Analytics"])
    
    # Page routing
    if page == "Tasks":
        tasks_page()
    elif page == "Notes":
        notes_page()
    elif page == "Goals":
        goals_page()
    else:
        analytics_page()

if __name__ == "__main__":
    main()