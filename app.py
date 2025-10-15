import streamlit as st
import requests
import pandas as pd
import json
from io import StringIO
import os
import altair as alt
import time 
import plotly.graph_objects as go 
import uuid
# --------------------------
# Backend URL
# --------------------------
BACKEND_URL = "http://127.0.0.1:8000"  # Match the backend server address

# --------------------------
# Session State Management
# --------------------------
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.auth_token = None
    st.session_state.user_email = None
    st.session_state.user_role = 'user' # New: Store user role
    st.session_state.last_result = None
    st.session_state.admin_history_filter_user = 'All' # New: For filtering history in admin view

# ===============================
# Custom CSS (Enhanced Dark Theme)
# ===============================
def set_custom_css():
    st.markdown(
        """
        <style>
        /* Main App Background (Deep Dark Blue/Purple Gradient) */
        .stApp {
            background: linear-gradient(135deg, #0f0c29, #1b0042, #000000);
            background-attachment: fixed;
            color: #E0E0E0;
            font-family: 'Inter', sans-serif;
        }
        
        /* Headers - Centered */
        h1, h2, h3, h4, h5 {
            color: #C280FF; /* Soft Violet */
            text-align: center;
        }

        /* Streamlit Widgets */
        .stTextInput > div > div > input, 
        .stTextArea > div > div > textarea,
        .stFileUploader,
        .stSelectbox > div > div,
        .stRadio > label,
        .stCheckbox > label {
            background-color: #2D245A; /* Darker Purple for input fields */
            color: #E0E0E0;
            border: 1px solid #C280FF40; /* Subtle border */
            border-radius: 8px;
            padding: 8px 12px;
        }

        /* Buttons */
        .stButton button {
            background-color: #6A1B9A; /* Dark Magenta/Purple */
            color: white;
            border-radius: 12px;
            padding: 10px 20px;
            border: none;
            font-weight: bold;
            transition: all 0.2s;
        }
        .stButton button:hover {
            background-color: #8E24AA; /* Lighter hover color */
            box-shadow: 0 4px 15px rgba(194, 128, 255, 0.4);
            transform: translateY(-2px);
        }
        
        /* Card-style containers */
        .main-card-wrapper, .card-wrapper {
            background-color: rgba(30, 0, 60, 0.6); /* Transparent Deep Purple */
            border-radius: 15px;
            padding: 20px;
            margin-top: 15px;
            box-shadow: 0 8px 20px rgba(0, 8px, 0, 0.5);
            border: 1px solid #C280FF20;
        }

        /* Sidebar Customization */
        .st-emotion-cache-1ltf5a4 { /* Target sidebar main container */
            background-color: #1a0a38; /* Very dark purple sidebar */
            border-right: 2px solid #6A1B9A;
        }

        /* Dataframes */
        .stDataFrame {
            border: 1px solid #6A1B9A;
            border-radius: 8px;
            overflow: hidden;
        }
        
        /* Info/Success Boxes */
        .stAlert {
            border-radius: 8px;
            background-color: #2D245A !important;
            border-color: #C280FF !important;
            color: #E0E0E0 !important;
        }
        
        /* Metrics */
        [data-testid="stMetric"] {
            background-color: rgba(30, 0, 60, 0.8);
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #C280FF20;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
        }
        
        /* Tab titles to use accent color */
        [data-testid="stTab"] {
            color: #C280FF !important; 
        }
        
        </style>
        """
        , unsafe_allow_html=True)

# Helper functions for card styling
def main_card_wrapper(title=""):
    st.markdown('<div class="main-card-wrapper">', unsafe_allow_html=True)
    if title:
        st.subheader(title)

def close_card_wrapper():
    st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# API Calls
# ===============================

def get_headers():
    """Returns headers including the authorization token if available."""
    headers = {"Content-Type": "application/json"}
    if st.session_state.auth_token:
        headers["Authorization"] = f"Bearer {st.session_state.auth_token}"
    return headers

def api_call(method, endpoint, json_data=None, files=None):
    """Generic API caller with token handling."""
    url = f"{BACKEND_URL}{endpoint}"
    try:
        # Special handling for file uploads (must not set Content-Type in headers if files are present)
        current_headers = get_headers()
        if files:
            # Pop 'Content-Type' so requests can set the correct boundary header
            current_headers.pop("Content-Type", None) 
            
        if method == "POST":
            response = requests.post(url, headers=current_headers, json=json_data, files=files)
        elif method == "GET":
            response = requests.get(url, headers=current_headers)
        elif method == "DELETE":
            response = requests.delete(url, headers=current_headers)
        # Added support for PUT
        elif method == "PUT":
            response = requests.put(url, headers=current_headers, json=json_data)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
            
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            
        return response.json()
        
    except requests.exceptions.HTTPError as e:
        # response should be available here if HTTPError occurred
        st.error(f"API Error ({e.response.status_code}): {e.response.text}")
        if e.response.status_code in [401, 403]:
            # For unauthorized/expired tokens, force logout
            logout(auto=True)
        return None
    except requests.exceptions.ConnectionError:
        st.error("Connection Error: Could not connect to the backend server. Ensure it is running.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

def submit_feedback(timestamp: str, score: int, comment: str):
    """Submits feedback to the backend."""
    data = {
        "timestamp": timestamp,
        "score": score,
        "comment": comment
    }
    response = api_call("POST", "/feedback", json_data=data)
    return response

def update_curated_content(timestamp: str, user_email: str, new_output: str):
    """Updates a history entry's generated_output via admin endpoint."""
    data = {
        "user_email": user_email,
        "generated_output": new_output
    }
    # Note the PUT method
    response = api_call("PUT", f"/admin/content/{timestamp}", json_data=data)
    return response

def delete_curated_content(timestamp: str, user_email: str):
    """Deletes a history entry via admin endpoint."""
    # user_email is passed as a query parameter for DELETE
    response = api_call("DELETE", f"/admin/content/{timestamp}?user_email={user_email}")
    return response

# ===============================
# Auth & Session Functions
# ===============================

def logout(auto=False):
    """Clears session state and resets the login screen."""
    if auto:
        st.warning("Your session has expired or you are unauthorized. Please log in again.")
    st.session_state.logged_in = False
    st.session_state.auth_token = None
    st.session_state.user_email = None
    st.session_state.user_role = 'user'
    st.session_state.last_result = None
    st.session_state.admin_history_filter_user = 'All'
    st.rerun() 

def validate_token_and_set_session(token):
    """Validates the token and sets session state variables, including role."""
    st.session_state.auth_token = token
    try:
        response = api_call("GET", "/validate_token")
        if response and response.get('email'):
            st.session_state.logged_in = True
            st.session_state.user_email = response['email']
            st.session_state.user_role = response.get('role', 'user') # Get role from validation
            st.session_state.last_result = None # Clear any previous results on new login/validation
            return True
        else:
            logout(auto=True)
            return False
    except Exception as e:
        st.error(f"Token validation failed: {e}")
        logout(auto=True)
        return False

# ===============================
# UI Render Functions
# ===============================

def render_login_page():
    main_card_wrapper("Welcome to Multi-Task NLP Platform")
    
    auth_mode = st.radio("Select Mode", ("Login", "Register", "Forgot Password"))
    
    with st.form("auth_form"):
        email = st.text_input("Email")
        if auth_mode in ["Login", "Register"]:
            password = st.text_input("Password", type="password")
        
        submitted = st.form_submit_button(auth_mode)
        
        if submitted:
            if auth_mode == "Login":
                data = {"email": email, "password": password}
                response = api_call("POST", "/login", json_data=data)
                if response:
                    validate_token_and_set_session(response['access_token'])
                    st.success(f"Successfully logged in as {st.session_state.user_email}!")
                    time.sleep(1)
                    st.rerun() 
            
            elif auth_mode == "Register":
                data = {"email": email, "password": password}
                response = api_call("POST", "/register", json_data=data)
                if response:
                    validate_token_and_set_session(response['access_token'])
                    st.success("Registration successful! You are now logged in.")
                    time.sleep(1)
                    st.rerun() 
                    
            elif auth_mode == "Forgot Password":
                data = {"email": email}
                response = api_call("POST", "/forgot_password", json_data=data)
                if response:
                    st.info(response['message'])
    close_card_wrapper()

def render_password_reset_page(token):
    main_card_wrapper("Password Reset")
    st.warning("Please set a new password for your account.")
    
    with st.form("reset_form"):
        new_password = st.text_input("New Password", type="password")
        submitted = st.form_submit_button("Reset Password")
        
        if submitted:
            data = {"token": token, "new_password": new_password}
            response = api_call("POST", "/reset_password", json_data=data)
            if response:
                st.success(response['message'] + " Redirecting to login...")
                time.sleep(2)
                st.query_params.clear()
                st.rerun() 
    close_card_wrapper()

def render_spider_chart(df: pd.DataFrame, key_suffix=""):
    """Renders a Plotly radar/spider chart for evaluation metrics."""
    
    if df.empty:
        st.warning("No metrics data to display for the chart.")
        return

    categories = df['name'].tolist()
    values = df['value'].tolist()
    
    fig = go.Figure(
        data=[
            go.Scatterpolar(
                r=values + [values[0]], # Close the loop
                theta=categories + [categories[0]],
                fill='toself',
                name='Generated Output Scores',
                line_color='#C280FF',
                fillcolor='rgba(194, 128, 255, 0.4)'
            )
        ],
        layout=go.Layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(df['max_value'].tolist())] 
                )
            ),
            showlegend=False,
            template="plotly_dark",
            title=dict(
                text="Generated Output Evaluation Metrics",
                font=dict(color='#E0E0E0')
            ),
            plot_bgcolor='rgba(30, 0, 60, 0.8)',
            paper_bgcolor='rgba(30, 0, 60, 0.8)'
        )
    )

    st.plotly_chart(fig, use_container_width=True, key=f"spider_chart_{key_suffix}")

def render_feedback_section(result: dict):
    """Renders the Thumbs Up/Down and comment box."""
    if not result or result.get('timestamp') is None:
        return

    st.markdown("---")
    st.subheader("Your Feedback")
    st.info("Please rate the quality of this generated result to help improve the model.")
    
    col1, col2, col3 = st.columns([1, 1, 4])
    
    if st.session_state.get('feedback_submitted') == result['timestamp']:
        st.success("Thank you! Your feedback has been recorded.")
        return

    if 'feedback_score' not in st.session_state:
        st.session_state.feedback_score = None

    with col1:
        if st.button("👍 Thumbs Up", use_container_width=True, key="btn_up"):
            st.session_state.feedback_score = 1
    
    with col2:
        if st.button("👎 Thumbs Down", use_container_width=True, key="btn_down"):
            st.session_state.feedback_score = -1

    if st.session_state.get('feedback_score') in [1, -1]:
        st.markdown(f"**Selected:** {'👍 Good Result' if st.session_state.feedback_score == 1 else '👎 Needs Improvement'}")
        
        comment = st.text_area("Optional Comment (Max 200 chars)", key="feedback_comment", max_chars=200)
        
        if st.button("Submit Feedback", key="btn_submit_feedback"):
            with st.spinner("Submitting feedback..."):
                response = submit_feedback(
                    timestamp=result['timestamp'],
                    score=st.session_state.feedback_score,
                    comment=comment
                )
                if response:
                    st.session_state.feedback_submitted = result['timestamp'] 
                    st.success("Feedback submitted successfully! Thank you.")
                    st.session_state.feedback_score = None 
                    time.sleep(1)
                    st.rerun() 
                else:
                    st.error("Failed to submit feedback. Please try again.")


def render_task_page(task_name: str):
    main_card_wrapper(f"{task_name} Task")
    
    with st.form(f"{task_name.lower()}_form"):
        text = st.text_area("Enter Text to Process", height=200, key=f"{task_name.lower()}_text")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            min_len = st.number_input("Min Length", min_value=10, max_value=300, value=30)
        with col2:
            max_len = st.number_input("Max Length", min_value=10, max_value=500, value=150)
        with col3:
            language = st.selectbox("Translate Output to", ["Hindi", "Telugu", "English"], index=0)
        with col4:
            ref_text = st.text_area("Manual Reference Text (Optional)", placeholder="Enter ground truth text here...", height=70)

        submitted = st.form_submit_button(f"Run {task_name}")

    if submitted and text:
        with st.spinner(f"Processing text for {task_name.lower()}..."):
            data = {
                "text": text,
                "model": "BART" if task_name == "Summarization" else "PEGASUS",
                "language": language,
                "task": task_name,
                "reference_text": ref_text if ref_text else None,
                "min_length": min_len,
                "max_length": max_len
            }
            
            result = api_call("POST", "/process_task", json_data=data)
            if result:
                st.session_state.last_result = result
                st.session_state.feedback_submitted = None 
                st.rerun() 

    if st.session_state.last_result:
        result = st.session_state.last_result
        
        st.markdown("---")
        st.subheader("Task Result")

        st.markdown("##### Performance and Evaluation")
        colA, colB, colC, colD = st.columns(4)
        colA.metric("Time Taken (s)", f"{result['elapsed_time']:.2f}")
        colB.metric("ROUGE-L Score", f"{result['rouge_metrics']['rouge-l']:.4f}")
        colC.metric("Readability (Flesch)", f"{result['readability_metrics']['flesch_reading_ease']:.2f}")
        colD.metric("Reference Status", result['reference_status'].replace('_', ' ').title())

        st.markdown("##### Generated Output")
        st.markdown(f"**Model:** `{result['model_name']}`")
        st.info(result['generated_output'])
        
        st.markdown(f"##### Translated Output ({result['translated_output'].split(':')[0]})")
        st.success(result['translated_output'])

        if result['reference_text']:
            st.markdown("##### Reference Text (Ground Truth)")
            st.code(result['reference_text'], language='markdown')
        
        st.markdown("---")
        st.subheader("Detailed Evaluation Visualization")

        spider_df = pd.DataFrame(result['spider_chart_data'])
        
        col_chart, col_metrics = st.columns([2, 1])
        
        with col_chart:
            render_spider_chart(spider_df)
        
        with col_metrics:
            st.markdown("###### All ROUGE Scores")
            st.json(result['rouge_metrics'])
            st.markdown("###### All Readability Scores")
            st.json(result['readability_metrics'])

        render_feedback_section(result)
    
    close_card_wrapper()

def render_batch_page():
    main_card_wrapper("Batch Processing")
    
    st.info("Upload a CSV file containing a 'text' column for batch summarization or paraphrasing. An optional 'reference_text' column is highly recommended for full evaluation.")

    with st.form("batch_form"):
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
        
        col1, col2 = st.columns(2)
        with col1:
            task = st.selectbox("Select Task", ["Summarization", "Paraphrasing"])
        with col2:
            language = st.selectbox("Translate Output to", ["Hindi", "Telugu", "English"])
        
        submitted = st.form_submit_button("Start Batch Processing")

    if submitted and uploaded_file:
        with st.spinner(f"Initiating batch {task.lower()}..."):
            files = {'file': (uploaded_file.name, uploaded_file.getvalue(), 'text/csv')}
            url_params = f"/batch_process_csv?task={task}&language={language}"
            response = api_call("POST", url_params, files=files)
            
            if response:
                st.success(f"Batch processing started: {response['message']}")
                st.balloons()
            else:
                st.error("Batch initiation failed.")

    close_card_wrapper()

def render_history_page():
    main_card_wrapper("Processing History")

    with st.spinner("Fetching history..."):
        response = api_call("GET", "/history")
        
    if response and response.get('history'):
        history = response['history']
        st.subheader(f"Your Last {len(history)} Tasks")
        
        df_history = pd.json_normalize(history)
        
        cols_to_display = [
            'timestamp', 'task', 'model', 'original_text_snippet', 'elapsed_time', 
            'rouge_metrics.rouge-l', 'readability_metrics.flesch_reading_ease', 
            'reference_status', 'feedback_score', 'feedback_comment'
        ]
        
        final_cols = [col for col in cols_to_display if col in df_history.columns]

        rename_map = {
            'timestamp': 'Timestamp',
            'task': 'Task',
            'model': 'Model',
            'original_text_snippet': 'Original Text (Snippet)',
            'elapsed_time': 'Time (s)',
            'rouge_metrics.rouge-l': 'ROUGE-L',
            'readability_metrics.flesch_reading_ease': 'Flesch Ease',
            'reference_status': 'Reference',
            'feedback_score': 'Feedback Score',
            'feedback_comment': 'Feedback Comment'
        }
        
        df_display = df_history[final_cols].rename(columns=rename_map)
        df_display['Timestamp'] = pd.to_datetime(df_display['Timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')

        st.dataframe(df_display, use_container_width=True)
    else:
        st.info("No processing history found.")

    close_card_wrapper()
    
# =======================================
# ADMIN DASHBOARD RENDER FUNCTIONS
# =======================================

def render_analytics_tab():
    """Displays key metrics and model usage charts."""
    st.markdown("### Analytics Overview")
    
    with st.spinner("Fetching analytics data..."):
        response = api_call("GET", "/admin/analytics") 
    
    if response and response.get('stats'):
        stats = response['stats']
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Unique Users Logged In", stats.get('unique_users', 0))
        col2.metric("Total Generated Texts", stats.get('total_generated_texts', 0))
        col3.metric("Top Used Model", stats.get('top_used_model', 'N/A'))
        
        st.markdown("---")
        st.markdown("### Model Usage Counts")
        
        if stats.get('model_usage_counts'):
            df_usage = pd.DataFrame(
                list(stats['model_usage_counts'].items()), 
                columns=['Model', 'Usage Count']
            )
            
            chart = alt.Chart(df_usage).mark_bar(color='#C280FF').encode(
                x=alt.X('Model', sort='-y'),
                y=alt.Y('Usage Count'),
                tooltip=['Model', 'Usage Count']
            ).properties(
                title="Model Usage Count"
            ).interactive()
            
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No model usage data available.")

def render_user_management_tab():
    """Lists users and provides role management actions."""
    st.markdown("### User Management")
    
    with st.spinner("Fetching all users..."):
        response = api_call("GET", "/admin/users") 
        
    if response and response.get('users'):
        users = response['users']
        
        for user in users:
            user_id = user['id']
            email = user['email']
            role = user['role']

            st.markdown("---")
            st.markdown(f"**{email}** | Role: **{role.capitalize()}** | ID: **{user_id}**")
            
            col_p, col_d, col_del = st.columns(3)
            
            with col_p:
                if st.button("Promote to Admin", use_container_width=True, key=f"promote_{user_id}", disabled=(role=='admin')):
                    res = api_call("POST", f"/admin/user/{user_id}/promote")
                    if res:
                        st.toast(f"{email} promoted to admin!")
                        time.sleep(1)
                        st.rerun() 
            with col_d:
                if st.button("Demote to User", use_container_width=True, key=f"demote_{user_id}", disabled=(role=='user')):
                    res = api_call("POST", f"/admin/user/{user_id}/demote")
                    if res:
                        st.toast(f"{email} demoted to user!")
                        time.sleep(1)
                        st.rerun() 
            with col_del:
                if st.button("Delete User", use_container_width=True, key=f"delete_{user_id}", type="secondary"):
                    res = api_call("DELETE", f"/admin/user/{user_id}")
                    if res:
                        st.toast(f"User {email} deleted!")
                        time.sleep(1)
                        st.rerun()
    else:
        st.info("No user data available.")

def render_user_feedback_tab():
    """Displays feedback summary and a list of the top 10 positive/negative feedback comments."""
    st.markdown("### User Feedback Summary")
    
    with st.spinner("Fetching feedback summary..."):
        response_summary = api_call("GET", "/admin/feedback_summary") 

    if response_summary and response_summary.get('feedback_stats'):
        stats = response_summary['feedback_stats']
        
        col1, col2 = st.columns(2)
        total_feedback = stats.get('total_feedback', 0)
        likes = stats.get('likes', 0)
        dislikes = stats.get('dislikes', 0)
        
        like_pct = f"({(likes/total_feedback * 100):.1f}%)" if total_feedback > 0 else ""
        dislike_pct = f"({(dislikes/total_feedback * 100):.1f}%)" if total_feedback > 0 else ""
        
        col1.metric("Likes", f"{likes} {like_pct}")
        col2.metric("Dislikes", f"{dislikes} {dislike_pct}")

    st.markdown("---")
    st.markdown("### Top 10 Positive and Negative Feedback")

    with st.spinner("Fetching detailed feedback..."):
        response_detailed = api_call("GET", "/admin/detailed_feedback") 

    if response_detailed:
        positive_feedback = response_detailed.get('positive_feedback', [])
        negative_feedback = response_detailed.get('negative_feedback', [])

        st.markdown("#### 🌟 Top 10 Positive Feedback")
        if positive_feedback:
            df_pos = pd.DataFrame(positive_feedback)
            df_pos['timestamp'] = pd.to_datetime(df_pos['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
            st.dataframe(df_pos.rename(columns={'user_email': 'User', 'feedback_comment': 'Comment', 'timestamp': 'Date', 'task': 'Task'}), use_container_width=True)
        else:
            st.info("No positive feedback with comments found.")
            
        st.markdown("---")
            
        st.markdown("#### 👎 Top 10 Negative Feedback")
        if negative_feedback:
            df_neg = pd.DataFrame(negative_feedback)
            df_neg['timestamp'] = pd.to_datetime(df_neg['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
            st.dataframe(df_neg.rename(columns={'user_email': 'User', 'feedback_comment': 'Comment', 'timestamp': 'Date', 'task': 'Task'}), use_container_width=True)
        else:
            st.info("No negative feedback with comments found.")
    else:
        st.info("Could not retrieve detailed feedback records.")

# ========================================================
# REVISED CONTENT CURATION TAB
# ========================================================
def render_content_curation_tab():
    """Renders the Admin Content Curation tab with a card-based layout for history review."""
    
    if st.session_state.user_role != 'admin':
        st.error("Access Denied: Only administrators can view this page.")
        return
        
    st.markdown("### All Generated Content History")
    
    with st.spinner("Fetching all content history..."):
        response = api_call("GET", "/admin/all_history") 
        
    if not response or not response.get('history'):
        st.info("No content history records found.")
        return

    all_data = response['history']
    df_history = pd.json_normalize(all_data)

    if df_history.empty:
        st.info("No content history records found.")
        return

    # Prepare dataframe: ensure timestamp is datetime for sorting, but keep original string
    df_history['timestamp_str'] = df_history['timestamp']
    df_history['timestamp_dt'] = pd.to_datetime(df_history['timestamp'])
    df_history = df_history.sort_values('timestamp_dt', ascending=False)

    # Create a lookup map using the original string timestamp
    history_map = {item['timestamp']: item for item in all_data}

    # -- Filter Controls --
    all_users = ['All'] + sorted(df_history['user_email'].unique().tolist())
    
    selected_user = st.selectbox(
        "Filter by User Email", 
        all_users,
        index=all_users.index(st.session_state.admin_history_filter_user)
    )
    # Update session state to remember the filter choice
    st.session_state.admin_history_filter_user = selected_user
    
    # Apply filter
    if selected_user != 'All':
        df_display = df_history[df_history['user_email'] == selected_user].copy()
    else:
        df_display = df_history.copy()

    st.markdown(f"**Total Records Displayed:** {len(df_display)}")
    st.markdown("---")

    records_to_display = df_display.to_dict('records')

    if not records_to_display:
        st.info("No records match the current filter.")
        return

    # -- Loop through and display records in cards --
    for record_summary in records_to_display:
        timestamp = record_summary['timestamp_str']
        record = history_map.get(timestamp)

        if not record:
            continue

        # Each record gets its own container with a card style
        with st.container():
            st.markdown('<div class="card-wrapper" style="margin-bottom: 20px;">', unsafe_allow_html=True)
            
            # Display main record info
            st.markdown(f"**Record ID:** `{timestamp}` | **User:** `{record.get('user_email', 'N/A')}`")
            st.caption(f"Type: **{record.get('task', 'N/A')}** | Model: **{record.get('model', 'N/A')}**")
            
            st.markdown("**Input Text (Snippet):**")
            st.code(record.get('original_text_snippet', 'None Provided'))
            
            st.markdown("**Output Text:**")
            st.info(record.get('generated_output', 'None Provided'))

            # Use an expander for edit controls to keep the UI tidy
            with st.expander(f"Edit Content #{timestamp}"):
                
                # Each widget inside the loop needs a unique key
                new_output = st.text_area(
                    "Generated Output (Editable)",
                    value=record.get('generated_output', record.get('output_text', '')),
                    height=150,
                    key=f"textarea_{timestamp}"
                )

                col1, col2 = st.columns(2)

                with col1:
                    # UPDATE button logic
                    if st.button("Update", key=f"update_{timestamp}", use_container_width=True, type="primary"):
                        current_val = st.session_state[f"textarea_{timestamp}"]
                        if current_val.strip() and current_val != record['generated_output']:
                            with st.spinner("Updating content..."):
                                res = update_curated_content(
                                    timestamp=timestamp,
                                    user_email=record['user_email'],
                                    new_output=current_val
                                )
                            if res:
                                st.success(res.get('message', 'Update successful!'))
                                time.sleep(1.5)
                                st.rerun()
                            else:
                                st.error("Failed to update content.")
                        else:
                            st.warning("Content is empty or has not been changed.")
                
                with col2:
                    # DELETE button logic
                    if st.button("Delete", key=f"delete_{timestamp}", use_container_width=True, type="secondary"):
                        with st.spinner("Deleting record..."):
                            res = delete_curated_content(
                                timestamp=timestamp,
                                user_email=record['user_email']
                            )
                        if res:
                            st.success("Record deleted successfully.")
                            time.sleep(1.5)
                            st.rerun()
                        else:
                            st.error("Failed to delete record.")

            st.markdown('</div>', unsafe_allow_html=True)

def render_admin_dashboard():
    """Renders the main Admin Dashboard with tabs for different functions."""
    
    st.title("Admin Dashboard ⚙️")
    st.caption("Manage users, review content, and view platform analytics.")
    
    tab_analytics, tab_users, tab_feedback, tab_curation = st.tabs(
        ["Analytics", "User Management", "Feedback", "Content Curation"]
    )
    
    with tab_analytics:
        render_analytics_tab()
        
    with tab_users:
        render_user_management_tab()
        
    with tab_feedback:
        render_user_feedback_tab()
        
    with tab_curation:
        render_content_curation_tab()

# ===============================
# Main Application
# ===============================

set_custom_css()

# Handle token in query params for password reset flow
reset_token = st.query_params.get("token", None)

if reset_token:
    render_password_reset_page(reset_token)
elif not st.session_state.logged_in:
    render_login_page()
else:
    # --- Main App Title and Logout ---
    st.title("Multi-Task NLP Platform")
    st.caption(f"Logged in as: **{st.session_state.user_email}** (Role: **{st.session_state.user_role.capitalize()}**)")
    st.markdown("---")

    # --- Sidebar Navigation and Logout ---
    with st.sidebar:
        st.title("Navigation")
        
        menu_options = ["Summarization", "Paraphrasing", "Batch Processing", "History"]
        if st.session_state.user_role == 'admin':
            menu_options.append("Admin Dashboard")
            
        menu_choice = st.radio(
            "Select Task",
            tuple(menu_options),
            key="main_menu"
        )
        st.markdown("---")
        if st.button("Logout", use_container_width=True, key="btn_logout"):
            logout()

    # --- Main Content Render ---
    if menu_choice == "Summarization":
        render_task_page("Summarization")
    elif menu_choice == "Paraphrasing":
        render_task_page("Paraphrasing")
    elif menu_choice == "Batch Processing":
        render_batch_page()
    elif menu_choice == "History":
        render_history_page()
    elif menu_choice == "Admin Dashboard" and st.session_state.user_role == 'admin':
        render_admin_dashboard()
    else:
        # Default to summarization page if something goes wrong
        render_task_page("Summarization")
