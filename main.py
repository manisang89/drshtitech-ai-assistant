# # import streamlit as st
# # import sqlite3
# # import pandas as pd
# # from datetime import datetime
# # import json
# # import re
# # from textblob import TextBlob
# # import google.generativeai as genai
# # from collections import Counter
# # import plotly.express as px
# # import plotly.graph_objects as go

# # # Page configuration
# # st.set_page_config(
# #     page_title="DrshtiTech - AI Business Assistant",
# #     page_icon="🔮",
# #     layout="wide",
# #     initial_sidebar_state="expanded"
# # )

# # # Initialize Database
# # def init_database():
# #     conn = sqlite3.connect('drshtitech.db')
# #     c = conn.cursor()
    
# #     # Feedback table
# #     c.execute('''CREATE TABLE IF NOT EXISTS feedback
# #                  (id INTEGER PRIMARY KEY AUTOINCREMENT,
# #                   source TEXT,
# #                   content TEXT,
# #                   sentiment REAL,
# #                   sentiment_label TEXT,
# #                   aspects TEXT,
# #                   timestamp DATETIME)''')
    
# #     # Insights table
# #     c.execute('''CREATE TABLE IF NOT EXISTS insights
# #                  (id INTEGER PRIMARY KEY AUTOINCREMENT,
# #                   insight_type TEXT,
# #                   content TEXT,
# #                   timestamp DATETIME)''')
    
# #     # Voice interactions table
# #     c.execute('''CREATE TABLE IF NOT EXISTS voice_logs
# #                  (id INTEGER PRIMARY KEY AUTOINCREMENT,
# #                   query TEXT,
# #                   response TEXT,
# #                   timestamp DATETIME)''')
    
# #     conn.commit()
# #     conn.close()

# # # Configure Gemini (you'll need to add your API key)
# # def configure_gemini():
# #     api_key = st.secrets.get("GEMINI_API_KEY", "")
# #     if api_key:
# #         genai.configure(api_key=api_key)
# #         return True
# #     return False

# # # Text Cleaning & Preprocessing
# # def clean_text(text):
# #     text = text.lower()
# #     text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
# #     text = re.sub(r'\@w+|\#','', text)
# #     text = re.sub(r'[^\w\s]', '', text)
# #     text = ' '.join(text.split())
# #     return text

# # # Sentiment Analysis
# # def analyze_sentiment(text):
# #     blob = TextBlob(text)
# #     polarity = blob.sentiment.polarity
    
# #     if polarity > 0.1:
# #         label = "Positive"
# #     elif polarity < -0.1:
# #         label = "Negative"
# #     else:
# #         label = "Neutral"
    
# #     return polarity, label

# # # Aspect Extraction (simplified)
# # def extract_aspects(text):
# #     aspects = {
# #         'product': ['product', 'quality', 'item', 'goods'],
# #         'service': ['service', 'support', 'help', 'staff'],
# #         'price': ['price', 'cost', 'expensive', 'cheap', 'affordable'],
# #         'delivery': ['delivery', 'shipping', 'arrived', 'late', 'fast'],
# #         'experience': ['experience', 'satisfaction', 'happy', 'disappointed']
# #     }
    
# #     text_lower = text.lower()
# #     found_aspects = []
    
# #     for aspect, keywords in aspects.items():
# #         if any(keyword in text_lower for keyword in keywords):
# #             found_aspects.append(aspect)
    
# #     return found_aspects if found_aspects else ['general']

# # # Store feedback in database
# # def store_feedback(source, content):
# #     cleaned_text = clean_text(content)
# #     sentiment_score, sentiment_label = analyze_sentiment(content)
# #     aspects = extract_aspects(content)
    
# #     conn = sqlite3.connect('drshtitech.db')
# #     c = conn.cursor()
# #     c.execute('''INSERT INTO feedback (source, content, sentiment, sentiment_label, aspects, timestamp)
# #                  VALUES (?, ?, ?, ?, ?, ?)''',
# #               (source, content, sentiment_score, sentiment_label, json.dumps(aspects), datetime.now()))
# #     conn.commit()
# #     conn.close()
    
# #     return sentiment_score, sentiment_label, aspects

# # # Generate AI insights using Gemini
# # def generate_insights(feedback_data):
# #     if not configure_gemini():
# #         return "Please configure GEMINI_API_KEY in Streamlit secrets to use AI insights."
    
# #     try:
# #         model = genai.GenerativeModel('gemini-2.5-flash')
        
# #         prompt = f"""
# #         Analyze the following customer feedback data and provide:
# #         1. Key trends and patterns
# #         2. Top issues to address
# #         3. Positive highlights
# #         4. 3 actionable recommendations
        
# #         Feedback Summary:
# #         - Total Feedback: {len(feedback_data)}
# #         - Positive: {len(feedback_data[feedback_data['sentiment_label'] == 'Positive'])}
# #         - Neutral: {len(feedback_data[feedback_data['sentiment_label'] == 'Neutral'])}
# #         - Negative: {len(feedback_data[feedback_data['sentiment_label'] == 'Negative'])}
        
# #         Recent Feedback Samples:
# #         {feedback_data.tail(10)['content'].to_string()}
# #         """
        
# #         response = model.generate_content(prompt)
# #         return response.text
# #     except Exception as e:
# #         return f"Error generating insights: {str(e)}"

# # # Gemini Query for Voice Interaction
# # def query_gemini(user_query, context=""):
# #     if not configure_gemini():
# #         return "Please configure GEMINI_API_KEY in Streamlit secrets."
    
# #     try:
# #         model = genai.GenerativeModel('gemini-2.5-flash')
# #         prompt = f"""You are DrshtiTech, an AI business assistant. 
# #         Context: {context}
# #         User Query: {user_query}
        
# #         Provide a helpful, concise response."""
        
# #         response = model.generate_content(prompt)
# #         return response.text
# #     except Exception as e:
# #         return f"Error: {str(e)}"

# # # Main Application
# # def main():
# #     init_database()
    
# #     # Sidebar Navigation
# #     st.sidebar.image("https://via.placeholder.com/150x50/4A90E2/ffffff?text=DrshtiTech", use_container_width=True)
# #     st.sidebar.title("Navigation")
# #     page = st.sidebar.radio("Go to", [
# #         "📊 Dashboard",
# #         "📥 Data Ingestion",
# #         "🤖 AI Analysis",
# #         "🎤 Voice Interaction",
# #         "💡 Insights & Suggestions"
# #     ])
    
# #     if page == "📊 Dashboard":
# #         show_dashboard()
# #     elif page == "📥 Data Ingestion":
# #         show_data_ingestion()
# #     elif page == "🤖 AI Analysis":
# #         show_ai_analysis()
# #     elif page == "🎤 Voice Interaction":
# #         show_voice_interaction()
# #     elif page == "💡 Insights & Suggestions":
# #         show_insights()

# # def show_dashboard():
# #     st.title("🔮 DrshtiTech - AI Business Assistant Dashboard")
# #     st.markdown("### Your Intelligent Micro-Business Companion")
    
# #     # Fetch data
# #     conn = sqlite3.connect('drshtitech.db')
# #     df = pd.read_sql_query("SELECT * FROM feedback ORDER BY timestamp DESC", conn)
# #     conn.close()
    
# #     if df.empty:
# #         st.info("No feedback data yet. Start by ingesting data from the Data Ingestion page!")
# #         return
    
# #     # Metrics
# #     col1, col2, col3, col4 = st.columns(4)
    
# #     with col1:
# #         st.metric("Total Feedback", len(df))
# #     with col2:
# #         positive = len(df[df['sentiment_label'] == 'Positive'])
# #         st.metric("Positive", positive, f"{positive/len(df)*100:.1f}%")
# #     with col3:
# #         negative = len(df[df['sentiment_label'] == 'Negative'])
# #         st.metric("Negative", negative, f"{negative/len(df)*100:.1f}%")
# #     with col4:
# #         avg_sentiment = df['sentiment'].mean()
# #         st.metric("Avg Sentiment", f"{avg_sentiment:.2f}")
    
# #     # Visualizations
# #     col1, col2 = st.columns(2)
    
# #     with col1:
# #         st.subheader("Sentiment Distribution")
# #         sentiment_counts = df['sentiment_label'].value_counts()
# #         fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index,
# #                      color_discrete_map={'Positive':'#2ecc71', 'Neutral':'#f39c12', 'Negative':'#e74c3c'})
# #         st.plotly_chart(fig, use_container_width=True)
    
# #     with col2:
# #         st.subheader("Feedback by Source")
# #         source_counts = df['source'].value_counts()
# #         fig = px.bar(x=source_counts.index, y=source_counts.values, 
# #                      labels={'x': 'Source', 'y': 'Count'})
# #         st.plotly_chart(fig, use_container_width=True)
    
# #     # Timeline
# #     st.subheader("Sentiment Trend Over Time")
# #     df['timestamp'] = pd.to_datetime(df['timestamp'])
# #     df_sorted = df.sort_values('timestamp')
# #     fig = px.line(df_sorted, x='timestamp', y='sentiment', color='source',
# #                   labels={'sentiment': 'Sentiment Score', 'timestamp': 'Date'})
# #     st.plotly_chart(fig, use_container_width=True)
    
# #     # Recent Feedback
# #     st.subheader("Recent Feedback")
# #     st.dataframe(df[['source', 'content', 'sentiment_label', 'timestamp']].head(10), use_container_width=True)

# # def show_data_ingestion():
# #     st.title("📥 Data Ingestion & Cleaning")
    
# #     tab1, tab2, tab3 = st.tabs(["Manual Entry", "CSV Upload", "Bulk Import"])
    
# #     with tab1:
# #         st.subheader("Enter Feedback Manually")
# #         source = st.selectbox("Feedback Source", ["Social Media", "Email", "Customer Reviews", "Survey"])
# #         content = st.text_area("Feedback Content", height=150)
        
# #         if st.button("Submit Feedback"):
# #             if content:
# #                 score, label, aspects = store_feedback(source, content)
# #                 st.success(f"✅ Feedback stored! Sentiment: {label} ({score:.2f})")
# #                 st.write(f"Detected Aspects: {', '.join(aspects)}")
# #             else:
# #                 st.warning("Please enter feedback content")
    
# #     with tab2:
# #         st.subheader("Upload CSV File")
# #         st.info("CSV should have columns: 'source' and 'content'")
# #         uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
        
# #         if uploaded_file:
# #             df = pd.read_csv(uploaded_file)
# #             st.write("Preview:", df.head())
            
# #             if st.button("Process CSV"):
# #                 if 'source' in df.columns and 'content' in df.columns:
# #                     progress_bar = st.progress(0)
# #                     for idx, row in df.iterrows():
# #                         store_feedback(row['source'], row['content'])
# #                         progress_bar.progress((idx + 1) / len(df))
# #                     st.success(f"✅ Processed {len(df)} feedback entries!")
# #                 else:
# #                     st.error("CSV must have 'source' and 'content' columns")
    
# #     with tab3:
# #         st.subheader("Bulk Sample Data")
# #         if st.button("Load Sample Data"):
# #             samples = [
# #                 ("Social Media", "Great product! Fast delivery and excellent quality."),
# #                 ("Email", "Customer service was unhelpful. Disappointed with response time."),
# #                 ("Customer Reviews", "Average experience. Product works but nothing special."),
# #                 ("Social Media", "Love this! Will definitely recommend to friends."),
# #                 ("Email", "Package arrived damaged. Waiting for replacement."),
# #             ]
# #             for source, content in samples:
# #                 store_feedback(source, content)
# #             st.success(f"✅ Loaded {len(samples)} sample feedback entries!")

# # def show_ai_analysis():
# #     st.title("🤖 AI Analysis Engine")
    
# #     conn = sqlite3.connect('drshtitech.db')
# #     df = pd.read_sql_query("SELECT * FROM feedback", conn)
# #     conn.close()
    
# #     if df.empty:
# #         st.warning("No feedback data available for analysis")
# #         return
    
# #     tab1, tab2 = st.tabs(["Sentiment Analysis", "Aspect Extraction"])
    
# #     with tab1:
# #         st.subheader("Sentiment Analysis Results")
        
# #         col1, col2 = st.columns(2)
# #         with col1:
# #             st.metric("Average Sentiment Score", f"{df['sentiment'].mean():.3f}")
# #             st.metric("Most Common Sentiment", df['sentiment_label'].mode()[0])
        
# #         with col2:
# #             st.write("**Sentiment Distribution:**")
# #             for label in ['Positive', 'Neutral', 'Negative']:
# #                 count = len(df[df['sentiment_label'] == label])
# #                 st.write(f"{label}: {count} ({count/len(df)*100:.1f}%)")
        
# #         # Detailed breakdown
# #         st.subheader("Sentiment by Source")
# #         sentiment_by_source = df.groupby(['source', 'sentiment_label']).size().reset_index(name='count')
# #         fig = px.bar(sentiment_by_source, x='source', y='count', color='sentiment_label',
# #                      barmode='group', color_discrete_map={'Positive':'#2ecc71', 'Neutral':'#f39c12', 'Negative':'#e74c3c'})
# #         st.plotly_chart(fig, use_container_width=True)
    
# #     with tab2:
# #         st.subheader("Aspect Extraction Results")
        
# #         all_aspects = []
# #         for aspects_str in df['aspects']:
# #             all_aspects.extend(json.loads(aspects_str))
        
# #         aspect_counts = Counter(all_aspects)
        
# #         col1, col2 = st.columns(2)
# #         with col1:
# #             st.write("**Top Aspects Mentioned:**")
# #             for aspect, count in aspect_counts.most_common(5):
# #                 st.write(f"- {aspect.title()}: {count} times")
        
# #         with col2:
# #             fig = px.bar(x=list(aspect_counts.keys()), y=list(aspect_counts.values()),
# #                         labels={'x': 'Aspect', 'y': 'Count'})
# #             st.plotly_chart(fig, use_container_width=True)

# # def show_voice_interaction():
# #     st.title("🎤 Voice Interaction")
# #     st.info("Voice input/output requires additional setup. Using text interface for now.")
    
# #     st.subheader("Chat with DrshtiTech AI")
    
# #     # Get context from database
# #     conn = sqlite3.connect('drshtitech.db')
# #     df = pd.read_sql_query("SELECT * FROM feedback", conn)
# #     conn.close()
    
# #     context = f"Business has {len(df)} feedback entries. "
# #     if not df.empty:
# #         context += f"Sentiment: {len(df[df['sentiment_label']=='Positive'])} positive, "
# #         context += f"{len(df[df['sentiment_label']=='Negative'])} negative."
    
# #     user_query = st.text_input("Ask DrshtiTech anything about your business:")
    
# #     if st.button("Get Response"):
# #         if user_query:
# #             with st.spinner("Thinking..."):
# #                 response = query_gemini(user_query, context)
# #                 st.write("**DrshtiTech:**", response)
                
# #                 # Log interaction
# #                 conn = sqlite3.connect('drshtitech.db')
# #                 c = conn.cursor()
# #                 c.execute("INSERT INTO voice_logs (query, response, timestamp) VALUES (?, ?, ?)",
# #                          (user_query, response, datetime.now()))
# #                 conn.commit()
# #                 conn.close()
# #         else:
# #             st.warning("Please enter a query")
    
# #     # Show history
# #     st.subheader("Recent Conversations")
# #     conn = sqlite3.connect('drshtitech.db')
# #     logs = pd.read_sql_query("SELECT * FROM voice_logs ORDER BY timestamp DESC LIMIT 5", conn)
# #     conn.close()
    
# #     for _, log in logs.iterrows():
# #         with st.expander(f"Q: {log['query'][:50]}..."):
# #             st.write(f"**You:** {log['query']}")
# #             st.write(f"**DrshtiTech:** {log['response']}")
# #             st.caption(f"Time: {log['timestamp']}")

# # def show_insights():
# #     st.title("💡 Insights & Suggestions")
    
# #     conn = sqlite3.connect('drshtitech.db')
# #     df = pd.read_sql_query("SELECT * FROM feedback", conn)
# #     conn.close()
    
# #     if df.empty:
# #         st.warning("No feedback data available for insights generation")
# #         return
    
# #     if st.button("🔮 Generate AI Insights", type="primary"):
# #         with st.spinner("Analyzing feedback and generating insights..."):
# #             insights = generate_insights(df)
            
# #             # Store insights
# #             conn = sqlite3.connect('drshtitech.db')
# #             c = conn.cursor()
# #             c.execute("INSERT INTO insights (insight_type, content, timestamp) VALUES (?, ?, ?)",
# #                      ("AI Generated", insights, datetime.now()))
# #             conn.commit()
# #             conn.close()
            
# #             st.success("✅ Insights generated!")
    
# #     # Display latest insights
# #     st.subheader("Latest AI Insights")
# #     conn = sqlite3.connect('drshtitech.db')
# #     insights_df = pd.read_sql_query("SELECT * FROM insights ORDER BY timestamp DESC LIMIT 3", conn)
# #     conn.close()
    
# #     if not insights_df.empty:
# #         for _, insight in insights_df.iterrows():
# #             with st.expander(f"Insight from {insight['timestamp']}", expanded=True):
# #                 st.markdown(insight['content'])
# #     else:
# #         st.info("No insights generated yet. Click the button above to generate insights!")
    
# #     # Quick stats
# #     st.subheader("Quick Campaign Suggestions")
# #     if not df.empty:
# #         positive_aspects = []
# #         negative_aspects = []
        
# #         for idx, row in df.iterrows():
# #             aspects = json.loads(row['aspects'])
# #             if row['sentiment_label'] == 'Positive':
# #                 positive_aspects.extend(aspects)
# #             elif row['sentiment_label'] == 'Negative':
# #                 negative_aspects.extend(aspects)
        
# #         col1, col2 = st.columns(2)
# #         with col1:
# #             st.success("**Strengths to Highlight:**")
# #             if positive_aspects:
# #                 top_positive = Counter(positive_aspects).most_common(3)
# #                 for aspect, count in top_positive:
# #                     st.write(f"✅ {aspect.title()}")
        
# #         with col2:
# #             st.warning("**Areas to Improve:**")
# #             if negative_aspects:
# #                 top_negative = Counter(negative_aspects).most_common(3)
# #                 for aspect, count in top_negative:
# #                     st.write(f"⚠️ {aspect.title()}")

# # if __name__ == "__main__":
# #     main()
# import streamlit as st
# import sqlite3
# import pandas as pd
# from datetime import datetime
# import json
# import re
# from textblob import TextBlob
# import google.generativeai as genai
# from collections import Counter
# import plotly.express as px
# import plotly.graph_objects as go

# # Page configuration
# st.set_page_config(
#     page_title="DrshtiTech - AI Business Assistant",
#     page_icon="🔮",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Initialize Database
# def init_database():
#     conn = sqlite3.connect('drshtitech.db')
#     c = conn.cursor()
    
#     # Feedback table
#     c.execute('''CREATE TABLE IF NOT EXISTS feedback
#                  (id INTEGER PRIMARY KEY AUTOINCREMENT,
#                   source TEXT,
#                   content TEXT,
#                   sentiment REAL,
#                   sentiment_label TEXT,
#                   aspects TEXT,
#                   timestamp DATETIME)''')
    
#     # Insights table
#     c.execute('''CREATE TABLE IF NOT EXISTS insights
#                  (id INTEGER PRIMARY KEY AUTOINCREMENT,
#                   insight_type TEXT,
#                   content TEXT,
#                   timestamp DATETIME)''')
    
#     # Voice interactions table
#     c.execute('''CREATE TABLE IF NOT EXISTS voice_logs
#                  (id INTEGER PRIMARY KEY AUTOINCREMENT,
#                   query TEXT,
#                   response TEXT,
#                   timestamp DATETIME)''')
    
#     conn.commit()
#     conn.close()

# # Configure Gemini (you'll need to add your API key)
# def configure_gemini():
#     api_key = st.secrets.get("GEMINI_API_KEY", "")
#     if api_key:
#         genai.configure(api_key=api_key)
#         return True
#     return False

# # Text Cleaning & Preprocessing
# def clean_text(text):
#     text = text.lower()
#     text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
#     text = re.sub(r'\@w+|\#','', text)
#     text = re.sub(r'[^\w\s]', '', text)
#     text = ' '.join(text.split())
#     return text

# # Sentiment Analysis
# def analyze_sentiment(text):
#     blob = TextBlob(text)
#     polarity = blob.sentiment.polarity
    
#     if polarity > 0.1:
#         label = "Positive"
#     elif polarity < -0.1:
#         label = "Negative"
#     else:
#         label = "Neutral"
    
#     return polarity, label

# # Aspect Extraction (simplified)
# def extract_aspects(text):
#     aspects = {
#         'product': ['product', 'quality', 'item', 'goods'],
#         'service': ['service', 'support', 'help', 'staff'],
#         'price': ['price', 'cost', 'expensive', 'cheap', 'affordable'],
#         'delivery': ['delivery', 'shipping', 'arrived', 'late', 'fast'],
#         'experience': ['experience', 'satisfaction', 'happy', 'disappointed']
#     }
    
#     text_lower = text.lower()
#     found_aspects = []
    
#     for aspect, keywords in aspects.items():
#         if any(keyword in text_lower for keyword in keywords):
#             found_aspects.append(aspect)
    
#     return found_aspects if found_aspects else ['general']

# # Store feedback in database
# def store_feedback(source, content):
#     cleaned_text = clean_text(content)
#     sentiment_score, sentiment_label = analyze_sentiment(content)
#     aspects = extract_aspects(content)
    
#     conn = sqlite3.connect('drshtitech.db')
#     c = conn.cursor()
#     c.execute('''INSERT INTO feedback (source, content, sentiment, sentiment_label, aspects, timestamp)
#                  VALUES (?, ?, ?, ?, ?, ?)''',
#               (source, content, sentiment_score, sentiment_label, json.dumps(aspects), datetime.now()))
#     conn.commit()
#     conn.close()
    
#     return sentiment_score, sentiment_label, aspects

# # Generate AI insights using Gemini
# def generate_insights(feedback_data):
#     if not configure_gemini():
#         return "Please configure GEMINI_API_KEY in Streamlit secrets to use AI insights."
    
#     try:
#         model = genai.GenerativeModel('gemini-2.5-flash')
        
#         prompt = f"""
#         Analyze the following customer feedback data and provide:
#         1. Key trends and patterns
#         2. Top issues to address
#         3. Positive highlights
#         4. 3 actionable recommendations
        
#         Feedback Summary:
#         - Total Feedback: {len(feedback_data)}
#         - Positive: {len(feedback_data[feedback_data['sentiment_label'] == 'Positive'])}
#         - Neutral: {len(feedback_data[feedback_data['sentiment_label'] == 'Neutral'])}
#         - Negative: {len(feedback_data[feedback_data['sentiment_label'] == 'Negative'])}
        
#         Recent Feedback Samples:
#         {feedback_data.tail(10)['content'].to_string()}
#         """
        
#         response = model.generate_content(prompt)
#         return response.text
#     except Exception as e:
#         return f"Error generating insights: {str(e)}"

# # Gemini Query for Voice Interaction
# def query_gemini(user_query, feedback_df=None):
#     if not configure_gemini():
#         return "Please configure GEMINI_API_KEY in Streamlit secrets."
    
#     try:
#         model = genai.GenerativeModel('gemini-2.5-flash')
        
#         # Build comprehensive context from database
#         context = ""
#         if feedback_df is not None and not feedback_df.empty:
#             context = f"""Business Feedback Database Summary:
# - Total Feedback: {len(feedback_df)}
# - Positive: {len(feedback_df[feedback_df['sentiment_label'] == 'Positive'])}
# - Neutral: {len(feedback_df[feedback_df['sentiment_label'] == 'Neutral'])}
# - Negative: {len(feedback_df[feedback_df['sentiment_label'] == 'Negative'])}

# Recent Feedback Details:
# """
#             # Add actual feedback content for analysis
#             for idx, row in feedback_df.tail(20).iterrows():
#                 context += f"\n[{row['source']} | {row['sentiment_label']}]: {row['content']}"
            
#             # Add negative feedback specifically if there are any
#             negative_feedback = feedback_df[feedback_df['sentiment_label'] == 'Negative']
#             if not negative_feedback.empty:
#                 context += f"\n\nNegative Feedback (Last 10):"
#                 for idx, row in negative_feedback.tail(10).iterrows():
#                     context += f"\n- {row['content']}"
            
#             # Add aspect analysis
#             all_aspects = []
#             for aspects_str in feedback_df['aspects']:
#                 all_aspects.extend(json.loads(aspects_str))
#             aspect_counts = Counter(all_aspects)
#             context += f"\n\nTop Mentioned Aspects: {dict(aspect_counts.most_common(5))}"
        
#         prompt = f"""You are DrshtiTech, an AI business assistant helping analyze customer feedback and business insights.

# {context}

# User Query: {user_query}

# Provide a helpful, specific, and actionable response based on the feedback data above. If analyzing issues, identify patterns and root causes."""
        
#         response = model.generate_content(prompt)
#         return response.text
#     except Exception as e:
#         return f"Error: {str(e)}"

# # Main Application
# def main():
#     init_database()
    
#     # Sidebar Navigation
#     st.sidebar.image("https://via.placeholder.com/150x50/4A90E2/ffffff?text=DrshtiTech", use_container_width=True)
#     st.sidebar.title("Navigation")
#     page = st.sidebar.radio("Go to", [
#         "📊 Dashboard",
#         "📥 Data Ingestion",
#         "🤖 AI Analysis",
#         "🎤 Voice Interaction",
#         "💡 Insights & Suggestions"
#     ])
    
#     if page == "📊 Dashboard":
#         show_dashboard()
#     elif page == "📥 Data Ingestion":
#         show_data_ingestion()
#     elif page == "🤖 AI Analysis":
#         show_ai_analysis()
#     elif page == "🎤 Voice Interaction":
#         show_voice_interaction()
#     elif page == "💡 Insights & Suggestions":
#         show_insights()

# def show_dashboard():
#     st.title("🔮 DrshtiTech - AI Business Assistant Dashboard")
#     st.markdown("### Your Intelligent Micro-Business Companion")
    
#     # Fetch data
#     conn = sqlite3.connect('drshtitech.db')
#     df = pd.read_sql_query("SELECT * FROM feedback ORDER BY timestamp DESC", conn)
#     conn.close()
    
#     if df.empty:
#         st.info("No feedback data yet. Start by ingesting data from the Data Ingestion page!")
#         return
    
#     # Metrics
#     col1, col2, col3, col4 = st.columns(4)
    
#     with col1:
#         st.metric("Total Feedback", len(df))
#     with col2:
#         positive = len(df[df['sentiment_label'] == 'Positive'])
#         st.metric("Positive", positive, f"{positive/len(df)*100:.1f}%")
#     with col3:
#         negative = len(df[df['sentiment_label'] == 'Negative'])
#         st.metric("Negative", negative, f"{negative/len(df)*100:.1f}%")
#     with col4:
#         avg_sentiment = df['sentiment'].mean()
#         st.metric("Avg Sentiment", f"{avg_sentiment:.2f}")
    
#     # Visualizations
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.subheader("Sentiment Distribution")
#         sentiment_counts = df['sentiment_label'].value_counts()
#         fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index,
#                      color_discrete_map={'Positive':'#2ecc71', 'Neutral':'#f39c12', 'Negative':'#e74c3c'})
#         st.plotly_chart(fig, use_container_width=True)
    
#     with col2:
#         st.subheader("Feedback by Source")
#         source_counts = df['source'].value_counts()
#         fig = px.bar(x=source_counts.index, y=source_counts.values, 
#                      labels={'x': 'Source', 'y': 'Count'})
#         st.plotly_chart(fig, use_container_width=True)
    
#     # Timeline
#     st.subheader("Sentiment Trend Over Time")
#     df['timestamp'] = pd.to_datetime(df['timestamp'])
#     df_sorted = df.sort_values('timestamp')
#     fig = px.line(df_sorted, x='timestamp', y='sentiment', color='source',
#                   labels={'sentiment': 'Sentiment Score', 'timestamp': 'Date'})
#     st.plotly_chart(fig, use_container_width=True)
    
#     # Recent Feedback
#     st.subheader("Recent Feedback")
#     st.dataframe(df[['source', 'content', 'sentiment_label', 'timestamp']].head(10), use_container_width=True)

# def show_data_ingestion():
#     st.title("📥 Data Ingestion & Cleaning")
    
#     tab1, tab2, tab3 = st.tabs(["Manual Entry", "CSV Upload", "Bulk Import"])
    
#     with tab1:
#         st.subheader("Enter Feedback Manually")
#         source = st.selectbox("Feedback Source", ["Social Media", "Email", "Customer Reviews", "Survey"])
#         content = st.text_area("Feedback Content", height=150)
        
#         if st.button("Submit Feedback"):
#             if content:
#                 score, label, aspects = store_feedback(source, content)
#                 st.success(f"✅ Feedback stored! Sentiment: {label} ({score:.2f})")
#                 st.write(f"Detected Aspects: {', '.join(aspects)}")
#             else:
#                 st.warning("Please enter feedback content")
    
#     with tab2:
#         st.subheader("Upload CSV File")
#         st.info("CSV should have columns: 'source' and 'content'")
#         uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
        
#         if uploaded_file:
#             df = pd.read_csv(uploaded_file)
#             st.write("Preview:", df.head())
            
#             if st.button("Process CSV"):
#                 if 'source' in df.columns and 'content' in df.columns:
#                     progress_bar = st.progress(0)
#                     for idx, row in df.iterrows():
#                         store_feedback(row['source'], row['content'])
#                         progress_bar.progress((idx + 1) / len(df))
#                     st.success(f"✅ Processed {len(df)} feedback entries!")
#                 else:
#                     st.error("CSV must have 'source' and 'content' columns")
    
#     with tab3:
#         st.subheader("Bulk Sample Data")
#         if st.button("Load Sample Data"):
#             samples = [
#                 ("Social Media", "Great product! Fast delivery and excellent quality."),
#                 ("Email", "Customer service was unhelpful. Disappointed with response time."),
#                 ("Customer Reviews", "Average experience. Product works but nothing special."),
#                 ("Social Media", "Love this! Will definitely recommend to friends."),
#                 ("Email", "Package arrived damaged. Waiting for replacement."),
#             ]
#             for source, content in samples:
#                 store_feedback(source, content)
#             st.success(f"✅ Loaded {len(samples)} sample feedback entries!")

# def show_ai_analysis():
#     st.title("🤖 AI Analysis Engine")
    
#     conn = sqlite3.connect('drshtitech.db')
#     df = pd.read_sql_query("SELECT * FROM feedback", conn)
#     conn.close()
    
#     if df.empty:
#         st.warning("No feedback data available for analysis")
#         return
    
#     tab1, tab2 = st.tabs(["Sentiment Analysis", "Aspect Extraction"])
    
#     with tab1:
#         st.subheader("Sentiment Analysis Results")
        
#         col1, col2 = st.columns(2)
#         with col1:
#             st.metric("Average Sentiment Score", f"{df['sentiment'].mean():.3f}")
#             st.metric("Most Common Sentiment", df['sentiment_label'].mode()[0])
        
#         with col2:
#             st.write("**Sentiment Distribution:**")
#             for label in ['Positive', 'Neutral', 'Negative']:
#                 count = len(df[df['sentiment_label'] == label])
#                 st.write(f"{label}: {count} ({count/len(df)*100:.1f}%)")
        
#         # Detailed breakdown
#         st.subheader("Sentiment by Source")
#         sentiment_by_source = df.groupby(['source', 'sentiment_label']).size().reset_index(name='count')
#         fig = px.bar(sentiment_by_source, x='source', y='count', color='sentiment_label',
#                      barmode='group', color_discrete_map={'Positive':'#2ecc71', 'Neutral':'#f39c12', 'Negative':'#e74c3c'})
#         st.plotly_chart(fig, use_container_width=True)
    
#     with tab2:
#         st.subheader("Aspect Extraction Results")
        
#         all_aspects = []
#         for aspects_str in df['aspects']:
#             all_aspects.extend(json.loads(aspects_str))
        
#         aspect_counts = Counter(all_aspects)
        
#         col1, col2 = st.columns(2)
#         with col1:
#             st.write("**Top Aspects Mentioned:**")
#             for aspect, count in aspect_counts.most_common(5):
#                 st.write(f"- {aspect.title()}: {count} times")
        
#         with col2:
#             fig = px.bar(x=list(aspect_counts.keys()), y=list(aspect_counts.values()),
#                         labels={'x': 'Aspect', 'y': 'Count'})
#             st.plotly_chart(fig, use_container_width=True)

# def show_voice_interaction():
#     st.title("🎤 Voice Interaction")
#     st.info("Voice input/output requires additional setup. Using text interface for now.")
    
#     st.subheader("Chat with DrshtiTech AI")
    
#     # Get complete feedback data from database
#     conn = sqlite3.connect('drshtitech.db')
#     df = pd.read_sql_query("SELECT * FROM feedback ORDER BY timestamp DESC", conn)
#     conn.close()
    
#     if df.empty:
#         st.warning("No feedback data available. Please add some feedback first from the Data Ingestion page.")
#         return
    
#     # Display quick stats
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         st.metric("Total Feedback", len(df))
#     with col2:
#         st.metric("Positive", len(df[df['sentiment_label']=='Positive']))
#     with col3:
#         st.metric("Negative", len(df[df['sentiment_label']=='Negative']))
    
#     st.markdown("---")
    
#     # Example queries
#     st.caption("💡 Try asking:")
#     example_queries = [
#         "What are customers complaining about?",
#         "Why are customers happy?",
#         "What should I improve first?",
#         "Summarize the main feedback trends",
#         "What aspects get the most negative feedback?"
#     ]
    
#     selected_example = st.selectbox("Or select an example query:", [""] + example_queries, label_visibility="collapsed")
    
#     user_query = st.text_input("Ask DrshtiTech anything about your business:", value=selected_example if selected_example else "")
    
#     if st.button("Get Response", type="primary"):
#         if user_query:
#             with st.spinner("Analyzing your feedback data..."):
#                 response = query_gemini(user_query, df)
                
#                 st.markdown("### 🤖 DrshtiTech Response:")
#                 st.markdown(response)
                
#                 # Log interaction
#                 conn = sqlite3.connect('drshtitech.db')
#                 c = conn.cursor()
#                 c.execute("INSERT INTO voice_logs (query, response, timestamp) VALUES (?, ?, ?)",
#                          (user_query, response, datetime.now()))
#                 conn.commit()
#                 conn.close()
                
#                 st.success("✅ Response saved to conversation history")
#         else:
#             st.warning("Please enter a query")
    
#     # Show history
#     st.markdown("---")
#     st.subheader("📜 Recent Conversations")
#     conn = sqlite3.connect('drshtitech.db')
#     logs = pd.read_sql_query("SELECT * FROM voice_logs ORDER BY timestamp DESC LIMIT 5", conn)
#     conn.close()
    
#     if not logs.empty:
#         for _, log in logs.iterrows():
#             with st.expander(f"Q: {log['query'][:60]}..." if len(log['query']) > 60 else f"Q: {log['query']}"):
#                 st.write(f"**You:** {log['query']}")
#                 st.markdown(f"**DrshtiTech:** {log['response']}")
#                 st.caption(f"🕐 {log['timestamp']}")
#     else:
#         st.info("No conversation history yet. Start asking questions!")

# def show_insights():
#     st.title("💡 Insights & Suggestions")
    
#     conn = sqlite3.connect('drshtitech.db')
#     df = pd.read_sql_query("SELECT * FROM feedback", conn)
#     conn.close()
    
#     if df.empty:
#         st.warning("No feedback data available for insights generation")
#         return
    
#     if st.button("🔮 Generate AI Insights", type="primary"):
#         with st.spinner("Analyzing feedback and generating insights..."):
#             insights = generate_insights(df)
            
#             # Store insights
#             conn = sqlite3.connect('drshtitech.db')
#             c = conn.cursor()
#             c.execute("INSERT INTO insights (insight_type, content, timestamp) VALUES (?, ?, ?)",
#                      ("AI Generated", insights, datetime.now()))
#             conn.commit()
#             conn.close()
            
#             st.success("✅ Insights generated!")
    
#     # Display latest insights
#     st.subheader("Latest AI Insights")
#     conn = sqlite3.connect('drshtitech.db')
#     insights_df = pd.read_sql_query("SELECT * FROM insights ORDER BY timestamp DESC LIMIT 3", conn)
#     conn.close()
    
#     if not insights_df.empty:
#         for _, insight in insights_df.iterrows():
#             with st.expander(f"Insight from {insight['timestamp']}", expanded=True):
#                 st.markdown(insight['content'])
#     else:
#         st.info("No insights generated yet. Click the button above to generate insights!")
    
#     # Quick stats
#     st.subheader("Quick Campaign Suggestions")
#     if not df.empty:
#         positive_aspects = []
#         negative_aspects = []
        
#         for idx, row in df.iterrows():
#             aspects = json.loads(row['aspects'])
#             if row['sentiment_label'] == 'Positive':
#                 positive_aspects.extend(aspects)
#             elif row['sentiment_label'] == 'Negative':
#                 negative_aspects.extend(aspects)
        
#         col1, col2 = st.columns(2)
#         with col1:
#             st.success("**Strengths to Highlight:**")
#             if positive_aspects:
#                 top_positive = Counter(positive_aspects).most_common(3)
#                 for aspect, count in top_positive:
#                     st.write(f"✅ {aspect.title()}")
        
#         with col2:
#             st.warning("**Areas to Improve:**")
#             if negative_aspects:
#                 top_negative = Counter(negative_aspects).most_common(3)
#                 for aspect, count in top_negative:
#                     st.write(f"⚠️ {aspect.title()}")

# if __name__ == "__main__":
#     main()
import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime
import json
import re
from textblob import TextBlob
import google.generativeai as genai
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
import hashlib
import secrets
import os

# Page configuration
st.set_page_config(
    page_title="DrshtiTech - AI Business Assistant",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Database with User Management
def init_database():
    conn = sqlite3.connect('drshtitech.db')
    c = conn.cursor()
    
    # Users table for authentication
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE,
                  email TEXT UNIQUE,
                  password_hash TEXT,
                  role TEXT DEFAULT 'user',
                  created_at DATETIME,
                  last_login DATETIME)''')
    
    # Feedback table
    c.execute('''CREATE TABLE IF NOT EXISTS feedback
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  source TEXT,
                  content TEXT,
                  sentiment REAL,
                  sentiment_label TEXT,
                  aspects TEXT,
                  timestamp DATETIME,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    # Insights table
    c.execute('''CREATE TABLE IF NOT EXISTS insights
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  insight_type TEXT,
                  content TEXT,
                  timestamp DATETIME,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    # Voice interactions table
    c.execute('''CREATE TABLE IF NOT EXISTS voice_logs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  query TEXT,
                  response TEXT,
                  timestamp DATETIME,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    # User sessions table
    c.execute('''CREATE TABLE IF NOT EXISTS user_sessions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  session_token TEXT,
                  created_at DATETIME,
                  expires_at DATETIME,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    # Create default admin user if not exists
    c.execute("SELECT COUNT(*) FROM users WHERE username = 'admin'")
    if c.fetchone()[0] == 0:
        default_password = hash_password("admin123")
        # Fixed datetime handling for Python 3.12+
        c.execute('''INSERT INTO users (username, email, password_hash, role, created_at) 
                     VALUES (?, ?, ?, ?, ?)''',
                 ('admin', 'admin@drshtitech.com', default_password, 'admin', datetime.now().isoformat()))
    
    conn.commit()
    conn.close()

# Enhanced Security Functions
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, password_hash):
    return hash_password(password) == password_hash

def generate_session_token():
    return secrets.token_urlsafe(32)

def create_session(user_id):
    conn = sqlite3.connect('drshtitech.db')
    c = conn.cursor()
    
    session_token = generate_session_token()
    created_at = datetime.now().isoformat()
    expires_at = datetime.now().timestamp() + 24 * 3600  # 24 hours
    
    # Fixed datetime handling
    c.execute('''INSERT INTO user_sessions (user_id, session_token, created_at, expires_at)
                 VALUES (?, ?, ?, ?)''', (user_id, session_token, created_at, expires_at))
    
    conn.commit()
    conn.close()
    return session_token

def validate_session(session_token):
    conn = sqlite3.connect('drshtitech.db')
    c = conn.cursor()
    
    c.execute('''SELECT u.id, u.username, u.role 
                 FROM user_sessions us 
                 JOIN users u ON us.user_id = u.id 
                 WHERE us.session_token = ? AND us.expires_at > ?''',
              (session_token, datetime.now().timestamp()))
    
    result = c.fetchone()
    conn.close()
    
    if result:
        return {'id': result[0], 'username': result[1], 'role': result[2]}
    return None

# Authentication System
def login_user(username, password):
    conn = sqlite3.connect('drshtitech.db')
    c = conn.cursor()
    
    c.execute('SELECT id, username, password_hash, role FROM users WHERE username = ?', (username,))
    user = c.fetchone()
    conn.close()
    
    if user and verify_password(password, user[2]):
        session_token = create_session(user[0])
        return {'success': True, 'session_token': session_token, 'user': {
            'id': user[0], 
            'username': user[1], 
            'role': user[3]
        }}
    return {'success': False, 'error': 'Invalid credentials'}

def register_user(username, email, password):
    conn = sqlite3.connect('drshtitech.db')
    c = conn.cursor()
    
    try:
        password_hash = hash_password(password)
        # Fixed datetime handling
        c.execute('''INSERT INTO users (username, email, password_hash, role, created_at) 
                     VALUES (?, ?, ?, ?, ?)''',
                 (username, email, password_hash, 'user', datetime.now().isoformat()))
        conn.commit()
        conn.close()
        return {'success': True, 'message': 'Registration successful'}
    except sqlite3.IntegrityError:
        conn.close()
        return {'success': False, 'error': 'Username or email already exists'}

# Enhanced Gemini Configuration with Fallback
def configure_gemini():
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if api_key:
        try:
            genai.configure(api_key=api_key)
            return True
        except Exception as e:
            st.error(f"Gemini configuration error: {str(e)}")
            return False
    return False

# Enhanced Text Processing
def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(text.split())
    return text

def analyze_sentiment(text):
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            label = "Positive"
        elif polarity < -0.1:
            label = "Negative"
        else:
            label = "Neutral"
        
        return polarity, label
    except Exception:
        return 0.0, "Neutral"

def extract_aspects(text):
    aspects = {
        'product': ['product', 'quality', 'item', 'goods', 'feature', 'design'],
        'service': ['service', 'support', 'help', 'staff', 'assistance', 'representative'],
        'price': ['price', 'cost', 'expensive', 'cheap', 'affordable', 'value', 'worth'],
        'delivery': ['delivery', 'shipping', 'arrived', 'late', 'fast', 'package', 'tracking'],
        'experience': ['experience', 'satisfaction', 'happy', 'disappointed', 'pleased', 'frustrated']
    }
    
    text_lower = text.lower()
    found_aspects = []
    
    for aspect, keywords in aspects.items():
        if any(keyword in text_lower for keyword in keywords):
            found_aspects.append(aspect)
    
    return found_aspects if found_aspects else ['general']

# Enhanced Data Storage with User Context
def store_feedback(source, content, user_id=None):
    cleaned_text = clean_text(content)
    sentiment_score, sentiment_label = analyze_sentiment(content)
    aspects = extract_aspects(content)
    
    conn = sqlite3.connect('drshtitech.db')
    c = conn.cursor()
    # Fixed datetime handling
    c.execute('''INSERT INTO feedback (user_id, source, content, sentiment, sentiment_label, aspects, timestamp)
                 VALUES (?, ?, ?, ?, ?, ?, ?)''',
              (user_id, source, content, sentiment_score, sentiment_label, json.dumps(aspects), datetime.now().isoformat()))
    conn.commit()
    conn.close()
    
    return sentiment_score, sentiment_label, aspects

# Enhanced AI Insights with Better Error Handling
def generate_insights(feedback_data, user_id=None):
    if not configure_gemini():
        # Fallback insights when Gemini is not available
        return generate_fallback_insights(feedback_data)
    
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = f"""
        As a business intelligence analyst, provide comprehensive insights about this customer feedback data:
        
        Data Overview:
        - Total Feedback: {len(feedback_data)}
        - Positive: {len(feedback_data[feedback_data['sentiment_label'] == 'Positive'])}
        - Neutral: {len(feedback_data[feedback_data['sentiment_label'] == 'Neutral'])}
        - Negative: {len(feedback_data[feedback_data['sentiment_label'] == 'Negative'])}
        
        Recent Feedback Samples:
        {feedback_data.tail(10)['content'].to_string() if len(feedback_data) >= 10 else feedback_data['content'].to_string()}
        
        Please provide:
        1. Key trends and patterns in customer feedback
        2. Top 3 issues requiring immediate attention
        3. Positive aspects to leverage in marketing
        4. 3 actionable recommendations for improvement
        5. Customer sentiment trajectory
        
        Format the response in clear sections with bullet points.
        """
        
        response = model.generate_content(prompt)
        
        # Store the insight
        if user_id:
            conn = sqlite3.connect('drshtitech.db')
            c = conn.cursor()
            # Fixed datetime handling
            c.execute('''INSERT INTO insights (user_id, insight_type, content, timestamp)
                         VALUES (?, ?, ?, ?)''',
                     (user_id, "AI Generated", response.text, datetime.now().isoformat()))
            conn.commit()
            conn.close()
        
        return response.text
    except Exception as e:
        st.error(f"AI Insight generation failed: {str(e)}")
        return generate_fallback_insights(feedback_data)

def generate_fallback_insights(feedback_data):
    """Generate basic insights when AI is unavailable"""
    positive_count = len(feedback_data[feedback_data['sentiment_label'] == 'Positive'])
    negative_count = len(feedback_data[feedback_data['sentiment_label'] == 'Negative'])
    neutral_count = len(feedback_data[feedback_data['sentiment_label'] == 'Neutral'])
    
    insights = f"""
    ## Basic Insights Analysis
    
    ### Sentiment Overview:
    - Positive Feedback: {positive_count} ({positive_count/len(feedback_data)*100:.1f}%)
    - Neutral Feedback: {neutral_count} ({neutral_count/len(feedback_data)*100:.1f}%)
    - Negative Feedback: {negative_count} ({negative_count/len(feedback_data)*100:.1f}%)
    
    ### Quick Recommendations:
    1. Focus on addressing negative feedback patterns
    2. Leverage positive aspects in your marketing
    3. Monitor sentiment trends regularly
    
    *Note: Enable Gemini AI for more detailed, actionable insights*
    """
    return insights

# Enhanced Query Function with Context
def query_gemini(user_query, feedback_df=None, user_context=None):
    if not configure_gemini():
        return "Please configure GEMINI_API_KEY in Streamlit secrets to enable AI features."
    
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Build comprehensive context
        context = "You are DrshtiTech, an AI business assistant specializing in customer feedback analysis and business intelligence.\n\n"
        
        if feedback_df is not None and not feedback_df.empty:
            context += f"""Business Feedback Analysis Context:
- Total Feedback Records: {len(feedback_df)}
- Customer Satisfaction: {len(feedback_df[feedback_df['sentiment_label'] == 'Positive'])} positive, {len(feedback_df[feedback_df['sentiment_label'] == 'Negative'])} negative
- Data Sources: {', '.join(feedback_df['source'].unique())}

Key Patterns:
"""
            
            # Add sentiment analysis
            sentiment_dist = feedback_df['sentiment_label'].value_counts()
            for sentiment, count in sentiment_dist.items():
                context += f"- {sentiment}: {count} cases\n"
            
            # Add aspect analysis
            all_aspects = []
            for aspects_str in feedback_df['aspects']:
                try:
                    all_aspects.extend(json.loads(aspects_str))
                except:
                    continue
            aspect_counts = Counter(all_aspects)
            context += f"\nTop Customer Concerns: {', '.join([f'{k}({v})' for k, v in aspect_counts.most_common(3)])}\n"
        
        if user_context:
            context += f"\nUser Context: {user_context}\n"
        
        prompt = f"""{context}

User Question: {user_query}

Provide a comprehensive, data-driven response that:
1. Directly addresses the question
2. References available data patterns
3. Offers actionable recommendations
4. Highlights potential business impacts
5. Suggests next steps

Keep the response professional yet accessible."""
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"I apologize, but I encountered an error: {str(e)}. Please try again or check your API configuration."

# Authentication UI
def show_login():
    st.title("🔮 DrshtiTech - AI Business Assistant")
    st.markdown("### Secure Login")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_btn = st.form_submit_button("Login", type="primary")
            
            if login_btn:
                if username and password:
                    result = login_user(username, password)
                    if result['success']:
                        st.session_state.update({
                            'authenticated': True,
                            'session_token': result['session_token'],
                            'user': result['user']
                        })
                        st.success("Login successful! Redirecting...")
                        st.rerun()
                    else:
                        st.error(result['error'])
                else:
                    st.warning("Please enter both username and password")
    
    with tab2:
        with st.form("register_form"):
            new_username = st.text_input("Choose Username")
            new_email = st.text_input("Email")
            new_password = st.text_input("Choose Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            register_btn = st.form_submit_button("Register", type="secondary")
            
            if register_btn:
                if new_username and new_email and new_password:
                    if new_password == confirm_password:
                        result = register_user(new_username, new_email, new_password)
                        if result['success']:
                            st.success(result['message'])
                        else:
                            st.error(result['error'])
                    else:
                        st.error("Passwords do not match")
                else:
                    st.warning("Please fill all fields")

def show_dashboard(user):
    st.title("🔮 DrshtiTech - AI Business Assistant Dashboard")
    st.markdown("### Your Intelligent Micro-Business Companion")
    
    # Fetch user-specific data
    conn = sqlite3.connect('drshtitech.db')
    df = pd.read_sql_query("SELECT * FROM feedback WHERE user_id = ? ORDER BY timestamp DESC", 
                          conn, params=(user['id'],))
    conn.close()
    
    if df.empty:
        st.info("No feedback data yet. Start by ingesting data from the Data Ingestion page!")
        return
    
    # Enhanced Metrics with better visual design
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Feedback", len(df), 
                 help="Total number of feedback entries collected")
    with col2:
        positive = len(df[df['sentiment_label'] == 'Positive'])
        st.metric("Positive", positive, f"{positive/len(df)*100:.1f}%",
                 help="Percentage of positive feedback")
    with col3:
        negative = len(df[df['sentiment_label'] == 'Negative'])
        st.metric("Negative", negative, f"{negative/len(df)*100:.1f}%",
                 help="Percentage of negative feedback")
    with col4:
        avg_sentiment = df['sentiment'].mean()
        st.metric("Avg Sentiment", f"{avg_sentiment:.2f}",
                 delta_color="off", help="Average sentiment score (-1 to 1)")
    
    # Enhanced Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Sentiment Distribution")
        sentiment_counts = df['sentiment_label'].value_counts()
        fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index,
                     color_discrete_map={'Positive':'#2ecc71', 'Neutral':'#f39c12', 'Negative':'#e74c3c'},
                     hole=0.4)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Feedback by Source")
        source_counts = df['source'].value_counts()
        fig = px.bar(x=source_counts.index, y=source_counts.values, 
                     labels={'x': 'Source', 'y': 'Count'},
                     color=source_counts.values,
                     color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
    
    # Enhanced Timeline with moving average
    st.subheader("📅 Sentiment Trend Over Time")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df_sorted = df.sort_values('timestamp')
    
    # Calculate rolling average for smoother trend
    df_sorted['sentiment_ma'] = df_sorted['sentiment'].rolling(window=5, min_periods=1).mean()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_sorted['timestamp'], y=df_sorted['sentiment_ma'],
                            mode='lines', name='Trend (Moving Avg)',
                            line=dict(color='#3498db', width=3)))
    fig.add_trace(go.Scatter(x=df_sorted['timestamp'], y=df_sorted['sentiment'],
                            mode='markers', name='Individual Feedback',
                            marker=dict(color=df_sorted['sentiment'], 
                                      colorscale='RdYlGn',
                                      size=6,
                                      colorbar=dict(title="Sentiment"))))
    
    fig.update_layout(xaxis_title='Date', yaxis_title='Sentiment Score')
    st.plotly_chart(fig, use_container_width=True)
    
    # Enhanced Recent Feedback with actions
    st.subheader("🆕 Recent Feedback")
    
    # Add quick filters
    col1, col2, col3 = st.columns(3)
    with col1:
        show_source = st.multiselect("Filter by Source", options=df['source'].unique(), default=df['source'].unique())
    with col2:
        show_sentiment = st.multiselect("Filter by Sentiment", options=df['sentiment_label'].unique(), default=df['sentiment_label'].unique())
    with col3:
        items_per_page = st.selectbox("Items per page", [5, 10, 20], index=1)
    
    filtered_df = df[(df['source'].isin(show_source)) & (df['sentiment_label'].isin(show_sentiment))]
    
    # Pagination
    if len(filtered_df) > 0:
        total_pages = (len(filtered_df) // items_per_page) + (1 if len(filtered_df) % items_per_page else 0)
        page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
        
        start_idx = (page - 1) * items_per_page
        end_idx = start_idx + items_per_page
        
        display_df = filtered_df.iloc[start_idx:end_idx][['source', 'content', 'sentiment_label', 'timestamp']]
        
        for _, row in display_df.iterrows():
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    sentiment_color = {
                        'Positive': '🟢',
                        'Neutral': '🟡', 
                        'Negative': '🔴'
                    }
                    st.write(f"{sentiment_color[row['sentiment_label']]} **{row['source']}** - {row['content']}")
                with col2:
                    st.caption(f"{row['timestamp'].strftime('%Y-%m-%d %H:%M')}")
                st.markdown("---")
    else:
        st.info("No feedback matches the selected filters")

def show_data_ingestion(user):
    st.title("📥 Data Ingestion & Cleaning")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Manual Entry", "CSV Upload", "Bulk Import", "API Integration"])
    
    with tab1:
        st.subheader("Enter Feedback Manually")
        source = st.selectbox("Feedback Source", ["Social Media", "Email", "Customer Reviews", "Survey", "Phone Call", "In-Person"])
        content = st.text_area("Feedback Content", height=150, placeholder="Enter customer feedback here...")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Submit Feedback", type="primary"):
                if content:
                    with st.spinner("Analyzing sentiment..."):
                        score, label, aspects = store_feedback(source, content, user['id'])
                    st.success(f"✅ Feedback stored! Sentiment: {label} ({score:.2f})")
                    st.write(f"🔍 Detected Aspects: {', '.join(aspects)}")
                    
                    # Show quick sentiment visualization
                    sentiment_emoji = {"Positive": "😊", "Negative": "😞", "Neutral": "😐"}
                    st.write(f"**Sentiment:** {sentiment_emoji[label]} {label}")
                else:
                    st.warning("Please enter feedback content")
        
        with col2:
            if st.button("Clear Form", type="secondary"):
                st.rerun()
    
    with tab2:
        st.subheader("Upload CSV File")
        st.info("""
        **CSV Format Requirements:**
        - Required columns: 'source', 'content'
        - Optional columns: 'timestamp'
        - Maximum file size: 200MB
        """)
        
        uploaded_file = st.file_uploader("Choose CSV file", type=['csv'], key="csv_upload")
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("**Preview:**", df.head())
                
                if st.button("Process CSV Data", type="primary"):
                    if 'source' in df.columns and 'content' in df.columns:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        success_count = 0
                        error_count = 0
                        
                        for idx, row in df.iterrows():
                            try:
                                store_feedback(row['source'], str(row['content']), user['id'])
                                success_count += 1
                            except Exception as e:
                                error_count += 1
                            
                            progress_bar.progress((idx + 1) / len(df))
                            status_text.text(f"Processed {idx + 1}/{len(df)} records...")
                        
                        st.success(f"✅ Processing complete! Success: {success_count}, Errors: {error_count}")
                        
                        if error_count > 0:
                            st.warning("Some records failed to process. Check the data format.")
                    else:
                        st.error("❌ CSV must contain 'source' and 'content' columns")
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")
    
    with tab3:
        st.subheader("Bulk Sample Data")
        st.warning("This will add sample data to your account for testing purposes.")
        
        sample_sizes = {
            "Small (10 records)": 10,
            "Medium (50 records)": 50,
            "Large (100 records)": 100
        }
        
        selected_size = st.selectbox("Sample Data Size", list(sample_sizes.keys()))
        
        if st.button(f"Generate {selected_size} Sample Records", type="secondary"):
            samples = generate_sample_data(sample_sizes[selected_size])
            
            progress_bar = st.progress(0)
            for idx, (source, content) in enumerate(samples):
                store_feedback(source, content, user['id'])
                progress_bar.progress((idx + 1) / len(samples))
            
            st.success(f"✅ Generated {len(samples)} sample feedback records!")
    
    with tab4:
        st.subheader("API Integration")
        st.info("""
        **Coming Soon:** 
        - REST API endpoints for automated data ingestion
        - Webhook integration for real-time feedback
        - Third-party platform connectors
        """)
        
        st.code("""
        # Example API Endpoint (Future)
        POST /api/v1/feedback
        Content-Type: application/json
        Authorization: Bearer YOUR_API_KEY
        
        {
            "source": "webhook",
            "content": "Customer feedback text",
            "metadata": {...}
        }
        """, language="python")

def generate_sample_data(count):
    """Generate realistic sample feedback data"""
    sources = ["Social Media", "Email", "Customer Reviews", "Survey", "Phone Call"]
    
    positive_feedback = [
        "Excellent product quality and fast shipping!",
        "Very satisfied with the customer service experience",
        "Great value for money, will purchase again",
        "The product exceeded my expectations",
        "Quick response and helpful support team"
    ]
    
    negative_feedback = [
        "Product arrived damaged and late",
        "Poor customer service, unhelpful staff",
        "Too expensive for what you get",
        "Quality doesn't match the price point",
        "Shipping took longer than promised"
    ]
    
    neutral_feedback = [
        "Average product, nothing special",
        "Met basic expectations",
        "Standard service and delivery",
        "Product works as described",
        "Reasonable experience overall"
    ]
    
    all_feedback = positive_feedback + negative_feedback + neutral_feedback
    samples = []
    
    for _ in range(count):
        source = secrets.choice(sources)
        content = secrets.choice(all_feedback)
        samples.append((source, content))
    
    return samples

def show_ai_analysis(user):
    st.title("🤖 AI Analysis Engine")
    
    conn = sqlite3.connect('drshtitech.db')
    df = pd.read_sql_query("SELECT * FROM feedback WHERE user_id = ?", conn, params=(user['id'],))
    conn.close()
    
    if df.empty:
        st.warning("No feedback data available for analysis")
        return
    
    tab1, tab2 = st.tabs(["Sentiment Analysis", "Aspect Extraction"])
    
    with tab1:
        st.subheader("Sentiment Analysis Results")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Average Sentiment Score", f"{df['sentiment'].mean():.3f}")
            st.metric("Most Common Sentiment", df['sentiment_label'].mode()[0] if not df['sentiment_label'].mode().empty else "N/A")
        
        with col2:
            st.write("**Sentiment Distribution:**")
            for label in ['Positive', 'Neutral', 'Negative']:
                count = len(df[df['sentiment_label'] == label])
                st.write(f"{label}: {count} ({count/len(df)*100:.1f}%)")
        
        # Detailed breakdown
        st.subheader("Sentiment by Source")
        sentiment_by_source = df.groupby(['source', 'sentiment_label']).size().reset_index(name='count')
        fig = px.bar(sentiment_by_source, x='source', y='count', color='sentiment_label',
                     barmode='group', color_discrete_map={'Positive':'#2ecc71', 'Neutral':'#f39c12', 'Negative':'#e74c3c'})
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Aspect Extraction Results")
        
        all_aspects = []
        for aspects_str in df['aspects']:
            try:
                all_aspects.extend(json.loads(aspects_str))
            except:
                continue
        
        aspect_counts = Counter(all_aspects)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Top Aspects Mentioned:**")
            for aspect, count in aspect_counts.most_common(5):
                st.write(f"- {aspect.title()}: {count} times")
        
        with col2:
            if aspect_counts:
                fig = px.bar(x=list(aspect_counts.keys()), y=list(aspect_counts.values()),
                            labels={'x': 'Aspect', 'y': 'Count'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No aspects detected in feedback")

def show_voice_interaction(user):
    st.title("🎤 Voice Interaction")
    st.info("Voice input/output requires additional setup. Using text interface for now.")
    
    st.subheader("Chat with DrshtiTech AI")
    
    # Get complete feedback data from database
    conn = sqlite3.connect('drshtitech.db')
    df = pd.read_sql_query("SELECT * FROM feedback WHERE user_id = ? ORDER BY timestamp DESC", conn, params=(user['id'],))
    conn.close()
    
    if df.empty:
        st.warning("No feedback data available. Please add some feedback first from the Data Ingestion page.")
        return
    
    # Display quick stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Feedback", len(df))
    with col2:
        st.metric("Positive", len(df[df['sentiment_label']=='Positive']))
    with col3:
        st.metric("Negative", len(df[df['sentiment_label']=='Negative']))
    
    st.markdown("---")
    
    # Example queries
    st.caption("💡 Try asking:")
    example_queries = [
        "What are customers complaining about?",
        "Why are customers happy?",
        "What should I improve first?",
        "Summarize the main feedback trends",
        "What aspects get the most negative feedback?"
    ]
    
    selected_example = st.selectbox("Or select an example query:", [""] + example_queries, label_visibility="collapsed")
    
    user_query = st.text_input("Ask DrshtiTech anything about your business:", value=selected_example if selected_example else "")
    
    if st.button("Get Response", type="primary"):
        if user_query:
            with st.spinner("Analyzing your feedback data..."):
                response = query_gemini(user_query, df)
                
                st.markdown("### 🤖 DrshtiTech Response:")
                st.markdown(response)
                
                # Log interaction
                conn = sqlite3.connect('drshtitech.db')
                c = conn.cursor()
                # Fixed datetime handling
                c.execute("INSERT INTO voice_logs (user_id, query, response, timestamp) VALUES (?, ?, ?, ?)",
                         (user['id'], user_query, response, datetime.now().isoformat()))
                conn.commit()
                conn.close()
                
                st.success("✅ Response saved to conversation history")
        else:
            st.warning("Please enter a query")
    
    # Show history
    st.markdown("---")
    st.subheader("📜 Recent Conversations")
    conn = sqlite3.connect('drshtitech.db')
    logs = pd.read_sql_query("SELECT * FROM voice_logs WHERE user_id = ? ORDER BY timestamp DESC LIMIT 5", 
                            conn, params=(user['id'],))
    conn.close()
    
    if not logs.empty:
        for _, log in logs.iterrows():
            with st.expander(f"Q: {log['query'][:60]}..." if len(log['query']) > 60 else f"Q: {log['query']}"):
                st.write(f"**You:** {log['query']}")
                st.markdown(f"**DrshtiTech:** {log['response']}")
                st.caption(f"🕐 {log['timestamp']}")
    else:
        st.info("No conversation history yet. Start asking questions!")

def show_insights(user):
    st.title("💡 Insights & Suggestions")
    
    conn = sqlite3.connect('drshtitech.db')
    df = pd.read_sql_query("SELECT * FROM feedback WHERE user_id = ?", conn, params=(user['id'],))
    conn.close()
    
    if df.empty:
        st.warning("No feedback data available for insights generation")
        return
    
    if st.button("🔮 Generate AI Insights", type="primary"):
        with st.spinner("Analyzing feedback and generating insights..."):
            insights = generate_insights(df, user['id'])
            
            st.success("✅ Insights generated!")
    
    # Display latest insights
    st.subheader("Latest AI Insights")
    conn = sqlite3.connect('drshtitech.db')
    insights_df = pd.read_sql_query("SELECT * FROM insights WHERE user_id = ? ORDER BY timestamp DESC LIMIT 3", 
                                   conn, params=(user['id'],))
    conn.close()
    
    if not insights_df.empty:
        for _, insight in insights_df.iterrows():
            with st.expander(f"Insight from {insight['timestamp']}", expanded=True):
                st.markdown(insight['content'])
    else:
        st.info("No insights generated yet. Click the button above to generate insights!")
    
    # Quick stats
    st.subheader("Quick Campaign Suggestions")
    if not df.empty:
        positive_aspects = []
        negative_aspects = []
        
        for idx, row in df.iterrows():
            try:
                aspects = json.loads(row['aspects'])
                if row['sentiment_label'] == 'Positive':
                    positive_aspects.extend(aspects)
                elif row['sentiment_label'] == 'Negative':
                    negative_aspects.extend(aspects)
            except:
                continue
        
        col1, col2 = st.columns(2)
        with col1:
            st.success("**Strengths to Highlight:**")
            if positive_aspects:
                top_positive = Counter(positive_aspects).most_common(3)
                for aspect, count in top_positive:
                    st.write(f"✅ {aspect.title()}")
            else:
                st.write("No positive aspects detected")
        
        with col2:
            st.warning("**Areas to Improve:**")
            if negative_aspects:
                top_negative = Counter(negative_aspects).most_common(3)
                for aspect, count in top_negative:
                    st.write(f"⚠️ {aspect.title()}")
            else:
                st.write("No negative aspects detected")

def show_user_management():
    st.title("👤 User Management")
    st.warning("Admin Access Required")
    
    conn = sqlite3.connect('drshtitech.db')
    users_df = pd.read_sql_query("SELECT id, username, email, role, created_at, last_login FROM users", conn)
    conn.close()
    
    st.subheader("User Accounts")
    st.dataframe(users_df, use_container_width=True)
    
    st.subheader("Create New User")
    with st.form("create_user_form"):
        col1, col2 = st.columns(2)
        with col1:
            new_username = st.text_input("Username")
            new_email = st.text_input("Email")
        with col2:
            new_password = st.text_input("Password", type="password")
            new_role = st.selectbox("Role", ["user", "admin"])
        
        if st.form_submit_button("Create User", type="primary"):
            if new_username and new_email and new_password:
                result = register_user(new_username, new_email, new_password)
                if result['success']:
                    st.success("User created successfully!")
                    st.rerun()
                else:
                    st.error(result['error'])
            else:
                st.warning("Please fill all fields")

# Enhanced Main Application with Authentication
def main():
    # Initialize database
    init_database()
    
    # Check authentication
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    # Fixed: Use st.query_params instead of st.experimental_get_query_params
    query_params = st.query_params
    if 'session_token' in query_params and not st.session_state.authenticated:
        session_data = validate_session(query_params['session_token'])
        if session_data:
            st.session_state.update({
                'authenticated': True,
                'user': session_data,
                'session_token': query_params['session_token']
            })
    
    if not st.session_state.authenticated:
        show_login()
        return
    
    # Main application for authenticated users
    user = st.session_state.get('user', {})
    
    # Enhanced sidebar with user info
    st.sidebar.image("logo.png", width=150)
    st.sidebar.title(f"Welcome, {user.get('username', 'User')}!")
    st.sidebar.caption(f"Role: {user.get('role', 'user')}")
    
    if st.sidebar.button("🚪 Logout"):
        st.session_state.clear()
        st.query_params.clear()
        st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", [
        "📊 Dashboard",
        "📥 Data Ingestion",
        "🤖 AI Analysis",
        "🎤 Voice Interaction",
        "💡 Insights & Suggestions",
        "👤 User Management"
    ])
    
    if page == "📊 Dashboard":
        show_dashboard(user)
    elif page == "📥 Data Ingestion":
        show_data_ingestion(user)
    elif page == "🤖 AI Analysis":
        show_ai_analysis(user)
    elif page == "🎤 Voice Interaction":
        show_voice_interaction(user)
    elif page == "💡 Insights & Suggestions":
        show_insights(user)
    elif page == "👤 User Management" and user.get('role') == 'admin':
        show_user_management()
    elif page == "👤 User Management":
        st.warning("⛔ Admin access required for user management")

if __name__ == "__main__":
    main()