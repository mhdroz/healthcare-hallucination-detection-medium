#Streamlit frontend for Healthcare RAG System.
#Connects to FastAPI backend for transparent medical AI.

import streamlit as st
import requests
import plotly.graph_objects as go
import pandas as pd


# Configure page
st.set_page_config(
    page_title="Healthcare RAG - Transparent AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE = "http://localhost:8000/api"

# Custom CSS for better styling and force light theme
st.markdown("""
<style>
    /* Force light theme */
    .stApp {
        background-color: white !important;
        color: black !important;
    }
    
    /* Force sidebar to be light */
    .css-1d391kg, .css-1d391kg .stSelectbox, .css-1d391kg .stTextArea {
        background-color: #f8f9fa !important;
        color: black !important;
    }
    
    /* Main header styling */
    .main-header {
        text-align: center;
        padding: 1.5rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        margin-bottom: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Safety metric boxes with better contrast */
    .safety-metric {
        text-align: center;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border: 2px solid #e1e5e9;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: black !important;
    }
    
    .metric-high { 
        background-color: #d4edda !important; 
        border-left: 5px solid #28a745 !important;
        border-color: #28a745 !important;
    }
    
    .metric-medium { 
        background-color: #fff3cd !important; 
        border-left: 5px solid #ffc107 !important;
        border-color: #ffc107 !important;
    }
    
    .metric-low { 
        background-color: #f8d7da !important; 
        border-left: 5px solid #dc3545 !important;
        border-color: #dc3545 !important;
    }
    
    /* Source chunk styling with better visibility */
    .source-chunk {
        background-color: #f8f9fa !important;
        padding: 1.2rem;
        border-radius: 8px;
        border: 2px solid #007bff;
        border-left: 6px solid #007bff !important;
        margin: 0.8rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: black !important;
    }
    
    /* Weak sentence styling with better contrast */
    .weak-sentence {
        background-color: #fff3cd !important;
        padding: 0.8rem;
        border-radius: 6px;
        border: 2px solid #ffc107;
        border-left: 6px solid #ffc107 !important;
        margin: 0.5rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        color: black !important;
    }
    
    /* Answer box styling */
    .answer-box {
        background-color: #f0f2f6 !important;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #007bff;
        border-left: 6px solid #007bff !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: black !important;
        font-size: 1.1rem;
        line-height: 1.6;
    }
    
    /* Force all text to be black */
    .stMarkdown, .stMarkdown p, .stMarkdown div, .stText {
        color: black !important;
    }
    
    /* Metric containers */
    .metric-container {
        background-color: white !important;
        border: 2px solid #e1e5e9;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa !important;
    }
    
    /* Force button styling */
    .stButton > button {
        background-color: #007bff !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 600 !important;
    }
    
    .stButton > button:hover {
        background-color: #0056b3 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


def check_backend_health():
    """Check if FastAPI backend is running."""
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        return response.status_code == 200 and response.json().get("models_loaded", False)
    except requests.exceptions.RequestException:
        return False


def get_sample_questions():
    """Get sample questions from backend."""
    try:
        response = requests.get(f"{API_BASE}/sample-questions", timeout=5)
        if response.status_code == 200:
            return response.json().get("samples", [])
    except requests.exceptions.RequestException:
        pass
    return []


def create_safety_gauge(score, max_score, title):
    """Create a circular gauge for safety scores."""
    percentage = (score / max_score) * 100 if max_score > 0 else 0
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = percentage,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 16}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': 'lightgray'},
                {'range': [50, 80], 'color': 'yellow'},
                {'range': [80, 100], 'color': 'lightgreen'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def create_metrics_chart(data):
    """Create a radar chart for safety metrics."""
    metrics = ['Attribution', 'Consistency', 'Entropy (inv)', 'External Val']
    
    # Invert entropy for display (lower entropy = better)
    entropy_display = max(0, 1 - (data['semantic_entropy'] / 3))
    external_score = 0
    if data.get('external_validation') and not data['external_validation'].get('error'):
        external_score = data['external_validation'].get('score', 0)
    
    values = [
        data['attribution_score'],
        data['consistency_score'], 
        entropy_display,
        external_score
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=metrics,
        fill='toself',
        name='Safety Metrics',
        line_color='rgb(0,123,255)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False,
        height=400,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    return fig


def display_safety_results(data):
    """Display comprehensive safety analysis results."""
    
    # Overall Safety Score
    st.markdown("## 🛡️ Safety Analysis Results")
    
    # Create columns for metrics
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        # Overall safety gauge
        safety_fig = create_safety_gauge(
            data['safety_score'], 
            data['max_safety_score'], 
            "Overall Safety"
        )
        st.plotly_chart(safety_fig, use_container_width=True)
        
        # Confidence badge
        confidence = data['confidence']
        if confidence == "HIGH CONFIDENCE":
            badge_class = "metric-high"
        elif confidence == "MEDIUM CONFIDENCE":
            badge_class = "metric-medium"
        else:
            badge_class = "metric-low"
        
        st.markdown(f"""
        <div class="safety-metric {badge_class}">
            <h4>{confidence}</h4>
            <p>Safety Score: {data['safety_score']}/{data['max_safety_score']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Individual metrics
        st.markdown("### 📊 Individual Metrics")
        
        interpretations = data['safety_interpretations']
        
        # Attribution
        attr_score = data['attribution_score']
        attr_class = "metric-high" if attr_score >= 0.7 else "metric-medium" if attr_score >= 0.5 else "metric-low"
        st.markdown(f"""
        <div class="safety-metric {attr_class}">
            <strong>Source Attribution: {attr_score:.3f}</strong><br>
            <small>{interpretations['attribution']}</small>
        </div>
        """, unsafe_allow_html=True)
        
        # Consistency
        cons_score = data['consistency_score']
        cons_class = "metric-high" if cons_score >= 0.8 else "metric-medium" if cons_score >= 0.6 else "metric-low"
        st.markdown(f"""
        <div class="safety-metric {cons_class}">
            <strong>Consistency: {cons_score:.3f}</strong><br>
            <small>{interpretations['consistency']}</small>
        </div>
        """, unsafe_allow_html=True)
        
        # Semantic Entropy
        entropy = data['semantic_entropy']
        entropy_class = "metric-high" if entropy < 1.0 else "metric-medium" if entropy < 2.0 else "metric-low"
        st.markdown(f"""
        <div class="safety-metric {entropy_class}">
            <strong>Semantic Entropy: {entropy:.3f}</strong><br>
            <small>{interpretations['entropy']}</small>
        </div>
        """, unsafe_allow_html=True)
        
        # External Validation
        if data.get('external_validation') and not data['external_validation'].get('error'):
            ext_score = data['external_validation'].get('score', 0)
            ext_class = "metric-high" if ext_score >= 0.7 else "metric-medium" if ext_score >= 0.5 else "metric-low"
            st.markdown(f"""
            <div class="safety-metric {ext_class}">
                <strong>External Validation: {ext_score:.3f}</strong><br>
                <small>Sources: {data['external_validation'].get('num_sources', 0)}</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="safety-metric metric-low">
                <strong>External Validation: Failed</strong><br>
                <small>External fact-checking unavailable</small>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        # Radar chart
        st.markdown("### 📈 Safety Profile")
        metrics_fig = create_metrics_chart(data)
        st.plotly_chart(metrics_fig, use_container_width=True)


def display_answer_and_sources(data):
    """Display the AI answer and source documents."""
    
    # AI Answer
    st.markdown("## 🤖 AI Response")
    st.markdown(f"**Question:** {data['question']}")

    st.markdown(f"""
    <div class="answer-box">
        {data['answer']}
    </div>
    """, unsafe_allow_html=True)
    
    # Weak sentences warning
    if data.get('has_weak_sentences') and data.get('weak_sentences'):
        st.warning("⚠️ Some sentences have weak source support:")
        for weak in data['weak_sentences']:
            st.markdown(f"""
            <div class="weak-sentence">
                <strong>Score: {weak['score']:.1%}</strong><br>
                {weak['sentence']}
            </div>
            """, unsafe_allow_html=True)
    
    # Source Documents
    if data.get('source_chunks'):
        st.markdown("## 📚 Source Documents")
        st.markdown(f"*Showing {len(data['source_chunks'])} source documents used to generate this answer*")
        
        for i, chunk in enumerate(data['source_chunks']):
            with st.expander(f"📄 {chunk.get('pmcid', f'Source {i+1}')} - {chunk.get('title', 'Medical Literature')[:80]}..."):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**PMCID:** {chunk.get('pmcid', 'N/A')}")
                    st.markdown(f"**Title:** {chunk.get('title', 'N/A')}")
                with col2:
                    if chunk.get('score'):
                        st.metric("Relevance", f"{chunk['score']:.1%}")
                
                st.markdown("**Content:**")
                st.text(chunk.get('text', 'No content available')[:1000] + ('...' if len(chunk.get('text', '')) > 1000 else ''))


def run_evaluation():
    """Run RAGAS evaluation on the system."""
    try:
        with st.spinner("Running RAGAS evaluation on test questions..."):
            response = requests.post(f"{API_BASE}/evaluate", timeout=300)
            
        if response.status_code == 200:
            eval_data = response.json()
            
            st.success("✅ Evaluation completed!")
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Faithfulness", 
                    f"{eval_data['faithfulness_score']:.1%}",
                    help="How well answers are grounded in sources"
                )
                st.caption(eval_data['faithfulness_interpretation'])
            
            with col2:
                st.metric(
                    "Answer Relevancy", 
                    f"{eval_data['relevancy_score']:.1%}",
                    help="How relevant answers are to questions"
                )
                st.caption(eval_data['relevancy_interpretation'])
            
            with col3:
                st.metric("Overall Grade", eval_data['overall_grade'])
                st.caption(f"Based on {eval_data['num_questions']} test questions")
            
            # Detailed scores
            if eval_data.get('detailed_scores'):
                st.markdown("### 📊 Detailed Question Scores")
                df = pd.DataFrame(eval_data['detailed_scores'])
                if not df.empty:
                    st.dataframe(df, use_container_width=True)
                    
                    # Create scatter plot - check which columns exist
                    if 'faithfulness' in df.columns and 'answer_relevancy' in df.columns:
                        fig = go.Figure()
                        
                        # Use index as text if 'id' column doesn't exist
                        hover_text = df.get('id', df.index).astype(str)
                        
                        fig.add_trace(go.Scatter(
                            x=df['faithfulness'],
                            y=df['answer_relevancy'],
                            mode='markers',
                            marker=dict(size=10, opacity=0.7),
                            text=hover_text,
                            hovertemplate="<b>Question %{text}</b><br>" +
                                        "Faithfulness: %{x:.3f}<br>" +
                                        "Relevancy: %{y:.3f}<extra></extra>",
                            name='Questions'
                        ))
                        fig.update_layout(
                            title="RAGAS Scores per Question",
                            xaxis_title="Faithfulness",
                            yaxis_title="Answer Relevancy",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Cannot create scatter plot - missing required columns in evaluation data")
                else:
                    st.info("No detailed scores available in evaluation results")
        else:
            st.error(f"Evaluation failed: {response.text}")
            
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to backend: {e}")
    except Exception as e:
        st.error(f"Unexpected error during evaluation: {e}")
        st.exception(e) 


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Healthcare RAG System</h1>
        <p>Transparent Medical AI with Comprehensive Safety Checks</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check backend health
    if not check_backend_health():
        st.error("🚨 Cannot connect to FastAPI backend. Please ensure it's running on http://localhost:8000")
        st.info("Run: `python web_app/main.py` to start the backend")
        return
    
    st.success("✅ Connected to FastAPI backend")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Sample questions
        st.subheader("📝 Sample Questions")
        sample_questions = get_sample_questions()
        if sample_questions:
            selected_sample = st.selectbox(
                "Choose a sample question:",
                [""] + sample_questions,
                help="Select a pre-written medical question"
            )
            if selected_sample:
                st.session_state.question = selected_sample
        
        # Safety options
        st.subheader("🛡️ Safety Options")
        fact_check = st.checkbox("External Fact-Checking", value=True, help="Validate against Semantic Scholar")
        multi_stage = st.checkbox("Multi-stage Retrieval", value=False, help="Break complex questions into parts")
        consistency_tries = st.slider("Consistency Checks", 2, 10, 3, help="Number of consistency tests")
        
        # System evaluation
        st.subheader("📊 System Evaluation")
        if st.button("🧪 Run RAGAS Evaluation", help="Test system with pneumonia questions"):
            run_evaluation()
    
    # Main content area
    st.header("❓ Ask a Medical Question")
    
    # Question input
    question = st.text_area(
        "Enter your medical question:",
        value=st.session_state.get('question', ''),
        height=100,
        placeholder="e.g., What are the common treatments for bacterial pneumonia?",
        help="Ask any medical question to see the AI's response with safety analysis"
    )
    
    # Analyze button
    if st.button("🔍 Analyze with Safety Checks", type="primary", use_container_width=True):
        if not question.strip():
            st.warning("Please enter a question first.")
            return
        
        # Prepare request
        request_data = {
            "question": question,
            "multi_stage": multi_stage,
            "fact_check": fact_check,
            "consistency_tries": consistency_tries
        }
        
        try:
            with st.spinner("Analyzing question with comprehensive safety checks..."):
                response = requests.post(
                    f"{API_BASE}/query", 
                    json=request_data,
                    timeout=300
                )
            
            if response.status_code == 200:
                data = response.json()
                
                # Display results
                display_safety_results(data)
                display_answer_and_sources(data)
                
            else:
                error_detail = response.json().get('detail', 'Unknown error')
                st.error(f"Analysis failed: {error_detail}")
                
        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to backend: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **⚠️ Medical Disclaimer:** This information is for educational purposes only and should not replace professional medical advice. 
    Always consult with a qualified healthcare provider for medical decisions.
    """)
    
    # Debug info in expander
    with st.expander("🔧 Debug Information"):
        st.write(f"API Base URL: {API_BASE}")
        st.write(f"Backend Health: {'✅ Healthy' if check_backend_health() else '❌ Unhealthy'}")


if __name__ == "__main__":
    main()