import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
import warnings
warnings.filterwarnings('ignore')

# Set page config with custom theme
st.set_page_config(
    page_title="Mental Health Treatment Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling

# Set page config
st.set_page_config(
    page_title="Mental Health Treatment Predictor",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* === General Background === */
    .stApp {
        background-color: #3D2217; /* Beige */
    }
            
    /* === Headers === */
    h1, h2, h3, h4, h5, h6 {
        color: #013220; /* dark green */
        font-weight: 700;
    }
            
     /* === Sidebar === */
    section[data-testid="stSidebar"] {
        background: #014d26; /* dark green */
        color: black;
    }
    section[data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* === Input fields === */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stTextArea textarea {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #4A90E2 !important;
    }
            
    /* Placeholder text color for all input types */
    .stTextInput > div > div > input::placeholder,
    .stNumberInput > div > div > input::placeholder,
    .stTextArea textarea::placeholder {
        color: #000000 !important;
        opacity: 1;
    }
    
    /* === Main content text === */
    .main * {
        olor: #000000 !important;
    }    
    .main p, .main div, .main span, .main h1, .main h2, .main h3 {
    color: #000000 !important;
   }           

     /* === Buttons === */
    .stButton>button {
        background: linear-gradient(90deg, #20b2aa, #87ceeb);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.8rem 1.5rem;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease-in-out;
    }
    .stButton>button:hover {
        background: linear-gradient( #20b2aa);
        transform: scale(1.05);
    }
     /* === Metrics === */
    .stMetric {
        background: #014d26; /* soft green-beige */
        border: 1.5px solid #20b2aa;
        border-radius: 12px;
        padding: 1rem;
    }

    /* === Tabs === */
    .stTabs [data-baseweb="tab"] {
        background: #f0f5e1;
        border-radius: 8px 8px 0 0;
        border: 1px solid #20b2aa;
        padding: 12px 18px;
        font-size: 15px;
        color: #2f4f4f;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: #20b2aa;
        color: white !important;
    }

    /* === Custom Cards === */
    .custom-card {
        background: linear-gradient(45deg, #ffffff, #f0f5e1);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #87ceeb;
        margin-bottom: 1rem;
    }       
            
</style>
""", unsafe_allow_html=True)







# Load models and encoders
@st.cache_resource
def load_models():
    try:
        mlp_model = joblib.load("mlp_model.pkl")
        label_encoders = joblib.load("label_encoders.pkl")
        gender_encoder = joblib.load("gender_encoder.pkl")
        scaler = joblib.load("scaler.pkl")
        target_encoder = joblib.load("target_encoder.pkl")
        ohe_columns = joblib.load("ohe_columns.pkl")
        
        # Load ordinal encoders
        ordinal_encoders = {}
        ordinal_features = ['Mood_Swings', 'Days_Indoors', 'care_options', 
                           'mental_health_interview', 'Growing_Stress']
        for feature in ordinal_features:
            try:
                ordinal_encoders[feature] = joblib.load(f"{feature}_encoder.pkl")
            except:
                st.error(f"Could not load {feature}_encoder.pkl")
        
        return mlp_model, label_encoders, gender_encoder, scaler, target_encoder, ohe_columns, ordinal_encoders
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None, None, None

# Generate sample data for dashboard
def generate_sample_data():
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'Age': np.random.randint(18, 70, n_samples),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Occupation': np.random.choice(['Corporate', 'Business', 'Student', 'Housewife', 'Other'], n_samples),
        'Treatment_Need': np.random.choice(['Yes', 'No'], n_samples, p=[0.35, 0.65]),
        'Stress_Level': np.random.randint(1, 10, n_samples),
        'Mood_Swing_Intensity': np.random.randint(1, 10, n_samples),
        'Country': np.random.choice(['USA', 'UK', 'Canada', 'Australia', 'Other'], n_samples)
    }
    
    return pd.DataFrame(data)

# Dashboard page
def show_dashboard():
    st.markdown("""
    <div style="background:#014d26; padding: 2rem; border-radius: 15px; margin-bottom: 2rem;">
        <h1 style="color: white; text-align: center; margin: 0;">📊 Mental Health Dashboard</h1>
        <p style="color: white; text-align: center; margin: 0;">Comprehensive overview of mental charts and predictions</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load your actual dataset
    data = pd.read_csv('Mental Health Dataset_NEW (2).csv')
    
    # Clean the data (based on your notebook preprocessing)
    data = data.dropna()
    data = data.drop_duplicates()
    data = data.drop(columns=["Timestamp"])
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", f"{len(data):,}")
    
    with col2:
        treatment_rate = (data['treatment'] == 'Yes').mean() * 100
        st.metric("Treatment Rate", f"{treatment_rate:.1f}%")
    
    with col3:
        # Calculate average mood swings (convert to numerical)
        mood_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
        data['Mood_Swings_Num'] = data['Mood_Swings'].map(mood_mapping)
        avg_mood = data['Mood_Swings_Num'].mean()
        st.metric("Avg. Mood Swing Level", f"{avg_mood:.1f}")
    
    with col4:
        # Count unique countries
        unique_countries = data['Country'].nunique()
        st.metric("Countries Represented", unique_countries)
    
    # Charts and visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Treatment by Gender")
        gender_treatment = data.groupby(['Gender', 'treatment']).size().unstack(fill_value=0)
        fig = px.bar(gender_treatment, barmode='group', 
                    color_discrete_sequence=['#2e8b57', '#014d26'],
                    labels={'value': 'Count', 'variable': 'Treatment', 'Gender': 'Gender'})
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Growing Stress by Treatment")
        stress_treatment = data.groupby(['Growing_Stress', 'treatment']).size().unstack(fill_value=0)
        fig = px.bar(stress_treatment, barmode='group', 
                    color_discrete_sequence=['#2e8b57', '#014d26'],  # Dark green and light green
                    labels={'value': 'Count', 'variable': 'Treatment', 'Growing_Stress': 'Growing Stress'})
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    
    # Additional charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Treatment by Occupation")
        occupation_treatment = data.groupby(['Occupation', 'treatment']).size().unstack(fill_value=0)
        fig = px.bar(occupation_treatment, barmode='group', 
                    color_discrete_sequence=['#2e8b57', '#1a75ff'],
                    labels={'value': 'Count', 'variable': 'Treatment', 'Occupation': 'Occupation'})
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Country Distribution")
        # Group smaller countries into "Other" category
        country_counts = data['Country'].value_counts()
        top_countries = country_counts.head(8)  # Show top 8 countries
        other_count = country_counts[8:].sum() if len(country_counts) > 8 else 0
        
        if other_count > 0:
            top_countries['Other'] = other_count
            
        fig = px.pie(values=top_countries.values, names=top_countries.index, 
                    color_discrete_sequence=px.colors.sequential.Blues_r)
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    
     # Additional insights based on actual data
  
    st.markdown("""
    <div style="display: flex; justify-content: center;">
        <div style="background:#014d26; padding: 0.5rem; border-radius: 10px; margin-bottom: 2rem; max-width: 400px; width: 100%;">
            <h2 style="color: white; text-align: center; margin: 0;"> Key Insights</h2>
        </div>
    </div>
    """, unsafe_allow_html=True)


    st.markdown("""
    <div style="background:#014d26; padding: 2rem; border-radius: 10px; border: 2px solid #2e8b57;">

    ### ⚡ Top Mental Health Indicators
    Family history  
    Care options  
    Gender  
    Mental health interview  
    
    ### ⚡ Impact of Coping Strategies
    Coping struggles affect willingness to seek help  
    Availability of care options improves treatment-seeking behavior  
    Support systems play a critical role in recovery  
    
    </div>
    """, unsafe_allow_html=True)


    
   

# Initialize the app
def main():
    # Header with gradient background
    st.markdown("""
    <div style="background:#014d26; padding: 2rem; border-radius: 15px; margin-bottom: 2rem;">
        <h1 style="color: white; text-align: center; margin: 0;">🧠 Mental Health Treatment Predictor</h1>
        <p style="color: white; text-align: center; margin: 0;">Predict the likelihood of seeking mental health treatment</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load models
    mlp_model, label_encoders, gender_encoder, scaler, target_encoder, ohe_columns, ordinal_encoders = load_models()
    
    if mlp_model is None:
        st.error("⚠️ Models not loaded. Please ensure all .pkl files are in the same directory.")
        st.info("Required files: mlp_model.pkl, label_encoders.pkl, gender_encoder.pkl, scaler.pkl, target_encoder.pkl, ohe_columns.pkl, and ordinal encoder files")
        return
    
    # Create input form in sidebar with styled header
    st.sidebar.markdown("""
    <div style="background:linear-gradient(#1a75ff, #2e8b57); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
        <h3 style="color: white; text-align: center; margin: 0;">📝 User Information</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Personal Information
    st.sidebar.markdown("#### 👤 Personal Details")
    
    gender = st.sidebar.selectbox(
        "Gender",
        options=["Female", "Male"]
    )
    
    self_employed = st.sidebar.selectbox(
        "Are you self-employed?",
        options=["No", "Yes"]
    )
    
    # Family and Medical History
    st.sidebar.markdown("####  Medical History")
    
    family_history = st.sidebar.selectbox(
        "Family history of mental illness?",
        options=["No", "Yes"]
    )
    
    mental_health_history = st.sidebar.selectbox(
        "Personal mental health history?",
        options=["No", "Yes"]
    )
    
    # Behavioral Patterns
    st.sidebar.markdown("#### 📊 Behavioral Patterns")
    
    days_indoors = st.sidebar.selectbox(
        "How many days do you typically stay indoors?",
        options=["Go Out Every Day", "1-14 Days", "15-30 Days", "31-60 Days", "More Than 60 Days"]
    )
    
    mood_swings = st.sidebar.selectbox(
        "How would you rate your mood swings?",
        options=["Low", "Medium", "High"]
    )
    
    growing_stress = st.sidebar.selectbox(
        "Are you experiencing growing stress?",
        options=["No", "Maybe", "Yes"]
    )
    
    coping_struggles = st.sidebar.selectbox(
        "Do you struggle with coping?",
        options=["No", "Yes"]
    )
    
    # Work and Support
    st.sidebar.markdown("#### 💼 Work Environment & Support")
    
    occupation = st.sidebar.selectbox(
        "Occupation",
        options=["Corporate", "Business", "Others", "Student", "Housewife"]
    )
    
    care_options = st.sidebar.selectbox(
        "Do you know your care options?",
        options=["No", "Not Sure", "Yes"]
    )
    
    mental_health_interview = st.sidebar.selectbox(
        "Would you bring up mental health in an interview?",
        options=["No", "Maybe", "Yes"]
    )
    
    # Country selection
    country = st.sidebar.selectbox(
        "Country",
        options=["United States", "United Kingdom", "Canada", "Australia", 
                "Netherlands", "Ireland", "Poland", "Croatia", "Moldova", 
                "New Zealand", "South Africa", "Other"]
    )
    
    # Sentiment score (calculated based on responses)
    sentiment_compound = st.sidebar.slider(
        "Sentiment Score (Auto-calculated based on responses)",
        min_value=-1.0,
        max_value=1.0,
        value=0.0,
        step=0.01,
        help="This is typically calculated automatically based on text analysis"
    )
    
    # Prediction button with custom styling
    st.sidebar.markdown("""
    <div style="text-align: center; margin-top: 2rem;">
    """, unsafe_allow_html=True)
    
    if st.sidebar.button("🔮 Predict Treatment Likelihood", type="primary"):
        
        # Create input dataframe
        input_data = {
            'Gender': gender,
            'self_employed': self_employed,
            'family_history': family_history,
            'Mental_Health_History': mental_health_history,
            'Occupation': occupation,
            'Days_Indoors': days_indoors,
            'Mood_Swings': mood_swings,
            'care_options': care_options,
            'mental_health_interview': mental_health_interview,
            'Growing_Stress': growing_stress,
            'Coping_Struggles': coping_struggles,
            'Country_Grouped': country,
            'sentiment_compound': sentiment_compound
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        try:
            
            
            # Preprocessing pipeline
            processed_df = preprocess_input(input_df, label_encoders, gender_encoder, 
                                          ordinal_encoders, ohe_columns)
            
            
            
            # Scale the features
            processed_scaled = scaler.transform(processed_df)
            
            # Make prediction
            prediction = mlp_model.predict(processed_scaled)[0]
            prediction_proba = mlp_model.predict_proba(processed_scaled)[0]
            
            # Convert prediction back to original labels
            predicted_label = target_encoder.inverse_transform([prediction])[0]
            
   
            
            # Display results with styled header
            st.markdown("""
            <div style="background: #808000.; padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;">
                <h2 style="color:  #808000.; text-align: center; margin: 0;">📊 Prediction Results</h2>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                treatment_prob = prediction_proba[1]  # Probability of seeking treatment
                
                if treatment_prob > 0.5:
                    st.markdown("""
                    <div style="background:#014d26; padding: 1rem; border-radius: 10px; border: 2px solid #6F4E37;text-align: center;">
                        <h3 style="color: white; margin: 0;"> High likelihood of seeking treatment</h3>
                        <p style="color: #FFFDD0; font-size: 1rem; margin: 0.5rem 0 0 0;">Confidence: {:.2%}</p>
                    </div>
                    """.format(treatment_prob), unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background:#014d26; padding: 1rem; border-radius: 10px; border: 2px solid #6F4E37;text-align: center;">
                        <h3 style="color: white; margin: 0;">Low likelihood of seeking treatment</h3>
                        <p style="color: #FFFDD0; font-size: 1rem; margin: 0.5rem 0 0 0;">Confidence: {:.2%}</p>
                    </div>
                    """.format(1-treatment_prob), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div style="background:#3C4C24; padding: 1rem; border-radius: 10px; border: 2px solid #6F4E37;">
                """, unsafe_allow_html=True)
                st.metric("Treatment Probability", f"{prediction_proba[1]:.2%}")
                st.metric("No Treatment Probability", f"{prediction_proba[0]:.2%}")
                st.markdown("</div>", unsafe_allow_html=True)

           
          
            
           
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            st.error("Please check that all required model files are present and properly formatted.")

def preprocess_input(input_df, label_encoders, gender_encoder, ordinal_encoders, ohe_columns):
    """
    Preprocess input data to match the training data format
    """
    df = input_df.copy()
    
    # 1. Handle Gender encoding
    df['Gender'] = gender_encoder.transform(df['Gender'].astype(str))
    
    # 2. Handle binary label encoding
    binary_cols = ["self_employed", "family_history", "Mental_Health_History", "Coping_Struggles"]
    for col in binary_cols:
        if col in df.columns:
            # Check if label_encoders is a dict or single encoder
            if isinstance(label_encoders, dict):
                if col in label_encoders:
                    df[col] = label_encoders[col].transform(df[col].astype(str))
                else:
                    # Create simple binary encoding if encoder not found
                    df[col] = df[col].map({"No": 0, "Yes": 1})
            else:
                # If it's a single encoder, create simple binary encoding
                df[col] = df[col].map({"No": 0, "Yes": 1})
    
    # 3. Handle ordinal encoding
    ordinal_mapping = {
        'Mood_Swings': ['Low', 'Medium', 'High'],
        'Days_Indoors': ['Go Out Every Day', '1-14 Days', '15-30 Days', '31-60 Days', 'More Than 60 Days'],
        'care_options': ['No', 'Not Sure', 'Yes'],
        'mental_health_interview': ['No', 'Maybe', 'Yes'],
        'Growing_Stress': ['No', 'Maybe', 'Yes']
    }
    
    for col in ordinal_mapping.keys():
        if col in df.columns:
            # Clean the data
            df[col] = df[col].astype(str).str.strip().str.title()
            # Map to ordinal values
            df[col] = df[col].map({category: idx for idx, category in enumerate(ordinal_mapping[col])})
    
    # 4. Handle One-Hot Encoding for nominal features
    nominal_cols = ['Occupation', 'Country_Grouped']
    df_encoded = pd.get_dummies(df, columns=nominal_cols, drop_first=True)
    
    # 5. Ensure all columns from training are present
    for col in ohe_columns:
        if col not in df_encoded.columns and col != 'treatment':
            df_encoded[col] = 0
    
    # 6. Select only the columns that were in training (excluding target)
    feature_columns = [col for col in ohe_columns if col != 'treatment']
    df_encoded = df_encoded[feature_columns]
    
    return df_encoded

def show_about():
    st.markdown("""
    <div style="display: flex; justify-content: center; margin-bottom: 2rem;">
        <div style="background:#014d26; padding: 0.5rem; border-radius: 10px; 
                    max-width: 1000px; width: 90%; text-align: center;">
            <h2 style="color: white; margin: 0;">About This App</h2>
        </div>
    </div>
    """, 
    unsafe_allow_html=True)

    
    st.markdown("""
    This application uses a **Multi-Layer Perceptron (MLP) Neural Network** to predict 
    the likelihood of someone seeking mental health treatment based on various personal, 
    behavioral, and environmental factors.
    
    ### 🎯 Model Performance
    - **Algorithm**: Multi-Layer Perceptron (Neural Network)
    - **Accuracy**: 79%
    - **Architecture**: 2 hidden layers (100, 50 neurons)
    - **Features**: Personal info, behavioral patterns, work environment
    
    ### 📊 Key Features Analyzed
    - Personal demographics and employment status
    - Family and personal mental health history
    - Behavioral patterns (mood swings, stress levels, indoor habits)
    - Work environment and support systems
    - Geographic location
    
    ### 🏛️ Policy Recommendations
    - Promote **workplace mental health support systems** to reduce stigma and improve access to care.  
    - Develop **inclusive public health programs** considering gender and self-development as key indicators.  
    - Encourage **preventive measures** through awareness campaigns and early intervention strategies.  
    - Support **data-driven policy-making** by leveraging insights from predictive models.  
    
    ### 🤝 Engagement with the Mental Health Community
    - Foster collaboration between **researchers, mental health professionals, and data scientists**.  
    - Share insights with **NGOs and advocacy groups** working on mental health awareness.  
    - Engage with **workplaces and educational institutions** to apply findings in real-world contexts.  
    - Encourage **open dialogue and knowledge exchange** to continuously improve mental health solutions.  
    

    """)


    st.markdown("</div>", unsafe_allow_html=True)

# Main app layout
def app_layout():
    tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Dashboard", "ℹ️ About"])
    
    with tab1:
        main()
    
    with tab2:
        show_dashboard()
    
    with tab3:
        show_about()

if __name__ == "__main__":
    app_layout()