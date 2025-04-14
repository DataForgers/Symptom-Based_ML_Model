# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold, StratifiedShuffleSplit
from imblearn.over_sampling import SMOTE, RandomOverSampler
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, roc_auc_score
from sklearn.feature_selection import SelectKBest, chi2
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter

# Set page config
st.set_page_config(
    page_title="Diabetes Symptoms Detection",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4169E1;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4682B4;
    }
    .section {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>Diabetes Detection Web App</h1>", unsafe_allow_html=True)
st.markdown("This interactive web application analyzes patient symptom data and predicts the likelihood of diabetes using machine learning.")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('diabetes_symptoms_data.csv')
    cleaned_df = pd.read_csv('cleaned_symptoms_data.csv')
    return df, cleaned_df

# Function for EDA
def explore_data(df, cleaned_df):
    st.markdown("<h2 class='sub-header'>📊 Exploratory Data Analysis</h2>", unsafe_allow_html=True)
    
    # Show first few rows
    if st.checkbox("Show sample data"):
        st.write(df.head())
    
    # Basic info
    if st.checkbox("Dataset information"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Dataset shape:**", df.shape)
            st.write("**Missing values:**", df.isnull().sum().sum())
            
        with col2:
            # Class distribution
            class_counts = df['class'].value_counts()
            st.write("**Class distribution:**")
            st.write(f"Positive: {class_counts['Positive']} ({class_counts['Positive']/len(df):.1%})")
            st.write(f"Negative: {class_counts['Negative']} ({class_counts['Negative']/len(df):.1%})")
    
    # Gender distribution
    if st.checkbox("Gender distribution"):
        fig = px.pie(df, names='gender', title='Gender Distribution', 
                     color_discrete_sequence=px.colors.sequential.Blues)
        st.plotly_chart(fig)
    
    # Age distribution
    if st.checkbox("Age distribution"):
        fig = px.histogram(df, x="age", color="class", marginal="box", 
                           title="Age Distribution by Diabetes Status",
                           color_discrete_map={"Positive": "#1E90FF", "Negative": "#D3D3D3"})
        st.plotly_chart(fig)
    
    # Symptoms distribution
    if st.checkbox("Symptoms distribution"):
        # Create a melted dataframe for symptoms
        symptom_cols = ['polyuria', 'polydipsia', 'sudden_weight_loss', 'weakness', 
                         'polyphagia', 'genital_thrush', 'visual_blurring', 'itching', 
                         'irritability', 'delayed_healing', 'partial_paresis', 
                         'muscle_stiffness', 'alopecia', 'obesity']
        
        symptoms_by_class = pd.DataFrame()
        for symptom in symptom_cols:
            positive_yes = df[df['class'] == 'Positive'][symptom].value_counts().get('Yes', 0)
            positive_total = len(df[df['class'] == 'Positive'])
            positive_pct = positive_yes / positive_total * 100
            
            negative_yes = df[df['class'] == 'Negative'][symptom].value_counts().get('Yes', 0)
            negative_total = len(df[df['class'] == 'Negative'])
            negative_pct = negative_yes / negative_total * 100
            
            temp_df = pd.DataFrame({
                'Symptom': [symptom, symptom],
                'Class': ['Positive', 'Negative'],
                'Percentage': [positive_pct, negative_pct]
            })
            symptoms_by_class = pd.concat([symptoms_by_class, temp_df])
        
        fig = px.bar(symptoms_by_class, x='Symptom', y='Percentage', color='Class', barmode='group',
                     title='Percentage of "Yes" Responses by Symptom and Diabetes Status',
                     color_discrete_map={"Positive": "#1E90FF", "Negative": "#D3D3D3"})
        fig.update_layout(xaxis={'categoryorder':'total descending'})
        st.plotly_chart(fig)
    
    # Correlation heatmap
    if st.checkbox("Feature correlation"):
        # Convert Yes/No to 1/0 for correlation analysis
        df_encoded = df.copy()
        for col in df.select_dtypes(include=['object']).columns:
            if col != 'gender':  # Skip gender for now
                df_encoded[col] = df_encoded[col].map({'Yes': 1, 'No': 0})
        
        # One-hot encode gender
        df_encoded = pd.get_dummies(df_encoded, columns=['gender'], drop_first=True)
        
        # Map class to 1/0
        df_encoded['class'] = df_encoded['class'].map({'Positive': 1, 'Negative': 0})
        
        # Calculate correlation matrix
        corr = df_encoded.corr()
        
        # Plot heatmap
        fig = px.imshow(corr, text_auto=True, aspect="auto", 
                       title="Feature Correlation Heatmap",
                       color_continuous_scale='Blues')
        st.plotly_chart(fig)
    
    # Feature importance by chi-square test
    if st.checkbox("Feature importance"):
        # Prepare data for chi-square test
        X = cleaned_df.drop('class', axis=1)
        y = cleaned_df['class']
        
        # Select features with chi-square test
        selector = SelectKBest(chi2, k='all')
        selector.fit(X, y)
        feature_scores = pd.DataFrame({
            'Feature': X.columns,
            'Score': selector.scores_
        })
        
        # Sort by importance
        feature_scores = feature_scores.sort_values('Score', ascending=False)
        
        # Plot importance
        fig = px.bar(feature_scores, x='Feature', y='Score', 
                     title='Feature Importance (Chi-Square Test)',
                     color='Score', color_continuous_scale='Blues')
        fig.update_layout(xaxis={'categoryorder':'total descending'})
        st.plotly_chart(fig)

# Function for model building
def build_model(cleaned_df):
    st.markdown("<h2 class='sub-header'>🔍 Model Selection & Training</h2>", unsafe_allow_html=True)

    # Organize UI into two tabs
    tab1, tab2 = st.tabs(["🚀 Model Selection & Training", "📊 Evaluation"])

    with tab1:
        st.markdown("### Choose a Model and Train")
        model_option = st.radio("Select a Model:", ["🔵 Logistic Regression", "🌲 Random Forest"], index=None, horizontal=True)
        st.markdown("---")

        model = None  # Ensure model is always defined
        selected_features = ['age', 'gender', 'polyuria', 'polydipsia', 'sudden_weight_loss',
                             'polyphagia', 'irritability', 'partial_paresis']
        
        if model_option:
            # Store selected model in session state
            st.session_state['selected_model'] = model_option

            # Select relevant features
            X = cleaned_df[selected_features]
            y = cleaned_df["class"]

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Apply SMOTE on training data only
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

            # Scale age column (if it exists)
            if 'age' in X_train.columns:
                scaler = StandardScaler()
                X_train_resampled['age'] = scaler.fit_transform(X_train_resampled[['age']])
                X_test['age'] = scaler.transform(X_test[['age']])

            # Define the Model
            if model_option == "🔵 Logistic Regression":
                model = LogisticRegression(max_iter=1000, random_state=42)
            else:
                model = RandomForestClassifier(n_estimators=30, max_depth=3, min_samples_split=15, min_samples_leaf=8, max_features="sqrt", class_weight="balanced", random_state=42)

            # Train the model
            model.fit(X_train_resampled, y_train_resampled)

            # Store trained model and data in session state
            st.session_state['model_trained'] = True
            st.session_state['trained_model'] = model
            st.session_state['X_test'] = X_test
            st.session_state['y_test'] = y_test
            st.session_state['feature_names'] = X.columns  # Save feature names in session state

    with tab2:
        if 'model_trained' not in st.session_state or not st.session_state['model_trained']:
            st.warning("⚠️ Please select and train a model first!")
        else:
            st.markdown("### 📊 Evaluation Results")

            # Retrieve stored model and data
            model = st.session_state['trained_model']
            X_test = st.session_state['X_test']
            y_test = st.session_state['y_test']

            # Evaluate Performance
            y_pred = model.predict(X_test)
            test_acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)

            col1, col2 = st.columns([1, 1])

            with col1:
                st.metric("🎯 Test Accuracy", f"{test_acc:.4f}")
                st.markdown("---")
                cv_scores = cross_val_score(model, X_test, y_test, cv=10)
                st.metric("📊 Cross-Validation Accuracy", f"{cv_scores.mean():.4f}")
                st.markdown("---")
                st.write("**Classification Report:**")
                st.dataframe(pd.DataFrame(report).transpose().style.format('{:.4f}'))

            with col2:
                st.markdown("**Confusion Matrix:**")
                fig = px.imshow(cm, text_auto=True, aspect="auto", color_continuous_scale="Blues")
                fig.update_layout(width=350, height=350)
                st.plotly_chart(fig)

            # Feature Importance for Random Forest
            if st.session_state.get('selected_model') == "🌲 Random Forest":
                st.markdown("### 🔍 Feature Importance")
                feature_importance = pd.Series(model.feature_importances_, index=selected_features).sort_values(ascending=False)
                fig_importance = px.bar(x=feature_importance.index, y=feature_importance.values,
                                        labels={'x': 'Feature', 'y': 'Importance'},
                                        title="Feature Importance in Random Forest Model")
                st.plotly_chart(fig_importance)

    return (st.session_state['trained_model'], st.session_state['feature_names']) if 'trained_model' in st.session_state else (None, None) 



# Function for prediction
def make_prediction(model, feature_names):
    st.markdown("<h2 class='sub-header'>🔮 Diabetes Detection</h2>", unsafe_allow_html=True)
    st.write("Enter the patient's information to get a detection:")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Enter Age:", min_value=1, max_value=120, step=1, format="%d")
        gender = st.selectbox("Select Gender:", ["", "Male", "Female"])  # Placeholder for empty selection
        polyuria = st.selectbox("Do you experience Polyuria (Excessive Urination)?", ["", "Yes", "No"])
        polydipsia = st.selectbox("Do you experience Polydipsia (Excessive Thirst)?", ["", "Yes", "No"])
    
    with col2:
        sudden_weight_loss = st.selectbox("Have you had Sudden Weight Loss?", ["", "Yes", "No"])
        polyphagia = st.selectbox("Do you experience Polyphagia (Excessive Hunger)?", ["", "Yes", "No"])
        irritability = st.selectbox("Do you feel Irritability often?", ["", "Yes", "No"])
        partial_paresis = st.selectbox("Do you have Partial Paresis (Muscle Weakness)?", ["", "Yes", "No"])

    # Function to preprocess inputs
    def preprocess_inputs():
        if not age or not gender or not polyuria or not polydipsia or not sudden_weight_loss or not polyphagia or not irritability or not partial_paresis:
            st.warning("⚠️ Please fill in all fields before predicting.")
            return None

        gender_val = 1 if gender == "Male" else 0
        polyuria_val = 1 if polyuria == "Yes" else 0
        polydipsia_val = 1 if polydipsia == "Yes" else 0
        sudden_weight_loss_val = 1 if sudden_weight_loss == "Yes" else 0
        polyphagia_val = 1 if polyphagia == "Yes" else 0
        irritability_val = 1 if irritability == "Yes" else 0
        partial_paresis_val = 1 if partial_paresis == "Yes" else 0

        return np.array([[age / 120, gender_val, polyuria_val, polydipsia_val, sudden_weight_loss_val, 
                          polyphagia_val, irritability_val, partial_paresis_val]])


    def generate_pdf(diabetes_prob, risk_level, message, recommendations):
        # Create a byte stream to hold the PDF data
        buffer = io.BytesIO()
        
        # Create a canvas object for the PDF
        c = canvas.Canvas(buffer, pagesize=letter)
        
        # Set font
        c.setFont("Helvetica", 12)
        
        # Add content to the PDF
        c.drawString(100, 750, "Diabetes Risk Assessment Report")
        c.drawString(100, 730, f"Your Result: {message}")
        c.drawString(100, 710, f"Model Confidence: {diabetes_prob * 100:.2f}%")
        c.drawString(100, 690, f"Risk Level: {risk_level}")
        
        # Add the recommendations
        c.drawString(100, 670, "Recommendations:")
        y_position = 650
        for recommendation in recommendations:
            c.drawString(100, y_position, f"- {recommendation}")
            y_position -= 20
        
        # Save the PDF
        c.showPage()
        c.save()
        
        # Get PDF data and return it
        buffer.seek(0)
        return buffer

    # Prediction Button
    if st.button("🔍 Detect Diabetes"):
        input_data = preprocess_inputs()

        if input_data is not None:
            with st.spinner("Calculating..."):
                # Get Prediction Probability
                prediction_proba = model.predict_proba(input_data)
                diabetes_prob = prediction_proba[0][1]  # Probability of having diabetes
                
                # Determine risk category
                if diabetes_prob < 0.3:
                    risk_level = "Low Risk"
                    color = "#33FF57"
                    message = "✅ No Diabetes Detected!"
                elif 0.3 <= diabetes_prob < 0.5:
                    risk_level = "Moderate Risk"
                    color = "#FFD733"
                    message = "⚠️ Mild Diabetes (Borderline Case) - Monitor Your Health"
                elif 0.5 <= diabetes_prob < 0.8:
                    risk_level = "High Risk"
                    color = "#FF8C33"
                    message = "🚨 High Risk of Diabetes - Take Preventive Measures"
                else:
                    risk_level = "Very High Risk"
                    color = "#FF5733"
                    message = "🚨 Diabetes Detected - Consult a Doctor Immediately"

                # Display result
                st.markdown(f"<div style='background-color:#F0F8FF; padding:20px; border-radius:10px;'>", unsafe_allow_html=True)
                st.subheader("Prediction Result:")
                st.markdown(f"<h3 style='color:{color}'>{message}</h3>", unsafe_allow_html=True)
                #st.write(f"📊 **Model Confidence:** {diabetes_prob*100:.2f}%")
                st.markdown(f"<h4 style='color:{color}'>Risk Level: {risk_level}</h4>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

                # Displaying Gauge (Visual)
                fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=diabetes_prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Diabetes Risk Probability"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 40], 'color': "#c8e6c9"},
                        {'range': [40, 70], 'color': "#fff9c4"},
                        {'range': [70, 100], 'color': "#ffccbc"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': diabetes_prob * 100
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations Section
            st.subheader("Recommendations:")
            recommendations = []
            if diabetes_prob >= 0.5:
                st.error("🚨 High Risk of Diabetes - Take Preventive Measures")
                recommendations = [
                    "Consult a healthcare provider for a diabetes assessment.",
                    "Consider an HbA1c test and fasting blood glucose test.",
                    "Monitor blood sugar levels regularly.",
                    "Adopt a healthy diet and exercise routine."
                ]
            elif 0.3 <= diabetes_prob < 0.5:
                st.warning("⚠️ Mild Diabetes (Borderline Case) - Monitor Your Health")
                recommendations = [
                    "Adopt a Healthy Diet: Reduce sugar intake and eat fiber-rich foods.",
                    "Exercise Regularly: Engage in at least 30 minutes of moderate activity daily.",
                    "Monitor Blood Sugar: Keep track of glucose levels and consult a doctor if needed.",
                    "Stay Hydrated: Drink plenty of water to help with metabolism.",
                    "Manage Stress: Practice relaxation techniques like meditation or yoga."
                ]
            else:
                st.success("✅ No Diabetes Detected! Keep Up the Healthy Lifestyle")
                recommendations = [
                    "Maintain a healthy lifestyle.",
                    "Regular check-ups with a healthcare provider.",
                    "Stay aware of diabetes symptoms and risk factors."
                ]

            # Provide the PDF download button outside the condition
            st.subheader("Generate PDF Report:")
            pdf_buffer = generate_pdf(diabetes_prob, risk_level, message, recommendations)
            st.download_button(
                label="Download PDF Report",
                data=pdf_buffer,
                file_name="detection_report.pdf",
                mime="application/pdf"
            )

        # Disclaimer Section
        st.info("Disclaimer: This prediction is based on a machine learning model and should not be considered as medical advice. Always consult with a healthcare provider.")

                # Disclaimer
            #st.info("Disclaimer: This prediction is based on a machine learning model and should not be considered as medical advice. Always consult with a healthcare professional for proper diagnosis and treatment.")
# Information section
def show_info():
    st.markdown("<h2 class='sub-header'>ℹ️ About Diabetes</h2>", unsafe_allow_html=True)
    
    #tabs = st.tabs(["What is Diabetes?", "Risk Factors", "Symptoms", "Prevention"])
    
    edu_tab1, edu_tab2, edu_tab3, edu_tab4 = st.tabs(["What is Diabetes?", "Symptoms", "Risk Factors", "Prevention"])
            
    with edu_tab1:
                st.write("""
                **Diabetes mellitus** is a chronic condition that affects how your body processes blood sugar (glucose).
                
                There are several types of diabetes:
                
                - **Type 1 Diabetes:** The immune system attacks and destroys insulin-producing cells in the pancreas.
                - **Type 2 Diabetes:** The body becomes resistant to insulin or doesn't produce enough insulin.
                - **Prediabetes:** Blood sugar levels are higher than normal but not high enough to be classified as diabetes.
                - **Gestational Diabetes:** Develops during pregnancy and usually resolves after delivery.
                
                Common symptoms include increased thirst, frequent urination, extreme hunger, unexplained weight loss,
                fatigue, irritability, blurred vision, and slow-healing sores.
                """)

    with edu_tab2:
        st.write("""
        **Common symptoms of diabetes include:**
        - Polyuria (frequent urination)
        - Polydipsia (excessive thirst)
        - Polyphagia (excessive hunger)
        - Unexpected weight loss
        - Fatigue
        - Blurred vision
        - Slow-healing wounds
        - Frequent infections
        - Tingling or numbness in hands or feet
        
        Note that many people with Type 2 diabetes may not experience symptoms for years.
        """)
            
    with edu_tab3:
                st.write("""
                Several factors can increase the risk of developing diabetes:
                
                **For Type 1 Diabetes:**
                - Family history and genetics
                - Environmental factors
                - Presence of certain autoimmune conditions
                
                **For Type 2 Diabetes:**
                - Excess weight and obesity
                - Physical inactivity
                - Family history
                - Age (risk increases with age)
                - Race/ethnicity (higher risk in certain populations)
                - High blood pressure
                - Abnormal cholesterol levels
                - History of gestational diabetes
                - Polycystic ovary syndrome
                """)
                
                # Create risk factor visualization
                risk_data = {
                    'Factor': ['Obesity', 'Physical Inactivity', 'Family History', 'Age (45+)', 'High Blood Pressure'],
                    'Relative Risk': [3.0, 2.2, 2.5, 1.8, 2.0]
                }
                
                risk_df = pd.DataFrame(risk_data)
                
                fig = px.bar(risk_df, x='Factor', y='Relative Risk', 
                            title='Relative Risk Factors for Type 2 Diabetes',
                            color='Relative Risk',
                            color_continuous_scale=px.colors.sequential.Reds)
                
                st.plotly_chart(fig, use_container_width=True)
            
    with edu_tab4:
                st.write("""
                While some risk factors like genetics cannot be changed, many lifestyle modifications can help prevent or delay diabetes:
                
                - **Maintain a healthy weight:** Losing even 5-7% of body weight can significantly reduce risk
                - **Regular physical activity:** Aim for at least 150 minutes of moderate exercise weekly
                - **Healthy diet:** Focus on fruits, vegetables, whole grains, lean proteins, and limit processed foods
                - **Avoid smoking:** Smoking increases diabetes risk and complications
                - **Limit alcohol consumption:** Excessive alcohol can increase blood sugar levels
                - **Regular check-ups:** Early detection can help manage prediabetes before it progresses
                """)
                
                # Prevention effectiveness chart
                prevention_data = {
                    'Method': ['Weight Loss (5-7%)', 'Regular Exercise', 'Healthy Diet', 'Smoking Cessation', 'Regular Screening'],
                    'Effectiveness': [58, 50, 35, 30, 25]
                }
                
                prevention_df = pd.DataFrame(prevention_data)
                
                fig = px.bar(prevention_df, x='Method', y='Effectiveness', 
                            title='Effectiveness of Prevention Methods (%)',
                            color='Effectiveness',
                            color_continuous_scale=px.colors.sequential.Greens)
                
                st.plotly_chart(fig, use_container_width=True)
            
            #st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# Main function

if 'page' not in st.session_state:
    st.session_state['page'] = "Home"
def main():
    # Sidebar
    st.sidebar.image("https://img.icons8.com/color/96/000000/diabetes.png", width=100)
    st.sidebar.title("Navigation")
    selected_page = st.sidebar.radio("Go to", ["Home", "Data Exploration", "Model Training", "Make Prediction", "About Diabetes"], 
                                 index=["Home", "Data Exploration", "Model Training", "Make Prediction", "About Diabetes"].index(st.session_state['page']))
    st.session_state['page'] = selected_page
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Note:** This app is for educational purposes only and should not be used as a substitute for professional medical advice.
    """)
    
    # Load data
    df, cleaned_df = load_data()
    
    # Home page
    if st.session_state['page'] == "Home":
        st.image("https://img.icons8.com/color/96/000000/diabetes.png", width=150)
        #st.markdown("<h1 class='main-header'>Welcome to the Diabetes Detection Web App</h1>", unsafe_allow_html=True)
        
        st.write("""
        
        ### Features:
        - **Data Exploration**: Visualize and understand the diabetes symptoms dataset
        - **Model Training**: Train and evaluate machine learning models for diabetes prediction
        - **Make Prediction**: Input patient information to get a diabetes risk assessment
        - **About Diabetes**: Learn about diabetes, its symptoms, risk factors, and prevention strategies
        
        ### How to use:
        1. Navigate through the app using the sidebar
        2. Explore the data to understand diabetes risk factors
        3. Train a model to see how accurately we can predict diabetes
        4. Input patient information to get a prediction
        
        ### Dataset:
        The application uses a dataset containing information about various diabetes symptoms and their correlation with diabetes diagnosis.
        """)
        
        st.markdown("---")
        st.markdown("<h3>Quick Statistics</h3>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Diabetes Positive", len(df[df['class'] == 'Positive']))
        with col3:
            st.metric("Diabetes Negative", len(df[df['class'] == 'Negative']))
    
    # Data Exploration page
    elif selected_page == "Data Exploration":
        explore_data(df, cleaned_df)
    
    # Model Training page
    elif selected_page == "Model Training":
        model, feature_names = build_model(cleaned_df)
        # Save the model and feature names in session state
        st.session_state['model'] = model
        st.session_state['feature_names'] = feature_names
    
    # Make Prediction page
    elif selected_page == "Make Prediction":
        if 'model' not in st.session_state:
            st.warning("Please train a model first on the 'Model Training' page.")
            if st.button("Go to Model Training"):
                st.session_state['page'] = "Model Training"
                st.experimental_rerun()
        else:
            make_prediction(st.session_state['model'], st.session_state['feature_names'])
    
    # About Diabetes page
    elif selected_page == "About Diabetes":
        show_info()
        
    # Add custom CSS
    st.markdown("""
    <style>
    .main-header {
        color: #2E86C1;
        font-size: 42px;
    }
    .sub-header {
        color: #3498DB;
    }
    </style>
    """, unsafe_allow_html=True)

# Run the application
if __name__ == "__main__":
    main()
