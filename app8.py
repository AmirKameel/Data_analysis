import streamlit as st
import pandas as pd
import numpy as np
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from clarifai.client.model import Model

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = "data_analysis"
if 'df' not in st.session_state:
    st.session_state.df = None
if 'feature_columns' not in st.session_state:
    st.session_state.feature_columns = []
if 'target_column' not in st.session_state:
    st.session_state.target_column = None

# Title of the app
st.title("Advanced Data Analysis Project 📊")

# Navigation sidebar
page = st.sidebar.radio(
    "Choose a page:", 
    ["**Data Analysis 📊**", "**Data Visualization 📈**", "**Machine Learning 🤖**", "**Chat with Data 🧑‍💻⚡**"], 
    index=0
)

# Choose problem type
problem_type = st.sidebar.selectbox("Select Problem Type", ["Regression", "Classification"])


if page == "**Data Analysis 📊**":
    st.session_state.page = "data_analysis"
    # Upload dataset
    uploaded_file = st.file_uploader("Upload Dataset", type=["csv", "xlsx"])

    if uploaded_file is not None:
        st.session_state.df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')

    # Display dataset information
    if st.session_state.df is not None:
        st.subheader("Dataset Information")
        st.write(st.session_state.df.describe())

        # Select features and target column
        feature_columns = st.multiselect("Select Feature Columns", st.session_state.df.columns.tolist())
        target_column = st.selectbox("Select Target Column", st.session_state.df.columns.tolist())

        # Store selected feature columns and target column in session state
        st.session_state.feature_columns = feature_columns
        st.session_state.target_column = target_column

        # Reorder columns with target column last
        columns_to_display = feature_columns + [target_column]
        if st.session_state.df is not None:
            st.session_state.df = st.session_state.df[columns_to_display]

        st.dataframe(st.session_state.df.head())

        # Data Preprocessing
        st.subheader("Data Preprocessing")

        # Separate columns by data type
        text_columns = st.session_state.df.select_dtypes(include=['object']).columns.tolist()
        numerical_columns = st.session_state.df.select_dtypes(include=['float64', 'int64']).columns.tolist()

        # Display report before preprocessing
        if st.button("Generate Preprocessing Report"):
            st.session_state.report_generated = True
            st.session_state.report_data = []

            for col in st.session_state.df.columns:
                data_type = st.session_state.df[col].dtype
                null_count = st.session_state.df[col].isnull().sum()
                unique_values = st.session_state.df[col].nunique()
                duplicate_count = st.session_state.df.duplicated(subset=[col]).sum()

                if data_type in ['float64', 'int64']:
                    outliers = len(st.session_state.df[(np.abs(st.session_state.df[col] - st.session_state.df[col].mean()) > 3 * st.session_state.df[col].std())])
                else:
                    outliers = ''

                # Append dictionary to the list
                st.session_state.report_data.append({'Column': col, 'Data Type': data_type, 'Null Values': null_count, 
                                    'Unique Values': unique_values, 'Duplicates': duplicate_count, 'Outliers': outliers})

            # Convert list of dictionaries into DataFrame
            report_df = pd.DataFrame(st.session_state.report_data)

            st.write(report_df)

            # Download report as CSV
            csv_file = report_df.to_csv(index=False)
            b64 = base64.b64encode(csv_file.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="preprocessing_report.csv">Download Preprocessing Report</a>'
            st.markdown(href, unsafe_allow_html=True)

        # If report generated, display preprocessing options
        if getattr(st.session_state, 'report_generated', False):
            # Text Columns Preprocessing
            if text_columns:
                st.subheader("Text Columns Preprocessing")
                selected_text_columns = st.multiselect("Select Text Columns", text_columns)

                if st.button("Remove Null Values"):
                    for column in selected_text_columns:
                        st.session_state.df[column].dropna(inplace=True)
                    st.write(f"Removed null values for selected text columns. Data shape: {st.session_state.df.shape}")
                    st.dataframe(st.session_state.df.head())

                if st.button("Remove Duplicates"):
                    for column in selected_text_columns:
                        st.session_state.df.drop_duplicates(subset=[column], inplace=True)
                    st.write(f"Removed duplicates for selected text columns. Data shape: {st.session_state.df.shape}")
                    st.dataframe(st.session_state.df.head())

                encoding_option = st.selectbox("Select Encoding Method", ["Label Encoding", "One-Hot Encoding"])
                if encoding_option == "Label Encoding":
                    for column in selected_text_columns:
                        le = LabelEncoder()
                        st.session_state.df[column] = le.fit_transform(st.session_state.df[column])
                    st.write("Applied Label Encoding on selected text columns.")
                    st.dataframe(st.session_state.df.head())
                elif encoding_option == "One-Hot Encoding":
                    for column in selected_text_columns:
                        encoder = OneHotEncoder(sparse=False, drop="first")
                        encoded_cols = pd.DataFrame(encoder.fit_transform(st.session_state.df[[column]]), columns=encoder.get_feature_names_out([column]))
                        st.session_state.df = pd.concat([st.session_state.df, encoded_cols], axis=1)
                        st.session_state.df.drop(columns=[column], inplace=True)
                    st.write("Applied One-Hot Encoding on selected text columns.")
                    st.dataframe(st.session_state.df.head())

            # Numerical Columns Preprocessing
            if numerical_columns:
                st.subheader("Numerical Columns Preprocessing")
                selected_numerical_columns = st.multiselect("Select Numerical Columns", numerical_columns)

                if st.button("Standardization"):
                    for column in selected_numerical_columns:
                        scaler = StandardScaler()
                        st.session_state.df[column] = scaler.fit_transform(st.session_state.df[[column]])
                    st.write("Applied Standardization on selected numerical columns.")
                    st.dataframe(st.session_state.df.head())

                if st.button("Normalization"):
                    for column in selected_numerical_columns:
                        st.session_state.df[column] = (st.session_state.df[column] - st.session_state.df[column].min()) / (st.session_state.df[column].max() - st.session_state.df[column].min())
                    st.write("Applied Normalization on selected numerical columns.")
                    st.dataframe(st.session_state.df.head())

        # Download processed data
        if getattr(st.session_state, 'report_generated', False):
            csv_file = st.session_state.df.to_csv(index=False)
            b64 = base64.b64encode(csv_file.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="processed_data.csv">Download Processed Data</a>'
            st.markdown(href, unsafe_allow_html=True)


elif page == "**Data Visualization 📈**":
    st.session_state.page = "data_visualization"

    st.subheader("Data Visualization")

    if st.session_state.df is not None:  # Check if df is loaded
        # Correlation Heatmap
        st.write("**Correlation Heatmap**")
        corr = st.session_state.df.corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        st.pyplot(plt)

        # Pairplot for selected features
        if st.session_state.feature_columns:
            if len(st.session_state.feature_columns) > 1:
                st.write("**Pairplot for Selected Features**")
                pairplot = sns.pairplot(st.session_state.df[st.session_state.feature_columns])
                st.pyplot(pairplot)
    else:
        st.write("Please upload a dataset in the Data Analysis page.")
        


elif page == "**Machine Learning 🤖**":
    st.session_state.page = "machine_learning"

    st.subheader("Machine Learning")

    if st.session_state.df is not None:  
        # Choose machine learning model
        if problem_type == "Regression":
            st.write("**Regression Models**")
            model_option = st.selectbox("Select Regression Model", ["Linear Regression", "Random Forest Regression"])

            if model_option == "Linear Regression":
                st.write("**Linear Regression**")
                model = LinearRegression()
            elif model_option == "Random Forest Regression":
                st.write("**Random Forest Regression**")
                model = RandomForestRegressor()

        elif problem_type == "Classification":
            st.write("**Classification Models**")
            model_option = st.selectbox("Select Classification Model", ["Logistic Regression", "Random Forest Classification"])

            if model_option == "Logistic Regression":
                st.write("**Logistic Regression**")
                model = LogisticRegression()
            elif model_option == "Random Forest Classification":
                st.write("**Random Forest Classification**")
                model = RandomForestClassifier()

        # Train-test split
        X = st.session_state.df[st.session_state.feature_columns]
        y = st.session_state.df[st.session_state.target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model training
        model.fit(X_train, y_train)

        # Model evaluation
        if problem_type == "Regression":
            y_pred = model.predict(X_test)
            st.write("**Regression Metrics**")
            metrics_df = pd.DataFrame({
                "Metric": ["Mean Squared Error", "Mean Absolute Error", "R-squared Score"],
                "Value": [mean_squared_error(y_test, y_pred), mean_absolute_error(y_test, y_pred), r2_score(y_test, y_pred)]
            })
            st.table(metrics_df)
        elif problem_type == "Classification":
            y_pred = model.predict(X_test)
            st.write("**Classification Metrics**")
            metrics_df = pd.DataFrame({
                "Metric": ["Accuracy Score"],
                "Value": [accuracy_score(y_test, y_pred)]
            })
            st.table(metrics_df)

    else:
        st.write("Please upload a dataset in the Data Analysis page.")



elif page == "**Chat with Data 🧑‍💻⚡**":
    

    # Chat with Data using Clarifai
    st.subheader("Chat with Data using Clarifai")
    
    prompt = st.text_input("Ask something about the data")
    
    if st.button("Ask"):
        model_url = "https://clarifai.com/openai/chat-completion/models/GPT-3_5-turbo"
        model_prediction = Model(url=model_url, pat="a859318378284560beec23442a19ba57").predict_by_bytes(prompt.encode(), input_type="text")
        
        st.write(model_prediction.outputs[0].data.text.raw)
