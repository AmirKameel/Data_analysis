import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import base64
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from clarifai.client.model import Model


# Title of the app
st.title("Data Analysis Project 📊")

# Navigation sidebar
page = st.sidebar.radio(
    "Choose a page:", 
    ["**Data Analysis 📊**", "**Chat with Data 🧑‍💻⚡**"], 
    index=1
)

# Choose problem type
problem_type = st.sidebar.selectbox("Select Problem Type", ["Regression", "Classification"])

if page == "**Data Analysis 📊**":
    # Upload dataset
    uploaded_file = st.file_uploader("Upload Dataset", type=["csv", "xlsx"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')

        # Display dataset information
        st.subheader("Dataset Information")
        st.write(df.describe())

        # Select features and target column
        feature_columns = st.multiselect("Select Feature Columns", df.columns.tolist())
        target_column = st.selectbox("Select Target Column", df.columns.tolist())

        # Reorder columns with target column last
        columns_to_display = feature_columns + [target_column]
        df = df[columns_to_display]

        st.dataframe(df.head())

        # Data Preprocessing
        st.subheader("Data Preprocessing")

        # Separate columns by data type
        text_columns = df.select_dtypes(include=['object']).columns.tolist()
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

        # Text Columns Preprocessing
        if text_columns:
            st.subheader("Text Columns Preprocessing")
            selected_text_columns = st.multiselect("Select Text Columns", text_columns)

            for column in selected_text_columns:
                st.write(f"**{column}**")

                # Handle missing values
                if st.button(f"Remove Null Values for {column}"):
                    df[column].dropna(inplace=True)
                    st.write(f"Removed null values. Data shape: {df.shape}")
                    st.dataframe(df.head())

                # Remove duplicates
                if st.button(f"Remove Duplicates for {column}"):
                    df.drop_duplicates(subset=[column], inplace=True)
                    st.write(f"Removed duplicates. Data shape: {df.shape}")
                    st.dataframe(df.head())

                # Encoding categorical features
                encoding_option = st.selectbox(f"Select Encoding Method for {column}", ["Label Encoding", "One-Hot Encoding"])
                if encoding_option == "Label Encoding":
                    le = LabelEncoder()
                    df[column] = le.fit_transform(df[column])
                    st.write(f"Applied Label Encoding on {column}.")
                    st.dataframe(df.head())
                elif encoding_option == "One-Hot Encoding":
                    encoder = OneHotEncoder(sparse=False, drop="first")
                    encoded_cols = pd.DataFrame(encoder.fit_transform(df[[column]]), columns=encoder.get_feature_names_out([column]))
                    df = pd.concat([df, encoded_cols], axis=1)
                    df.drop(columns=[column], inplace=True)
                    st.write(f"Applied One-Hot Encoding on {column}.")
                    st.dataframe(df.head())

        # Numerical Columns Preprocessing
        if numerical_columns:
            st.subheader("Numerical Columns Preprocessing")
            selected_numerical_columns = st.multiselect("Select Numerical Columns", numerical_columns)

            processed_dfs = {}  # Dictionary to store processed DataFrames for each numerical column

            for column in selected_numerical_columns:
                st.write(f"**{column}**")

                # Create a temporary copy of the original DataFrame
                temp_df = df.copy()

                # Standardization
                if st.button(f"Standardization for {column}"):
                    scaler = StandardScaler()
                    temp_df[column] = scaler.fit_transform(temp_df[[column]])
                    st.write(f"Applied Standardization on {column}.")
                    st.dataframe(temp_df.head())
                    processed_dfs[column] = temp_df

                # Normalization
                if st.button(f"Normalization for {column}"):
                    temp_df[column] = (temp_df[column] - temp_df[column].min()) / (temp_df[column].max() - temp_df[column].min())
                    st.write(f"Applied Normalization on {column}.")
                    st.dataframe(temp_df.head())
                    processed_dfs[column] = temp_df

            # Update the original DataFrame with the processed results
            for col, processed_df in processed_dfs.items():
                df[col] = processed_df[col]

        # Download processed data
        csv_file = df.to_csv(index=False)
        b64 = base64.b64encode(csv_file.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="processed_data.csv">Download Processed Data</a>'
        st.markdown(href, unsafe_allow_html=True)

        # Model Selection
        st.subheader("Model Selection")

        X = df.drop(columns=[target_column])
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if st.button("Train and Evaluate"):
            if problem_type == "Regression":
                model_options = ['Linear Regression', 'Random Forest']
                selected_model = st.selectbox("Select Model", model_options)

                if selected_model == 'Linear Regression':
                    model = LinearRegression()
                elif selected_model == 'Random Forest':
                    model = RandomForestRegressor()

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                metrics = {
                    "Mean Squared Error (MSE)": mean_squared_error(y_test, y_pred),
                    "Root Mean Squared Error (RMSE)": mean_squared_error(y_test, y_pred, squared=False),
                    "Mean Absolute Error (MAE)": mean_absolute_error(y_test, y_pred),
                    "R-squared (R2)": r2_score(y_test, y_pred)
                }
                st.write(pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value']))

            elif problem_type == "Classification":
                model_options = ['Logistic Regression', 'Random Forest']
                selected_model = st.selectbox("Select Model", model_options)

                if selected_model == 'Logistic Regression':
                    model = LogisticRegression()
                elif selected_model == 'Random Forest':
                    model = RandomForestClassifier()

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                metrics = {
                    "Accuracy Score": accuracy_score(y_test, y_pred),
                    # Add more classification metrics as needed
                }
                st.write(pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value']))

elif page == "**Chat with Data 🧑‍💻⚡**":
    # Chat with Data using Clarifai
    st.subheader("Chat with Data using Clarifai")
    
    prompt = st.text_input("Ask something about the data")
    
    if st.button("Ask"):
        model_url = "https://clarifai.com/openai/chat-completion/models/GPT-3_5-turbo"
        model_prediction = Model(url=model_url, pat="a859318378284560beec23442a19ba57").predict_by_bytes(prompt.encode(), input_type="text")
        
        st.write(model_prediction.outputs[0].data.text.raw)
