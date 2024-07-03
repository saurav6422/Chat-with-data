import os
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import pipeline
from PIL import Image
import base64
from io import BytesIO
import torch
from transformers import BertForSequenceClassification, BertTokenizer,AutoTokenizer
from sklearn.preprocessing import LabelEncoder
import re

expander_content = """

1. **Mathematical Operations:**
   - **Scaling Values:** `lambda x: x * 2`
     - This doubles the values in the column.
   - **Add a Constant:** `lambda x: x + 10`
     - This adds 10 to each value in the column.
   - **Square Values:** `lambda x: x ** 2`
     - This squares each value in the column.

2. **String Operations:**
   - **Uppercase:** `lambda x: x.upper() if isinstance(x, str) else x`
     - This converts string values to uppercase.
   - **Substring Extraction:** `lambda x: x[:5] if isinstance(x, str) else x`
     - This extracts the first 5 characters of string values.

3. **Conditional Operations:**
   - **Conditional Replacement:** `lambda x: 'High' if x > 50 else 'Low'`
     - This categorizes values into 'High' or 'Low' based on a threshold.
   - **Replace Missing Values:** `lambda x: x if pd.notnull(x) else 'Missing'`
     - This replaces missing values with the string 'Missing'.

4. **Date Operations:**
   - **Extract Year:** `lambda x: x.year if isinstance(x, pd.Timestamp) else x`
     - This extracts the year from datetime values.
   - **Day of Week:** `lambda x: x.day_name() if isinstance(x, pd.Timestamp) else x`
     - This gets the day of the week from datetime values.

5. **Custom Aggregations:**
   - **Cumulative Sum:** `lambda x: x.cumsum()`
     - This calculates the cumulative sum of the column.
   - **Rolling Average:** `lambda x: x.rolling(window=3).mean()`
     - This calculates the rolling average with a window of 3.
"""
expander_content2 = """

This Streamlit application provides a chatbot interface for interacting with datasets. Users can upload their data files (CSV or Excel) and interact with the data using natural language queries.

**Features:**
- **File Upload**: Upload CSV or Excel files containing your dataset.
- **Natural Language Queries**: Use natural language to ask questions or perform operations on the data.
- **Data Exploration**: View summary statistics, column information, and sample data.
- **Filtering**: Filter the dataset based on specific column values.
- **Data Visualization**: Create various plot types (scatter, line, histogram, box, bar, pie) to visualize the data.
- **Aggregations**: Calculate maximum, minimum, average, and sum values for columns.
- **Value Counting**: Count the occurrences of specific values in a column.
- **Missing Value Analysis**: Identify missing values in the dataset.
- **Unique Value Retrieval**: Retrieve unique values in a column.
- **Column Operations**: Add, drop, or rename columns in the dataset.
- **Grouping**: Group the data by a specific column.
- **Heatmap Generation**: Generate a heatmap to visualize correlations between numeric columns.
- **Data Concatenation**: Concatenate multiple datasets vertically.
- **Data Joining**: Join datasets based on a common column.
- **Rolling Window Calculations**: Apply rolling window calculations on a column.
- **Lambda Function Application**: Apply custom Python functions to columns.
"""
df = None
def load_model_and_tokenizer():
    model_path = 'model_3'
    #tokenizer_path = 'tokenizer_3'
    csv_path = 'dataset/queries.csv'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    label_encoder = load_label_encoder(csv_path)

    model.to(device)
    return model, tokenizer, label_encoder, device

def load_label_encoder(csv_path):
    data = pd.read_csv(csv_path)
    label_encoder = LabelEncoder()
    label_encoder.fit(data['action'])
    return label_encoder

def predict_query(model, tokenizer, label_encoder, query, device):
    model.eval()
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    predicted = torch.argmax(outputs.logits, dim=1).item()
    return label_encoder.inverse_transform([predicted])[0]

def get_image_base64(image_path):
    img = Image.open(image_path)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

background_base64 = get_image_base64("assets/background.png")
logo_base64 = get_image_base64("assets/logo.png")
photo_base64 = get_image_base64("assets/bot.png")

CSS_SELECTOR = '.stApp'

st.markdown(
    f"""
    <style>
    {CSS_SELECTOR} {{
        background-image: url("data:image/png;base64,{background_base64}")!important;
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
        z-index: -1; 
    }}
    .background-overlay {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.75); 
        z-index: -1; 
    }}
    .title-container {{
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 20px;
        color: white;
        text-align: center;
    }}
    .title-container img {{
        width: 350px;
        margin-right: 20px;
    }}
    .title-container h1 {{
        margin-left: 20px; 
    }}
    .expander-container {{
        background-color: #262630;
        padding: 20px;
        border-radius: 5px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    f"""
    <div class="background-overlay"></div>
    <div class="title-container">
        <img src="data:image/png;base64,{logo_base64}" alt="Company Logo">
        <h1>ECL Dataset Chatbot</h1>
    </div>
    """,
    unsafe_allow_html=True
)

with st.expander("About"):
    st.markdown(
        f"""
        <div class="expander-container">
        {expander_content2}
        """,
        unsafe_allow_html=True
    )
uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx'])

def remove_spaces_from_columns(df):
    return df.rename(columns=lambda x: x.replace(' ', '_'))

def load_dataset(file):
    if file.type == "text/csv":
        return pd.read_csv(file)
    elif file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        return pd.read_excel(file)
    else:
        return None
if uploaded_file is not None:
    try:
        df = load_dataset(uploaded_file)
        st.write("Dataset loaded successfully")
        if any(' ' in col for col in df.columns):
            st.warning("WARNING : Some column names contain spaces. Spaces are removed from column name for easier querying.")
            df_no_spaces = remove_spaces_from_columns(df)
            csv_no_spaces = df_no_spaces.to_csv(index=False)
            st.download_button(
                label="Download CSV with spaces removed from column names",
                data=csv_no_spaces,
                file_name="dataset_no_spaces.csv",
                mime="text/csv",
            )
            df = df_no_spaces
        
        with st.expander("🔎 Dataframe Preview"):
            preview_rows = st.slider("Number of rows to display", min_value=1, max_value=len(df), value=5)
            st.dataframe(df.head(preview_rows))
    except Exception as e:
        st.write(f"Error loading dataset: {e}")

model, tokenizer, label_encoder, device = load_model_and_tokenizer()

b1, b2 = st.columns([5, 1])
with b1:
    query = st.text_area("🗣️ Chat with Data", key="query_input")
with b2:
    st.markdown("<br><br>", unsafe_allow_html=True)
    submit_button = st.button("Enter", key="submit")

def rename_column(dataframe, old_col_name, new_col_name):
    dataframe.rename(columns={old_col_name: new_col_name}, inplace=True)
    return dataframe

def extractor(query, df):
    col1, col2, numeric_value = None, None, None
    numeric_match = re.search(r'\d+', query)
    if numeric_match:
        numeric_value = numeric_match.group()
    
    words = query.split()
    matched_columns = []
    matched_columns = [word for word in words if word in df.columns]
    
    if matched_columns:
        col1 = matched_columns[0]
        if len(matched_columns) > 1:
            col2 = matched_columns[1]
    
    return col1, col2, numeric_value

def ex(query, df, col1):
    if col1 is None:
        return None
    
    unique_values = df[col1].unique().tolist()
    unique_values_lower = [str(val).lower() for val in unique_values]
    for value in sorted(unique_values, key=lambda x: len(str(x)), reverse=True):
        if str(value).lower() in query.lower():
            return value
    words = query.split()
    phrases = []
    for i in range(len(words)):
        for j in range(i + 1, len(words) + 1):
            phrases.append(' '.join(words[i:j]))
    for phrase in sorted(phrases, key=len, reverse=True):
        if phrase.lower() in unique_values_lower:
            return unique_values[unique_values_lower.index(phrase.lower())]
    for word in words:
        if word.replace(',', '').isdigit():
            numeric_value = int(word.replace(',', ''))
            if numeric_value in unique_values:
                return numeric_value
    
    return None

def create_plot(dataframe, plot_type, x_col, y_col=None):
    plt.figure(figsize=(10, 6))
    if x_col is None:
        st.markdown(f"<p style='font-size: 20px; text-align: center;'>No matching column found for '{x_col}' in the dataset.</p>", unsafe_allow_html=True)
        return
    
    if y_col:
        if y_col is None:
            st.markdown(f"<p style='font-size: 20px; text-align: center;'>No matching column found for '{y_col}' in the dataset.</p>", unsafe_allow_html=True)
            return

    if plot_type == "scatter":
        sns.scatterplot(data=dataframe, x=x_col, y=y_col)
    elif plot_type == "line":
        sns.lineplot(data=dataframe, x=x_col, y=y_col)
    elif plot_type == "histogram":
        sns.histplot(data=dataframe[x_col])
    elif plot_type == "box":
        sns.boxplot(data=dataframe, x=x_col, y=y_col)
    elif plot_type == "bar":
        sns.barplot(data=dataframe, x=x_col, y=y_col)
    elif plot_type == "pie":
        pie_data = dataframe[x_col].value_counts()
        plt.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%')
    
    plt.title(f"{plot_type.capitalize()} Plot of {x_col}" + (f" and {y_col}" if y_col else ""))
    plt.xlabel(x_col)
    if y_col:
        plt.ylabel(y_col)
    
    plot_buffer = BytesIO()
    plt.savefig(plot_buffer, format='png')
    plot_buffer.seek(0)
    plot_base64 = base64.b64encode(plot_buffer.getvalue()).decode('utf-8')
    st.pyplot(plt)
    st.download_button(
        label="Download Plot",
        data=plot_base64,
        file_name=f"{plot_type}_plot.png",
        mime="image/png"
    )

    st.markdown(f"<p style='font-size: 20px; text-align: center;'>Plot created using column(s): {x_col}" + (f" and {y_col}" if y_col else "") + "</p>", unsafe_allow_html=True)


def apply_filter(df, column, value):
    is_numeric = pd.api.types.is_numeric_dtype(df[column])

    if value.startswith(('>', '<', '=')):
        op = value[0]
        val = value[1:].strip()
        try:
            if is_numeric:
                val = pd.to_numeric(val.replace(',', ''))
                if op == '>':
                    return df[df[column] > val]
                elif op == '<':
                    return df[df[column] < val]
                elif op == '=':
                    return df[df[column] == val]
            else:
                if op == '>':
                    return df[df[column] > val]
                elif op == '<':
                    return df[df[column] < val]
                elif op == '=':
                    return df[df[column] == val]
        except ValueError:
            return df[df[column].astype(str).str.contains(value, case=False, na=False)]
    else:
        try:
            if is_numeric:
                numeric_value = pd.to_numeric(value.replace(',', ''))
                return df[df[column] == numeric_value]
            else:
                return df[df[column] == value]
        except ValueError:
            return df[df[column].astype(str).str.contains(value, case=False, na=False)]
    
def count_value(df, query):
    column,col2,v=extractor(query, df)
    if column is None:
        return f"Column not found in file. Available columns are: {', '.join(df.columns)}"
    value=ex(query,df,column)
    if value is None :
        return f"Value not present in the Column '{column}'."
    try:
        if df[column].dtype.kind in 'biufc':
            count_value = df[df[column] == value].shape[0]
        else:
            count_value = df[df[column].astype(str).str.contains(value, case=False)].shape[0]
        return f"The number of '{value}' in {column} is {count_value}."
    except Exception as e:
        return f"Error counting value: {e}"
if submit_button:
    if query:
        if df is None:
            st.markdown(f"<p style='font-size: 20px; text-align: center;'>No dataset loaded. Please upload a file first.</p>", unsafe_allow_html=True)
        else:
            action = predict_query(model, tokenizer, label_encoder, query, device)
            action = action.strip().lower().replace('"', '')
            #st.write(f"Predicted Action: {action}")
            if action == "show columns":
                st.markdown(f"<p style='font-size: 20px; text-align: center;'>The columns in file are :</p>", unsafe_allow_html=True)
                st.write(df.columns.tolist())
            elif action == "show data":
                column,col2,value=extractor(query, df)
                if value is None :
                    st.markdown(f"<p style='font-size: 20px; text-align: center;'>Please enter the number of rows.</p>", unsafe_allow_html=True)
                else:    
                    try:
                        n = int(value)
                        st.dataframe(df.head(n))
                    except ValueError:
                        st.markdown(f"<p style='font-size: 20px; text-align: center;'>Please provide a valid number for rows to display.</p>", unsafe_allow_html=True)
            elif action == "describe":
                st.markdown(f"<p style='font-size: 20px; text-align: center;'>Description of the Data </p>", unsafe_allow_html=True)
                st.dataframe(df.describe())
            elif action == "filter":
                try:
                    column,col2,_=extractor(query, df)
                    if column is None:
                        st.markdown(f"<p style='font-size: 20px; text-align: center;'>Column not found in file.</p>", unsafe_allow_html=True)
                    value=ex(query,df,column)
                    if value is None :
                        st.markdown(f"<p style='font-size: 20px; text-align: center;'>Value not present in column.</p>", unsafe_allow_html=True)
                    filtered_df = apply_filter(df, column, str(value))
                    st.write(f"Filtered by {column} = {value}")
                    lengg=len(filtered_df)
                    st.markdown(f"<p style='font-size: 20px; text-align: center;'>Number of rows in filtered dataset: <strong>{lengg:,}.</p>", unsafe_allow_html=True)
                    st.dataframe(filtered_df)
                    csv = filtered_df.to_csv(index=False)
                    st.download_button(
                        label="Download filtered data as CSV",
                        data=csv,
                        file_name="filtered_data.csv",
                        mime="text/csv",
                    )
                except Exception as e:
                    #st.error(f"Error processing filter: {e}")
                    st.markdown(f"<p style='font-size: 20px; text-align: center;'>Error processing filter.</p>", unsafe_allow_html=True)

            elif action == "count rows":
                row_count = len(df)
                st.markdown(f"<p style='font-size: 20px; text-align: center;'>The dataset contains <strong>{row_count:,}</strong> rows of data.</p>", unsafe_allow_html=True)
            elif action in ["maximum", "minimum", "average", "sum"]:
                try:
                    column,col2,value=extractor(query, df)
                    if column:
                        if action == "maximum":
                            m1=df[column].max()
                            st.markdown(f"<p style='font-size: 20px; text-align: center;'>Maximum value in column '<strong>{column}</strong>' : <strong>{m1}.</p>", unsafe_allow_html=True)
                        elif action == "minimum":
                            m2=df[column].min()
                            st.markdown(f"<p style='font-size: 20px; text-align: center;'>Minimum value in column '<strong>{column}</strong>' : <strong>{m2}.</p>", unsafe_allow_html=True)
                        elif action == "average":
                            m3=df[column].mean()
                            st.markdown(f"<p style='font-size: 20px; text-align: center;'>Average value in column '<strong>{column}</strong>' : <strong>{m3}.</p>", unsafe_allow_html=True)
                        elif action == "sum":
                            m4=df[column].sum()
                            st.markdown(f"<p style='font-size: 20px; text-align: center;'>Sum value in column '<strong>{column}</strong>' : <strong>{m4}.</p>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<p style='font-size: 20px; text-align: center;'>Column not found in the dataset.</p>", unsafe_allow_html=True)
                except Exception as e:
                    if "can only concatenate str (not" in str(e):
                        st.markdown(f"<p style='font-size: 20px; text-align: center;'>The column contains non-numeric values.</p>", unsafe_allow_html=True)
                    else:
                        #st.write(f"Error calculating {action}: {e}")
                        st.markdown(f"<p style='font-size: 20px; text-align: center;'>Error calculating.</p>", unsafe_allow_html=True)
            elif action == "count value":
                result = count_value(df, query)
                #st.write(result)
                st.markdown(f"<p style='font-size: 20px; text-align: center;'><strong>{result}</strong></p>", unsafe_allow_html=True)
            elif action == "missing values":
                st.markdown(f"<p style='font-size: 20px; text-align: left;'>Missing Values : </p>", unsafe_allow_html=True)
                st.write(df.isnull().sum())
            elif action == "unique values":
                column,col2,value=extractor(query, df)
                if column is None :
                    st.markdown(f"<p style='font-size: 20px; text-align: center;'>Column not present in the file.</p>", unsafe_allow_html=True)
                else:    
                    try:
                        st.write(df[column].unique().tolist())
                    except Exception as e:
                        #st.write(f"Error fetching unique values: {e}")
                        st.markdown(f"<p style='font-size: 20px; text-align: center;'>Error fetching unique values.</p>", unsafe_allow_html=True)
            elif action == "plot":
                plot_query = query.lower()
                
                plot_types = {
                    'bar': 'bar',
                    'pie': 'pie',
                    'line': 'line',
                    'scatter': 'scatter',
                    'histogram': 'histogram',
                    'box': 'box'
                }
                plot_type = next((plot_types[key] for key in plot_types if key in plot_query), None)
                
                if plot_type:
                    x_col,y_col,value=extractor(query, df)
                    if x_col:
                        try:
                            create_plot(df, plot_type, x_col, y_col)
                        except Exception as e:
                            st.markdown(f"<p style='font-size: 20px; text-align: center;'>Error creating plot: {str(e)}</p>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<p style='font-size: 20px; text-align: center;'>Column not presnt in the file.</p>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<p style='font-size: 20px; text-align: center;'>Unsupported plot type. Please try again.</p>", unsafe_allow_html=True)
            elif action == "add column":
                new_col_name = st.text_input("New Column Name")
                existing_col = st.selectbox("Choose Existing Column", df.columns)
                operation = st.selectbox("Choose Operation", ["multiply", "add", "subtract", "divide"])
                if operation == "multiply" or operation == "divide":
                    factor = st.number_input(f"Enter the {operation} factor", value=2.0, step=1.0)
                else:
                    factor = st.number_input(f"Enter the {operation} value", value=0.0, step=1.0)
                if st.button("Add Column"):
                    try:
                        if operation == "multiply":
                            df[new_col_name] = df[existing_col] * factor
                        elif operation == "add":
                            df[new_col_name] = df[existing_col] + factor
                        elif operation == "subtract":
                            df[new_col_name] = df[existing_col] - factor
                        elif operation == "divide":
                            df[new_col_name] = df[existing_col] / factor
                        #st.write(f"Column {new_col_name} added successfully.")
                        st.markdown(f"<p style='font-size: 20px; text-align: center;'>Column '<strong>{new_col_name}</strong>' added successfully.</p>", unsafe_allow_html=True)

                        st.dataframe(df.head())
                        csv = df.to_csv(index=False)
                        btn = st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name='updated_data.csv',
                            mime='text/csv',
                        )
                    except Exception as e:
                        #st.write(f"Error adding column: {e}")
                        st.markdown(f"<p style='font-size: 20px; text-align: center;'>Error adding column.</p>", unsafe_allow_html=True)
                        st.markdown(f"<p style='font-size: 20px; text-align: center;'>Please try again.</p>", unsafe_allow_html=True)
            elif action == "drop column":
                column,col2,value=extractor(query, df)
                if column is None :
                    st.markdown(f"<p style='font-size: 20px; text-align: center;'>Column not present in the file.</p>", unsafe_allow_html=True)
                else:    
                    try:
                        df.drop(columns=[column], inplace=True)
                        st.markdown(f"<p style='font-size: 20px; text-align: center;'>Column '<strong>{column}</strong>' dropped successfully.</p>", unsafe_allow_html=True)
                        st.dataframe(df.head())
                        csv = df.to_csv(index=False)
                        st.download_button(label="Download CSV", data=csv, file_name='updated_data.csv', mime='text/csv')
                    except Exception as e:
                        #st.write(f"Error dropping column: {e}")
                        st.markdown(f"<p style='font-size: 20px; text-align: center;'>Error dropping column.</p>", unsafe_allow_html=True)
            elif action == "sample data":
                try:
                    column,col2,value=extractor(query, df)
                    n = int(value)
                    st.markdown(f"<p style='font-size: 20px; text-align: center;'>Sample of size '<strong>{n}</strong>':</p>", unsafe_allow_html=True)
                    st.dataframe(df.sample(n))
                except ValueError:
                    #st.write("Please provide a valid number for sampling.")
                    st.markdown(f"<p style='font-size: 20px; text-align: center;'>Please provide a valid number for sampling.</p>", unsafe_allow_html=True)

            elif action == "rename column":
                try:
                    old_col_name,_,_=extractor(query, df)
                    if old_col_name is None:
                        st.markdown("<p style='font-size: 20px; text-align: center;'>Old column name not found in the query.</p>", unsafe_allow_html=True)
                    words = query.split()
                    try:
                        old_col_index = words.index(old_col_name)
                        if old_col_index + 2 < len(words):
                            new_col_name = words[old_col_index + 2]
                        else:
                            st.markdown("<p style='font-size: 20px; text-align: center;'>New column name not found in the query.</p>", unsafe_allow_html=True)
                    except ValueError:
                        st.markdown("<p style='font-size: 20px; text-align: center;'>Error processing the query.</p>", unsafe_allow_html=True)
                    
                    df = rename_column(df, old_col_name, new_col_name)
                    #st.write(f"Column {old_col_name} renamed to {new_col_name} successfully.")
                    st.markdown(f"<p style='font-size: 20px; text-align: center;'>Column <strong>{old_col_name}</strong> renamed to <strong>{new_col_name} </strong> successfully.</p>", unsafe_allow_html=True)
                    st.dataframe(df.head())
                    csv = df.to_csv(index=False)
                    btn = st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name='updated_data.csv',
                        mime='text/csv',
                    )
                except Exception as e:
                    #st.write(f"Error renaming column: {e}")
                    st.markdown(f"<p style='font-size: 20px; text-align: center;'>Error renaming column.</p>", unsafe_allow_html=True)
            elif action == "group by":
                group_col,col2,value=extractor(query, df)
                if group_col is None :
                    st.markdown(f"<p style='font-size: 20px; text-align: center;'>Column not present in the file.</p>", unsafe_allow_html=True)
                else:    
                    try:
                        grouped_data = df.groupby(group_col).apply(lambda x: x.to_dict('records'))
                        gr=group_col.capitalize()
                        #st.write(f"### Grouped Data by {group_col.capitalize()} ###")
                        st.markdown(f"<p style='font-size: 20px; text-align: center;'>Grouped Data by '<strong>{gr}</strong>'successfully.</p>", unsafe_allow_html=True)
                        for group_name, group_data in grouped_data.items():
                            st.write(f"**{group_name}**")
                            st.write(pd.DataFrame(group_data))
                    except Exception as e:
                        #st.write(f"Error grouping by column: {e}")
                        st.markdown(f"<p style='font-size: 20px; text-align: center;'>Error grouping by column.</p>", unsafe_allow_html=True)
        
            elif action == "heat map":
                try:
                    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                    numeric_df = df[numeric_cols]
                    corr_matrix = numeric_df.corr()
                    plt.figure(figsize=(10, 6))
                    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
                    plot_buffer = BytesIO()
                    plt.savefig(plot_buffer, format='png')
                    plot_buffer.seek(0)
                    plot_base64 = base64.b64encode(plot_buffer.getvalue()).decode('utf-8')
                    st.markdown(f"<p style='font-size: 20px; text-align: center;'>Heatmap : </p>", unsafe_allow_html=True)
                    st.pyplot(plt)
                    st.download_button(
                        label="Download Plot",
                        data=plot_base64,
                        file_name=f"heatmap.png",
                        mime="image/png"
                    )
                except Exception as e:
                    #st.write(f"Error creating heatmap: {e}")
                    st.markdown(f"<p style='font-size: 20px; text-align: center;'>Error creating Heatmap.</p>", unsafe_allow_html=True)
            elif action == "concat data":
                uploaded_file_2 = st.file_uploader("Choose a file to concatenate", type=['csv', 'xlsx'])
                if uploaded_file_2 is not None:
                    try:
                        df2 = load_dataset(uploaded_file_2)
                        df = pd.concat([df, df2], axis=0)
                        st.markdown(f"<p style='font-size: 20px; text-align: center;'>Dataframes concatenated successfully.</p>", unsafe_allow_html=True)
                        st.dataframe(df.head())
                        csv = df.to_csv(index=False)
                        btn = st.download_button(
                            label="Download Concatenated CSV",
                            data=csv,
                            file_name='concatenated_data.csv',
                            mime='text/csv',
                        )
                    except Exception as e:
                        #st.write(f"Error concatenating data: {e}")
                        st.markdown(f"<p style='font-size: 20px; text-align: center;'>Error concatenating data.</p>", unsafe_allow_html=True)
            elif action == "join data":
                uploaded_file_2 = st.file_uploader("Choose a file to join", type=['csv', 'xlsx'])
                join_col = st.selectbox("Choose Column to Join On", df.columns)
                join_type = st.selectbox("Choose Join Type", ["inner", "outer", "left", "right"])
                if uploaded_file_2 is not None:
                    try:
                        df2 = load_dataset(uploaded_file_2)
                        df = df.merge(df2, on=join_col, how=join_type)
                        st.markdown(f"<p style='font-size: 20px; text-align: center;'>Dataframes joined successfully..</p>", unsafe_allow_html=True)
                        st.dataframe(df.head())
                        csv = df.to_csv(index=False)
                        btn = st.download_button(
                            label="Download Joined CSV",
                            data=csv,
                            file_name='joined_data.csv',
                            mime='text/csv',
                        )
                    except Exception as e:
                        #st.write(f"Error joining data: {e}")
                        st.markdown(f"<p style='font-size: 20px; text-align: center;'>Error joining data.</p>", unsafe_allow_html=True)
            elif action == "rolling window":
                column,col2,window_size = extractor(query, df)
                if column is None :
                    st.markdown(f"<p style='font-size: 20px; text-align: center;'>Column not present in the file.</p>", unsafe_allow_html=True)
                elif window_size is None :
                    st.markdown(f"<p style='font-size: 20px; text-align: center;'>Enter a valid size(number).</p>", unsafe_allow_html=True) 
                else:    
                    try:
                        window_size=1
                        window_size = int(window_size)
                        rolling_df = df[column].rolling(window=window_size).mean()
                        st.markdown(f"<p style='font-size: 20px; text-align: center;'> Rolling mean of '<strong>{column}</strong>' by '<strong>{window_size}</strong>':</p>", unsafe_allow_html=True)
                        st.line_chart(rolling_df)
                    except Exception as e:
                        #st.write(f"Error applying rolling window: {e}")
                        st.markdown(f"<p style='font-size: 20px; text-align: center;'>Error applying rolling window.</p>", unsafe_allow_html=True)
            elif action == "apply function":
                column = st.selectbox("Choose Column to Apply Function", df.columns)
                with st.expander("Functions Available for Apply Function"):
                    st.markdown(
                        f"""
                        <div class="expander-container">
                        {expander_content}
                        """,
                        unsafe_allow_html=True
                    )
                custom_func = st.text_area("Enter Custom Function (e.g., lambda x: x*2)")
                if st.button("Apply Function"):
                    try:
                        df[column] = df[column].apply(eval(custom_func))
                        st.markdown(f"<p style='font-size: 20px; text-align: center;'>Function applied to <strong>{column}</strong>successfully.</p>", unsafe_allow_html=True)
                        st.dataframe(df.head())
                        csv = df.to_csv(index=False)
                        btn = st.download_button(
                            label="Download Updated CSV",
                            data=csv,
                            file_name='updated_data.csv',
                            mime='text/csv',
                        )
                    except Exception as e:
                        #st.write(f"Error applying function: {e}")
                        st.markdown(f"<p style='font-size: 20px; text-align: center;'>Error applying function.</p>", unsafe_allow_html=True)
            else:
                if action not in ["show data", "describe", "filter", "plot", "count rows", 
                        "maximum", "minimum", "average", "sum", "count value", "missing values", 
                        "unique values", "add column", "drop column", "rename column", "group by", 
                        "heat map", "sample data", "concat data", "join data", "rolling window", 
                        "apply function"]:
                    #st.write(f"Unrecognized action ,for query: {query}")
                    st.markdown(f"<p style='font-size: 20px; text-align: center;'>Please try again .</p>", unsafe_allow_html=True)
