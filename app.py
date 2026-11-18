# streamlit builds the web page; we import it with the short name st
import streamlit as st

# pandas reads and handles the table data
import pandas as pd

# Set the browser tab title and use the full page width
st.set_page_config(page_title="Student Score Predictor", layout="wide")

# Show a big title at the top of the page
st.title("Student Score Predictor")

# Show a small grey line under the title explaining what the tool does
st.caption("Upload your school's student data and get a predictor built from your own students.")

# Create an upload box that accepts CSV or Excel files; it returns None until a file is chosen
uploaded = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

# If nothing has been uploaded yet, show a hint and stop the script here
if uploaded is None:
    st.info("Upload a file to begin.")
    st.stop()

# Check the file name to decide how to read it
if uploaded.name.endswith(".csv"):
    # Read a CSV file into a table
    df = pd.read_csv(uploaded)
else:
    # Read an Excel file into a table
    df = pd.read_excel(uploaded)

# Show a section heading
st.subheader("Preview of your data")

# Show the first 10 rows as an interactive table
st.dataframe(df.head(10))

# Show how big the data is
st.write("Rows:", df.shape[0], "| Columns:", df.shape[1])
# Let the user pick which column holds the score to predict (the target)
target = st.selectbox("Which column is the score you want to predict?", df.columns)

# The target must be numbers; if it is text, show an error and stop
if not pd.api.types.is_numeric_dtype(df[target]):
    st.error("The score column must contain numbers.")
    st.stop()

# Warn the user if the file is small, because small data gives unreliable results
if df.shape[0] < 200:
    st.warning("Your file has fewer than 200 students. Results may be unreliable.")


# A function that cleans a table; we put the steps in a function so we can reuse them later
def clean_data(data, target_col):
    # Work on a copy so the original upload is never changed (this avoids the "ran it twice" bug we hit earlier)
    data = data.copy()

    # Remove rows where the score itself is missing, since we cannot learn from them
    data = data.dropna(subset=[target_col])

    # Remove rows that are exact duplicates
    data = data.drop_duplicates()

    # Remember which columns we drop, so we can tell the user
    dropped = []

    # Go through every column except the score
    for col in list(data.columns):
        # Skip the target column
        if col == target_col:
            continue

        # If the column is numeric, fill empty cells with the median (middle value)
        if pd.api.types.is_numeric_dtype(data[col]):
            data[col] = data[col].fillna(data[col].median())
        else:
            # Text columns with many different values are probably names or IDs, so drop them
            if data[col].nunique() > 20:
                data = data.drop(columns=col)
                dropped.append(col)
            else:
                # Otherwise fill empty cells with the most common value (the mode)
                data[col] = data[col].fillna(data[col].mode()[0])

    # Find the text columns that are left
    text_cols = [c for c in data.columns if c != target_col and not pd.api.types.is_numeric_dtype(data[c])]

    # Turn each text column into 0/1 columns (one-hot encoding); drop_first avoids a duplicate column
    data = pd.get_dummies(data, columns=text_cols, drop_first=True, dtype=int)

    # Give back the cleaned table and the list of dropped columns
    return data, dropped


# Count the missing cells before cleaning
missing_before = int(df.isnull().sum().sum())

# Run the cleaning function on the uploaded table
clean_df, dropped_cols = clean_data(df, target)

# Count the missing cells after cleaning (should be 0)
missing_after = int(clean_df.isnull().sum().sum())

# Show a heading for the cleaning report
st.subheader("Cleaning report")

# Show before/after numbers in a row of three boxes
c1, c2, c3 = st.columns(3)
c1.metric("Missing cells before", missing_before)
c2.metric("Missing cells after", missing_after)
c3.metric("Rows kept", clean_df.shape[0])

# If any columns were dropped, tell the user which ones
if dropped_cols:
    st.info("Dropped columns that look like names or IDs: " + ", ".join(dropped_cols))

# Show a preview of the cleaned, fully numeric table
    st.dataframe(clean_df.head(10))

# Import the function that divides rows into training and testing groups.
from sklearn.model_selection import train_test_split
# Import the linear regression model.
from sklearn.linear_model import LinearRegression
# Import the random forest regression model.
from sklearn.ensemble import RandomForestRegressor
# Import the three measurements used to compare predictions.
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Put every input column except the chosen score into the feature table.
X = clean_df.drop(columns=[target])
# Put the chosen score column into the values the models will learn to predict.
y = clean_df[target]
# Divide the rows into training and testing groups, keeping 20 percent for testing.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model.
linear_model = LinearRegression()
# Create a random forest model with 100 trees and a repeatable random seed.
forest_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
# Train the linear regression model using only the training rows.
linear_model.fit(X_train, y_train)
# Train the random forest model using only the training rows.
forest_model.fit(X_train, y_train)

# Predict the scores for the rows held back from training with the linear model.
linear_predictions = linear_model.predict(X_test)
# Predict the scores for the rows held back from training with the forest model.
forest_predictions = forest_model.predict(X_test)
# Calculate the linear model's average absolute prediction error.
linear_mae = mean_absolute_error(y_test, linear_predictions)
# Calculate the linear model's root mean squared prediction error.
linear_rmse = mean_squared_error(y_test, linear_predictions) ** 0.5
# Calculate the linear model's R-squared score.
linear_r2 = r2_score(y_test, linear_predictions)
# Calculate the forest model's average absolute prediction error.
forest_mae = mean_absolute_error(y_test, forest_predictions)
# Calculate the forest model's root mean squared prediction error.
forest_rmse = mean_squared_error(y_test, forest_predictions) ** 0.5
# Calculate the forest model's R-squared score.
forest_r2 = r2_score(y_test, forest_predictions)

# Show a heading above the model results.
st.subheader("Model comparison")
# Put each model's three scores into a table for easy comparison.
comparison_df = pd.DataFrame({"Model": ["Linear Regression", "Random Forest"], "MAE": [linear_mae, forest_mae], "RMSE": [linear_rmse, forest_rmse], "R2": [linear_r2, forest_r2]})
# Display the comparison table without showing its row index.
st.dataframe(comparison_df, hide_index=True)
# Choose the model with the lower average absolute error.
if linear_mae <= forest_mae:
    # Save the name of the linear model when it has the lowest MAE.
    best_model_name = "Linear Regression"
    # Save the linear model's MAE for the plain-language message.
    best_model_mae = linear_mae
# Otherwise, use the random forest model as the winner.
else:
    # Save the name of the forest model when it has the lowest MAE.
    best_model_name = "Random Forest"
    # Save the forest model's MAE for the plain-language message.
    best_model_mae = forest_mae
# Tell the user which model was more accurate and its average error in marks.
st.write(f"On students it has never seen, the best model ({best_model_name}) is off by about {best_model_mae:.2f} marks on average.")
# Remind the user that predictions should support teachers rather than decide outcomes.
st.caption("Predictions are estimates to support teachers, not decisions about students.")

# ---------- Model training: Linear Regression only ----------

# Import the function that splits the data into a learning part and a testing part
from sklearn.model_selection import train_test_split

# Import the Linear Regression model
from sklearn.linear_model import LinearRegression

# Import the three measurements used to judge the model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# X = every column except the score (the inputs the model learns from)
X = clean_df.drop(columns=[target])

# y = only the score column (the thing we want to predict)
y = clean_df[target]

# Keep 20 percent of students hidden for testing; random_state=42 makes the split repeatable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an empty Linear Regression model
model = LinearRegression()

# Train it on the 80 percent learning students
model.fit(X_train, y_train)

# Predict scores for the 20 percent test students the model has never seen
test_predictions = model.predict(X_test)

# MAE: the average size of the mistake, in marks
mae = mean_absolute_error(y_test, test_predictions)

# RMSE: like MAE, but big mistakes count more
rmse = mean_squared_error(y_test, test_predictions) ** 0.5

# R2: how much of the differences between students the model explains (1.0 is perfect)
r2 = r2_score(y_test, test_predictions)

# Show a heading for the accuracy section
st.subheader("Model accuracy")

# Make three side-by-side boxes
c1, c2, c3 = st.columns(3)

# Put one number in each box
c1.metric("Average error (MAE)", round(mae, 2))
c2.metric("RMSE", round(rmse, 2))
c3.metric("R2", round(r2, 2))

# Explain the result in plain language
st.write(f"On students it has never seen, the model is off by about {mae:.2f} marks on average.")

# Remind everyone that these are estimates
st.caption("Predictions are estimates to support teachers, not decisions about students.")


# ---------- Predict a new student ----------

# Show a heading for this section
st.subheader("Predict a new student")

# The boxes to show: every original column except the score and any name/ID columns we dropped
input_cols = [c for c in df.columns if c != target and c not in dropped_cols]

# Which of those boxes hold words (not numbers)
text_input_cols = [c for c in input_cols if not pd.api.types.is_numeric_dtype(df[c])]

# A form waits until the button is pressed before doing anything
with st.form("predict_form"):
    # An empty dictionary to collect the answers: column name -> what the person chose
    answers = {}

    # Split the form into 3 side-by-side columns so it is not too long
    form_cols = st.columns(3)

    # Go through every input column; i counts 0, 1, 2, 3 ...
    for i, col in enumerate(input_cols):
        # i % 3 gives 0, 1, 2, 0, 1, 2 ..., so the boxes spread across the 3 columns
        with form_cols[i % 3]:
            if col in text_input_cols:
                # Text column: list the words that appear in the data, in alphabetical order
                options = sorted(df[col].dropna().unique().tolist())
                # Start the dropdown on the most common word
                default_index = options.index(df[col].mode()[0])
                # Show the dropdown and store the chosen word
                answers[col] = st.selectbox(col, options, index=default_index)
            elif pd.api.types.is_integer_dtype(df[col]):
                # Whole-number column: limited to the smallest and largest value in the data
                answers[col] = st.number_input(col, min_value=int(df[col].min()), max_value=int(df[col].max()), value=int(df[col].median()))
            else:
                # Decimal column: same idea, but with decimals
                answers[col] = st.number_input(col, min_value=float(df[col].min()), max_value=float(df[col].max()), value=float(df[col].median()))

    # The button that sends the form; submitted becomes True when it is pressed
    submitted = st.form_submit_button("Predict score")

# Only run the prediction after the button has been pressed
if submitted:
    # Turn the answers into a one-row table
    new_row = pd.DataFrame([answers])

    # Put the new student under all the original students (without their scores)
    combined = pd.concat([df[input_cols], new_row], ignore_index=True)

    # Convert words to 0/1 columns the same way as in training
    combined = pd.get_dummies(combined, columns=text_input_cols, drop_first=True, dtype=int)

    # Keep only the new student's row, with the exact columns and order the model expects
    new_X = combined.tail(1).reindex(columns=X.columns, fill_value=0)

    # Ask the model for a score
    prediction = float(model.predict(new_X)[0])

    # A score cannot be below 0
    prediction = max(prediction, 0)

    # If the school's scores never go above 100, a prediction cannot go above 100 either
    if y.max() <= 100:
        prediction = min(prediction, 100)

    # Show the predicted score in big text
    st.metric("Predicted score", round(prediction, 1))

    # Explain how reliable it is
    st.write(f"On students it has never seen, the model is off by about {mae:.2f} marks on average.")
