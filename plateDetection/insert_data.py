import pandas as pd
import mysql.connector
from config import db_config

# Read cleaned data from CSV
input_file = "E:/AutoParkAI/plateDetection/cleaned_data.csv"
df = pd.read_csv(input_file)

# Connect to MySQL
conn = mysql.connector.connect(**db_config)
cursor = conn.cursor()

# Insert cleaned data into the database
for index, row in df.iterrows():
    cursor.execute("""
    INSERT INTO users (plate_number, plate_confidence_score)
    VALUES (%s, %s)
    """, (row["license_number"], row["license_number_score"]))

# Commit the changes
conn.commit()
conn.close()

print("Data successfully inserted into MySQL database.")