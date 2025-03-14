import pandas as pd

#Load raw data
input_file=r"E:\AutoParkAI\plateDetection\raw_data.csv"
output_file=r"E:\AutoParkAI\plateDetection\cleaned_data.csv"

#read input file
df=pd.read_csv(input_file)

#remove unused columns
columns_to_remove = ["frame_nmr","car_id", "car_bbox","license_plate_bbox","license_plate_bbox_score"]
df = df.drop(columns=columns_to_remove)

# Remove duplicate plate numbers, keeping the one with highest confidence
df = df.sort_values(by=['license_number_score'], ascending=False).drop_duplicates(subset=['license_number'])

# Keep only rows with confidence above a threshold
confidence_threshold = 0.5
df = df[df['license_number_score'] >= confidence_threshold]

# Save cleaned data to new CSV file
df.to_csv(output_file, index=False)

print(f"✅ Data cleaned and saved to {output_file}")