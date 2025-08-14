import pandas as pd
import sys

def merge_data(main_file, location_file):
    """
    Merges latitude and longitude from the location_file into the main_file.

    The merge is based on matching the start date columns.
    """
    try:
        main_df = pd.read_csv(main_file)
        loc_df = pd.read_csv(location_file)

        print("--- Data loaded ---")
        print(f"Main dataframe shape: {main_df.shape}")
        print(f"Location dataframe shape: {loc_df.shape}")

        # Define the columns to merge on
        # Assuming 'startyear', 'startmonth', 'startday' uniquely identify a fire event
        # in both files for the purpose of this merge.
        merge_cols = ['startyear', 'startmonth', 'startday']
        
        # Check if merge columns exist in both dataframes
        if not all(col in main_df.columns for col in merge_cols):
            print(f"Error: Main file '{main_file}' is missing one of the required merge columns: {merge_cols}")
            return
        if not all(col in loc_df.columns for col in merge_cols):
            print(f"Error: Location file '{location_file}' is missing one of the required merge columns: {merge_cols}")
            return

        # Select only the necessary columns from the location dataframe
        loc_subset_df = loc_df[merge_cols + ['lat', 'lng']]

        # Drop duplicates from the location subset to avoid creating extra rows in the main df
        loc_subset_df = loc_subset_df.drop_duplicates(subset=merge_cols)

        print("\nPerforming merge...")
        # Perform a 'left' merge to keep all rows from the main_df
        merged_df = pd.merge(main_df, loc_subset_df, on=merge_cols, how='left')

        # Rename the columns to what the training script expects
        merged_df.rename(columns={'lat': 'start_latitude', 'lng': 'start_longitude'}, inplace=True)

        print(f"Merged dataframe shape: {merged_df.shape}")

        # Check for rows that didn't get a location
        missing_locs = merged_df['start_latitude'].isnull().sum()
        if missing_locs > 0:
            print(f"Warning: {missing_locs} rows in the main file did not have a matching location and will have NaN values.")

        # Overwrite the original main file
        merged_df.to_csv(main_file, index=False)
        print(f"\nSuccessfully merged location data into '{main_file}'.")

    except FileNotFoundError as e:
        print(f"Error: File not found. {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    main_file_path = '/Users/heejunida/wild_fire_project/wild_fire_client/final_merged_feature_engineered.csv'
    location_file_path = '/Users/heejunida/wild_fire_project/wild_fire_client/gangwon_fire_ml_input.csv'
    merge_data(main_file_path, location_file_path)
