import os
import sys
import glob
import pandas as pd
from astropy.table import Table

def join_catalogs_to_csv(input_folder, output_path):
    # Get all the FITS files in the input folder
    fits_files = glob.glob(os.path.join(input_folder, '*.fits'))
    if not fits_files:
        raise ValueError(f"No FITS files found in {input_folder}")

    print(f"Found {len(fits_files)} FITS files in {input_folder}.")

    for i, fits_file in enumerate(fits_files):
        print(f"Processing file: {fits_file}")
        table = Table.read(fits_file)
        df = table.to_pandas()  # Convert to pandas DataFrame

        if i == 0:
            # Write the header and data for the first file
            df.to_csv(output_path, index=False, mode="w")
        else:
            # Append data only (no header)
            df.to_csv(output_path, index=False, header=False, mode="a")

        print(f"Processed {i + 1}/{len(fits_files)} files.")

    print(f"Concatenated table saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python join_catalogs_to_csv.py input_folder output_path")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_path = sys.argv[2]

    try:
        join_catalogs_to_csv(input_folder, output_path)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
