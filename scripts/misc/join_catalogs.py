"""
Join catalogs given an input folder and an output path
It will look for fits files in the input folder and join 
them into a single fits file. Assuming that all the fits files
have the same columns and units.

Usage:
    python join_catalogs.py input_folder output_path

Args:
    input_folder: str
        Path to the folder with the fits files
    output_path: str
        Path to the output fits file
"""

import os
import sys
import glob
import numpy as np
from astropy.table import Table, vstack

def join_catalogs(input_folder, output_path):
    # Get all the fits files in the input folder
    fits_files = glob.glob(os.path.join(input_folder, '*.fits'))
    if not fits_files:
        raise ValueError(f"No fits files found in {input_folder}")

    # Read the first fits file
    first_file = fits_files[0]
    first_table = Table.read(first_file)
    columns = first_table.colnames
    units = first_table.columns.units

    # Read the rest of the fits files
    tables = [first_table]
    for fits_file in fits_files[1:]:
        table = Table.read(fits_file)
        tables.append(table)

    # Concatenate all the tables
    final_table = vstack(tables)

    # Save the output
    final_table.write(output_path, overwrite=True)