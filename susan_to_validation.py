import sys
import os
import pandas as pd
from shutil import copyfile

conversion_csv = "susan_labels.csv"
biristol_labels = ["1", "2", "3", "4", "5", "6", "7", "weird"]


def convert_susan_to_validation(susan_dir: str, output_dir: str = "validation"):
    """
    Convert Susan's expert rated images into how we store out data for validation

    Input:
        susan_dir: string of path to susan raw data
        output_dir: optional string of path to output directory

    Output:
        None
    """
    # Try to get the conversion CSV
    conversion_df = None
    try:
        conversion_df = pd.read_csv(os.path.join(susan_dir, conversion_csv))
    except Exception as e:
        print(e)
        sys.exit(0)

    # Make the output directory
    os.makedirs(output_dir, exist_ok=True)
    for label in biristol_labels:
        os.makedirs(os.path.join(output_dir, label))

    for _, data in conversion_df.iterrows():
        # Get the expert id
        expert_id = data["susan_label"]

        # If expert ID is Unreadable
        if expert_id == "Unreadable":
            expert_id = "weird"

        # Place this picture in appropriate directory
        image_file = data["uuid"]
        copyfile(
            os.path.join(susan_dir, image_file),
            os.path.join(output_dir, expert_id, image_file),
        )


if __name__ == "__main__":
    if len(sys.argv) == 3:
        convert_susan_to_validation(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 2:
        convert_susan_to_validation(sys.argv[1])
    else:
        print("Usage: python susan_to_validation.py [SUSAN DIR] OPTIONAL: [OUTPUT DIR]")
        sys.exit(0)