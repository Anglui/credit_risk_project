import os
import yaml
import dask.dataframe as dd
import pandas as pd
import openpyxl
import re

os.environ["DASK_TEMP_DIR"] = "/Volumes/Elements/fannie_mae_data/tmp" 

# --- Load paths from config ---
with open("config/paths.yaml", "r") as f:
    paths = yaml.safe_load(f)
    
raw_dir = paths["raw"]
out_path = os.path.join(paths["processed"], "train_dataset.parquet")
glossary_path = paths["metadata"]


# --- Get columns and data types from SF data glossary ---
def parse_glossary(glossary_path):
    workbook = openpyxl.load_workbook(glossary_path, data_only=True)
    sheet = workbook.active
    
    column_names = []
    dtypes_dict = {}
    cast_types_dict = {}
    parsing_info_dict = {}
    
    if sheet:
        for row in sheet.iter_rows(min_row = 2):
            field = row[1].value
            sf_flag = row[8].value
            dtype = row[9].value
            format_hint = row[10].value
            
            if not field or str(sf_flag).strip().upper() == "NA":
                continue
            
            column_names.append(field)
            dtypes_dict[field] = "str"
            post_type = "str"
            
            # DATE parsing
            if dtype and "DATE" in dtype.upper():
                post_type = "datetime64[ns]"
                parsing_info_dict[field] = {"format": "%m%Y"}
                
            # NUMERIC parsing
            elif dtype and "NUMERIC" in dtype.upper():
                if format_hint:
                    if re.match(r"9\(\d+\)$", format_hint):
                        post_type = "int64"
                    elif re.match(r"9\(\d+\)\.\d+$", format_hint):
                        post_type = "float64"
                    elif re.match(r"9\.\(\d+\)\.\d+$", format_hint):  # e.g., 9.(2).9999
                        post_type = "float64"
                    else:
                        post_type = "float64"
                else:
                    post_type = "float64"
            
            # ALPHANUMERIC parsing
            elif dtype and "ALPHA" in dtype.upper():
                post_type = "str"
            
            cast_types_dict[field] = post_type
        
    return column_names, dtypes_dict, cast_types_dict, parsing_info_dict

column_names, dtypes_dict, cast_types_dict, parsing_info_dict = parse_glossary(glossary_path)
print(f"Loaded {len(column_names)} column names from glossary.")


# --- Load SF Performance data with Dask ---
dask_df = dd.read_csv(
    os.path.join(raw_dir, "*.csv"),
    sep = "|",
    header = None,
    names = column_names,
    dtype = dtypes_dict,
    encoding = "ISO-8859-1",
    assume_missing = True
)


# --- Generate labels and filter out all temporal data ---
def is_default(code):
    if isinstance(code, str):
        code = code.strip().upper()
        if code == "R":
            return True
        try:
            return int(code) >= 3
        except ValueError:
            return False
    return False

def label_and_features(group_df):
    group_df = group_df.sort_values("Monthly Reporting Period")
    first_row = group_df.iloc[0]
    first_12 = group_df[group_df["Loan Age"] < 12]

    default = first_12["Current Loan Delinquency Status"].map(is_default).any()

    return pd.Series({
        "Default_in_12M": int(default),
        "Channel": first_row["Channel"],
        "Seller Name": first_row["Seller Name"],
        "Servicer Name": first_row["Servicer Name"],
        "Original Interest Rate": first_row["Original Interest Rate"],
        "Original UPB": first_row["Original UPB"],
        "Original Loan Term": first_row["Original Loan Term"],
        "Original Loan to Value Ratio (LTV)": first_row["Original Loan to Value Ratio (LTV)"],
        "Original Combined Loan to Value Ratio (CLTV)": first_row["Original Combined Loan to Value Ratio (CLTV)"],
        "Number of Borrowers": first_row["Number of Borrowers"],
        "Debt-To-Income (DTI)": first_row["Debt-To-Income (DTI)"],
        "Borrower Credit Score at Origination": first_row["Borrower Credit Score at Origination"],
        "Co-Borrower Credit Score at Origination": first_row["Co-Borrower Credit Score at Origination"],
        "First Time Home Buyer Indicator": first_row["First Time Home Buyer Indicator"],
        "Loan Purpose": first_row["Loan Purpose"],
        "Property Type": first_row["Property Type"],
        "Number of Units": first_row["Number of Units"],
        "Occupancy Status": first_row["Occupancy Status"],
        "Property State": first_row["Property State"],
        "Metropolitan Statistical Area (MSA)": first_row["Metropolitan Statistical Area (MSA)"],
        "Zip Code Short": first_row["Zip Code Short"],
        "Mortgage Insurance Percentage": first_row["Mortgage Insurance Percentage"],
        "Amortization Type": first_row["Amortization Type"],
        "Mortgage Insurance Type": first_row["Mortgage Insurance Type"],
        "Special Eligibility Program": first_row["Special Eligibility Program"],
        "High Balance Loan Indicator": first_row["High Balance Loan Indicator"],
        "Origination Date": first_row["Origination Date"],
        "LoanAge_max": group_df["Loan Age"].max()
    })


# --- Group by loan and get features/labels ---
print("Grouping and computing loan features and labels.")
loan_groups = dask_df.groupby("Loan Identifier").apply(label_and_features, meta={k: 'str' for k in cast_types_dict}).compute()

# --- Enforce correct types ---
print("Assigning accurate types.")
for col, dtype in cast_types_dict.items():
    if col in loan_groups.columns:
        try:
            if dtype == "datetime64[ns]":
                fmt = parsing_info_dict.get(col, {}).get("format", None)
                loan_groups[col] = pd.to_datetime(loan_groups[col], format=fmt, errors="coerce")
            else:
                loan_groups[col] = loan_groups[col].astype(dtype)
        except Exception as e:
            print(f"Could not cast '{col}' to {dtype}: {e}")

loan_groups["Default_in_12M"] = loan_groups["Default_in_12M"].astype("int64")
loan_groups["LoanAge_max"]    = loan_groups["LoanAge_max"].astype("float64")


# --- Filter loans without enough history ---
print("Filtering loans with less than 15 months of history.")
loan_groups = loan_groups[loan_groups["LoanAge_max"].astype(float) >= 15]


# --- Save final result to parquet ---
print(f"Saving labeled loans to: {out_path}")
loan_groups.to_parquet(out_path, engine="pyarrow", write_index=False)

print("Done.")