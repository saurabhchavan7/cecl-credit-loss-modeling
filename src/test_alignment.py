"""
test_alignment.py
-----------------
Quick validation script to verify column alignment before bulk processing.
Reads 5 rows from the first raw file and checks that each field contains
the expected type of data based on the glossary mapping.

Run this BEFORE running data_pipeline.py on all quarters.
"""

import pandas as pd
from pathlib import Path


def main():

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    filepath = PROJECT_ROOT / "data" / "raw" / "2005Q1" / "2005Q1.csv"

    # Mapping: field_name -> raw column index (glossary field number - 1)
    FIELD_MAP = {
        "loan_id": 1,
        "monthly_reporting_period": 2,
        "channel": 3,
        "original_interest_rate": 7,
        "original_upb": 9,
        "original_loan_term": 12,
        "origination_date": 13,
        "first_payment_date": 14,
        "original_ltv": 19,
        "original_cltv": 20,
        "number_of_borrowers": 21,
        "dti": 22,
        "borrower_credit_score": 23,
        "first_time_home_buyer": 25,
        "loan_purpose": 26,
        "property_type": 27,
        "number_of_units": 28,
        "occupancy_status": 29,
        "property_state": 30,
        "current_actual_upb": 11,
        "loan_age": 15,
        "current_loan_delinquency_status": 39,
        "zero_balance_code": 43,
        "foreclosure_costs": 53,
        "net_sale_proceeds": 58,
    }

    indices = sorted(FIELD_MAP.values())
    names_map = {v: k for k, v in FIELD_MAP.items()}
    sorted_names = [names_map[i] for i in indices]

    df = pd.read_csv(
        filepath, sep="|", header=None, nrows=5,
        usecols=indices, names=sorted_names, dtype=str,
    )

    row = df.iloc[0]

    print("=" * 60)
    print("COLUMN ALIGNMENT TEST")
    print("=" * 60)
    print()
    print("Field values from first row:")
    print(f"  loan_id:                  {row['loan_id']}")
    print(f"  monthly_reporting_period: {row['monthly_reporting_period']}")
    print(f"  channel:                  {row['channel']}")
    print(f"  original_interest_rate:   {row['original_interest_rate']}")
    print(f"  original_upb:             {row['original_upb']}")
    print(f"  original_loan_term:       {row['original_loan_term']}")
    print(f"  origination_date:         {row['origination_date']}")
    print(f"  first_payment_date:       {row['first_payment_date']}")
    print(f"  original_ltv:             {row['original_ltv']}")
    print(f"  dti:                      {row['dti']}")
    print(f"  borrower_credit_score:    {row['borrower_credit_score']}")
    print(f"  loan_purpose:             {row['loan_purpose']}")
    print(f"  property_type:            {row['property_type']}")
    print(f"  occupancy_status:         {row['occupancy_status']}")
    print(f"  property_state:           {row['property_state']}")
    print(f"  loan_age:                 {row['loan_age']}")
    print(f"  delinquency_status:       {row['current_loan_delinquency_status']}")

    print()
    print("Validation checks:")

    all_passed = True

    # Check 1: loan_id is a long numeric string
    lid = str(row["loan_id"]).strip()
    passed = len(lid) >= 8 and lid.isdigit()
    print(f"  loan_id is numeric (>=8 digits):  {'PASS' if passed else 'FAIL'} -> {lid}")
    all_passed = all_passed and passed

    # Check 2: channel is R, C, or B
    ch = str(row["channel"]).strip()
    passed = ch in ("R", "C", "B")
    print(f"  channel in (R,C,B):               {'PASS' if passed else 'FAIL'} -> {ch}")
    all_passed = all_passed and passed

    # Check 3: property_type is valid
    pt = str(row["property_type"]).strip()
    passed = pt in ("SF", "CO", "PU", "MH", "CP")
    print(f"  property_type in (SF,CO,PU,MH,CP):{'PASS' if passed else 'FAIL'} -> {pt}")
    all_passed = all_passed and passed

    # Check 4: FICO between 300-850
    fico = float(row["borrower_credit_score"])
    passed = 300 <= fico <= 850
    print(f"  FICO between 300-850:             {'PASS' if passed else 'FAIL'} -> {fico}")
    all_passed = all_passed and passed

    # Check 5: LTV between 1-200
    ltv = float(row["original_ltv"])
    passed = 1 <= ltv <= 200
    print(f"  LTV between 1-200:                {'PASS' if passed else 'FAIL'} -> {ltv}")
    all_passed = all_passed and passed

    # Check 6: loan_term is valid
    term = str(row["original_loan_term"]).strip()
    passed = term in ("120", "180", "240", "360")
    print(f"  loan_term in (120,180,240,360):   {'PASS' if passed else 'FAIL'} -> {term}")
    all_passed = all_passed and passed

    # Check 7: state is 2 letters
    state = str(row["property_state"]).strip()
    passed = len(state) == 2 and state.isalpha()
    print(f"  state is 2 letters:               {'PASS' if passed else 'FAIL'} -> {state}")
    all_passed = all_passed and passed

    # Check 8: occupancy is P, S, or I
    occ = str(row["occupancy_status"]).strip()
    passed = occ in ("P", "S", "I", "U")
    print(f"  occupancy in (P,S,I,U):           {'PASS' if passed else 'FAIL'} -> {occ}")
    all_passed = all_passed and passed

    # Check 9: loan_purpose is P, R, C, or U
    lp = str(row["loan_purpose"]).strip()
    passed = lp in ("P", "R", "C", "U")
    print(f"  loan_purpose in (P,R,C,U):        {'PASS' if passed else 'FAIL'} -> {lp}")
    all_passed = all_passed and passed

    print()
    if all_passed:
        print("ALL CHECKS PASSED. Column alignment is correct.")
        print("Safe to run: python -m src.data_pipeline")
    else:
        print("SOME CHECKS FAILED. Do NOT run the bulk pipeline.")
        print("Column alignment needs to be fixed first.")


if __name__ == "__main__":
    main()