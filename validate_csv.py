"""
Script untuk validasi format CSV EEG sebelum upload ke Streamlit
Jalankan: python validate_csv.py path/to/your/file.csv
"""

import sys
import pandas as pd
import numpy as np
import os

def validate_csv(file_path):
    """
    Validasi format CSV untuk EEG classification
    
    Requirements:
    - 20 channels (columns)
    - ≥ 800 samples (rows)
    - No header
    - Pure numeric data
    - No NaN/Inf values
    """
    print("="*80)
    print("CSV VALIDATION FOR EEG CLASSIFICATION")
    print("="*80 + "\n")
    
    # Check file exists
    if not os.path.exists(file_path):
        print(f"❌ ERROR: File not found: {file_path}")
        return False
    
    print(f"📁 File: {file_path}")
    print(f"📊 Size: {os.path.getsize(file_path) / 1024:.2f} KB\n")
    
    # Try load CSV
    try:
        print("⏳ Loading CSV...")
        
        # Try without header first (correct format)
        data = pd.read_csv(file_path, header=None)
        
        print(f"✓ Loaded successfully")
        print(f"  Shape: {data.shape} (rows × columns)\n")
        
    except Exception as e:
        print(f"❌ ERROR loading CSV: {e}")
        return False
    
    # Validation checks
    issues = []
    warnings = []
    
    print("="*80)
    print("VALIDATION CHECKS")
    print("="*80 + "\n")
    
    # Check 1: Number of columns (must be 20)
    print("1. Checking number of channels...")
    if data.shape[1] == 20:
        print(f"   ✅ PASS: 20 channels (columns)")
    else:
        print(f"   ❌ FAIL: {data.shape[1]} channels, expected 20")
        issues.append(f"Wrong channel count: {data.shape[1]} (expected 20)")
    
    # Check 2: Number of rows (must be ≥ 800)
    print("\n2. Checking number of samples...")
    duration = data.shape[0] / 200  # Assuming 200 Hz
    if data.shape[0] >= 800:
        print(f"   ✅ PASS: {data.shape[0]} samples ({duration:.2f} seconds)")
    else:
        print(f"   ❌ FAIL: {data.shape[0]} samples (only {duration:.2f} seconds)")
        print(f"   Required: ≥ 800 samples (≥ 4 seconds)")
        issues.append(f"Too few samples: {data.shape[0]} (minimum 800)")
    
    # Check 3: Data type (must be numeric)
    print("\n3. Checking data types...")
    non_numeric = data.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric) == 0:
        print(f"   ✅ PASS: All columns are numeric")
    else:
        print(f"   ❌ FAIL: {len(non_numeric)} non-numeric columns found")
        issues.append(f"Non-numeric columns: {list(non_numeric)}")
    
    # Check 4: NaN values
    print("\n4. Checking for NaN values...")
    nan_count = data.isna().sum().sum()
    if nan_count == 0:
        print(f"   ✅ PASS: No NaN values")
    else:
        print(f"   ⚠️  WARNING: {nan_count} NaN values found")
        warnings.append(f"Contains {nan_count} NaN values (will be filled with mean)")
    
    # Check 5: Inf values
    print("\n5. Checking for Inf values...")
    inf_count = np.isinf(data.select_dtypes(include=[np.number])).sum().sum()
    if inf_count == 0:
        print(f"   ✅ PASS: No Inf values")
    else:
        print(f"   ⚠️  WARNING: {inf_count} Inf values found")
        warnings.append(f"Contains {inf_count} Inf values")
    
    # Check 6: Data range (sanity check)
    print("\n6. Checking data range...")
    data_min = data.min().min()
    data_max = data.max().max()
    data_mean = data.mean().mean()
    data_std = data.std().mean()
    
    print(f"   Range: [{data_min:.4f}, {data_max:.4f}]")
    print(f"   Mean : {data_mean:.4f}")
    print(f"   Std  : {data_std:.4f}")
    
    if abs(data_min) > 1000 or abs(data_max) > 1000:
        print(f"   ⚠️  WARNING: Data range seems unusual for EEG")
        warnings.append("Data range unusual (|values| > 1000)")
    else:
        print(f"   ✅ PASS: Data range looks reasonable")
    
    # Check 7: Zero variance
    print("\n7. Checking for zero variance channels...")
    zero_var_cols = (data.std() == 0).sum()
    if zero_var_cols == 0:
        print(f"   ✅ PASS: All channels have variance")
    else:
        print(f"   ⚠️  WARNING: {zero_var_cols} channels have zero variance")
        warnings.append(f"{zero_var_cols} channels with zero variance")
    
    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80 + "\n")
    
    if len(issues) == 0 and len(warnings) == 0:
        print("✅ ✅ ✅  ALL CHECKS PASSED! ✅ ✅ ✅")
        print("\nYour CSV is ready for upload to Streamlit app!")
        return True
    
    elif len(issues) == 0:
        print("⚠️  PASSED WITH WARNINGS")
        print(f"\nFound {len(warnings)} warning(s):")
        for i, warning in enumerate(warnings, 1):
            print(f"  {i}. {warning}")
        print("\nCSV can be used, but may need preprocessing.")
        return True
    
    else:
        print("❌  VALIDATION FAILED")
        print(f"\nFound {len(issues)} critical issue(s):")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        
        if warnings:
            print(f"\nAdditional warning(s):")
            for i, warning in enumerate(warnings, 1):
                print(f"  {i}. {warning}")
        
        print("\n" + "="*80)
        print("HOW TO FIX")
        print("="*80)
        
        if "Wrong channel count" in str(issues):
            print("\n❌ Wrong Channel Count:")
            print("  → Your CSV must have exactly 20 columns (channels)")
            print("  → Check if there are extra columns (timestamp, label, etc)")
            print("  → Remove non-EEG columns")
        
        if "Too few samples" in str(issues):
            print("\n❌ Too Few Samples:")
            print("  → Your CSV must have at least 800 rows (4 seconds @ 200 Hz)")
            print("  → Current duration:", f"{data.shape[0] / 200:.2f} seconds")
            print("  → Use longer recording or concatenate multiple files")
        
        if "Non-numeric columns" in str(issues):
            print("\n❌ Non-numeric Data:")
            print("  → CSV must contain only numeric values")
            print("  → Remove text columns (names, labels, timestamps)")
            print("  → Save as CSV without header")
        
        return False


def fix_csv(input_file, output_file):
    """
    Attempt to auto-fix common CSV issues
    """
    print("\n" + "="*80)
    print("AUTO-FIX ATTEMPT")
    print("="*80 + "\n")
    
    try:
        # Load
        data = pd.read_csv(input_file, header=None)
        
        # Fix 1: Remove non-numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        
        # Fix 2: Fill NaN with mean
        numeric_data = numeric_data.fillna(numeric_data.mean())
        
        # Fix 3: Replace Inf with max/min
        numeric_data = numeric_data.replace([np.inf, -np.inf], [numeric_data.max().max(), numeric_data.min().min()])
        
        # Save
        numeric_data.to_csv(output_file, index=False, header=False)
        
        print(f"✓ Fixed CSV saved to: {output_file}")
        print(f"  Original shape: {data.shape}")
        print(f"  Fixed shape   : {numeric_data.shape}")
        
        # Validate fixed file
        print("\nValidating fixed file...")
        return validate_csv(output_file)
        
    except Exception as e:
        print(f"❌ Auto-fix failed: {e}")
        return False


if __name__ == "__main__":
    print("\n" + "#"*80)
    print("# CSV VALIDATOR FOR EEG CLASSIFICATION")
    print("#"*80 + "\n")
    
    if len(sys.argv) < 2:
        print("Usage: python validate_csv.py <path_to_csv_file>")
        print("\nExample:")
        print("  python validate_csv.py data/my_eeg_data.csv")
        print("\nOptional: Auto-fix")
        print("  python validate_csv.py input.csv --fix output_fixed.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # Check if fix requested
    if "--fix" in sys.argv and len(sys.argv) >= 4:
        output_file = sys.argv[3]
        success = fix_csv(input_file, output_file)
    else:
        success = validate_csv(input_file)
    
    print("\n" + "="*80)
    if success:
        print("✅ VALIDATION COMPLETE: PASSED")
    else:
        print("❌ VALIDATION COMPLETE: FAILED")
    print("="*80 + "\n")
    
    sys.exit(0 if success else 1)