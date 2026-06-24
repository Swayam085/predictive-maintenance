# tests/test_data_loader.py
# Author      : Vrushabh (Data Engineer)
# Description : Unit tests for data_loader.py

import pytest
import pandas as pd
import numpy as np
import os
import sys

sys.path.append("src")
from data_loader import load_raw, clean, make_splits


class TestLoadRaw:

    def test_shape(self):
        df = load_raw()
        assert df.shape == (10000, 14), \
            f"Expected (10000, 14), got {df.shape}"

    def test_columns_present(self):
        df = load_raw()
        required = ["Machine failure", "Type", "Tool wear [min]"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_no_missing_values(self):
        df = load_raw()
        assert df.isnull().sum().sum() == 0, "Missing values found!"

    def test_no_duplicates(self):
        df = load_raw()
        assert df.duplicated().sum() == 0, "Duplicate rows found!"


class TestClean:

    def test_output_shape(self):
        df = load_raw()
        df_clean = clean(df)
        assert df_clean.shape == (10000, 7), \
            f"Expected (10000, 7), got {df_clean.shape}"

    def test_type_encoded(self):
        df = load_raw()
        df_clean = clean(df)
        assert set(df_clean["Type"].unique()).issubset({0, 1, 2}), \
            "Type column not encoded correctly!"

    def test_no_identifier_cols(self):
        df = load_raw()
        df_clean = clean(df)
        assert "UDI" not in df_clean.columns
        assert "Product_ID" not in df_clean.columns

    def test_no_failure_subtypes(self):
        df = load_raw()
        df_clean = clean(df)
        for col in ["TWF", "HDF", "PWF", "OSF", "RNF"]:
            assert col not in df_clean.columns, \
                f"Failure subtype {col} still present!"

    def test_failure_rate(self):
        df = load_raw()
        df_clean = clean(df)
        rate = df_clean["Machine_failure"].mean() * 100
        assert 3.0 <= rate <= 4.0, \
            f"Unexpected failure rate: {rate:.2f}%"


class TestMakeSplits:

    def test_split_sizes(self):
        df = pd.read_csv("data/processed/featured_data.csv")
        splits = make_splits(df)
        assert len(splits["X_train"]) == 7000
        assert len(splits["X_val"])   == 1500
        assert len(splits["X_test"])  == 1500

    def test_no_overlap(self):
        df = pd.read_csv("data/processed/featured_data.csv")
        splits = make_splits(df)
        train_idx = set(splits["X_train"].index)
        val_idx   = set(splits["X_val"].index)
        test_idx  = set(splits["X_test"].index)
        assert len(train_idx & val_idx)  == 0
        assert len(train_idx & test_idx) == 0
        assert len(val_idx   & test_idx) == 0

    def test_stratified_failure_rate(self):
        df = pd.read_csv("data/processed/featured_data.csv")
        splits = make_splits(df)
        for name in ["y_train", "y_val", "y_test"]:
            rate = splits[name].mean() * 100
            assert 3.0 <= rate <= 4.0, \
                f"{name} failure rate unexpected: {rate:.2f}%"