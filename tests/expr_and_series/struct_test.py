from __future__ import annotations

from typing import Any

import pytest

import narwhals as nw
from tests.utils import Constructor, assert_equal_data


def test_struct_basic(constructor: Constructor) -> None:
    """Test basic struct creation from column names."""
    data = {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}
    df = nw.from_native(constructor(data))
    
    # Create a struct from two columns
    result = df.select(nw.struct("a", "b").alias("my_struct"))
    
    # The result should have a single column with struct data
    schema = result.collect_schema()
    assert "my_struct" in schema
    assert len(result.columns) == 1


def test_struct_with_expressions(constructor: Constructor) -> None:
    """Test struct creation with expressions instead of just column names."""
    data = {"a": [1, 2, 3], "b": [4, 5, 6]}
    df = nw.from_native(constructor(data))
    
    # Create a struct using col() expressions
    result = df.select(nw.struct(nw.col("a"), nw.col("b")).alias("my_struct"))
    
    assert "my_struct" in result.collect_schema()
    assert len(result.columns) == 1


def test_struct_naming_left_hand_rule(constructor: Constructor) -> None:
    """Test that struct naming follows left-hand rule."""
    data = {"a": [1, 2, 3], "b": [4, 5, 6]}
    df = nw.from_native(constructor(data))
    
    # Without alias, the name should come from the first expression
    result = df.select(nw.struct("a", "b"))
    
    # The column should be named after the first input
    assert "a" in result.columns


def test_struct_with_alias(constructor: Constructor) -> None:
    """Test struct with aliasing."""
    data = {"a": [1, 2, 3], "b": [4, 5, 6]}
    df = nw.from_native(constructor(data))
    
    # Test aliasing
    result = df.select(nw.struct("a", "b").alias("my_alias"))
    assert "my_alias" in result.columns
    assert len(result.columns) == 1


def test_struct_mixed_inputs(constructor: Constructor) -> None:
    """Test struct with mixed column names and expressions."""
    data = {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}
    df = nw.from_native(constructor(data))
    
    # Mix column names (strings) and expressions
    result = df.select(nw.struct("a", nw.col("b"), "c").alias("mixed"))
    
    assert "mixed" in result.collect_schema()
    assert len(result.columns) == 1


def test_struct_multi_output_expression(constructor: Constructor) -> None:
    """Test struct with expressions that produce multiple outputs."""
    data = {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}
    df = nw.from_native(constructor(data))
    
    # Use nw.all() which produces multiple columns
    result = df.select(nw.struct(nw.all()).alias("all_struct"))
    
    # Should create a struct with all columns
    assert "all_struct" in result.collect_schema()


def test_struct_single_column(constructor: Constructor) -> None:
    """Test struct with a single column."""
    data = {"a": [1, 2, 3]}
    df = nw.from_native(constructor(data))
    
    result = df.select(nw.struct("a").alias("single"))
    
    assert "single" in result.collect_schema()
    assert len(result.columns) == 1


def test_struct_with_transformations(constructor: Constructor) -> None:
    """Test struct with transformed expressions."""
    data = {"a": [1, 2, 3], "b": [4, 5, 6]}
    df = nw.from_native(constructor(data))
    
    # Create struct with transformed columns
    result = df.select(nw.struct(nw.col("a") * 2, nw.col("b") + 1).alias("transformed"))
    
    assert "transformed" in result.collect_schema()


def test_struct_in_with_columns(constructor: Constructor) -> None:
    """Test struct in with_columns context."""
    data = {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}
    df = nw.from_native(constructor(data))
    
    # Add a struct column alongside existing columns
    result = df.with_columns(my_struct=nw.struct("a", "b"))
    
    assert len(result.columns) == 4  # a, b, c, my_struct
    assert "my_struct" in result.columns


def test_struct_empty_error() -> None:
    """Test that creating a struct with no arguments raises an error."""
    with pytest.raises(ValueError, match="At least one expression"):
        nw.struct()
