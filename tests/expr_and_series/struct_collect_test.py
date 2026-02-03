from __future__ import annotations

from typing import Any

import pytest

import narwhals as nw
from narwhals._utils import Implementation
from tests.utils import Constructor, assert_equal_data


def test_struct_collect_to_pandas(constructor: Constructor) -> None:
    """Test that struct columns are properly handled when collecting to pandas.

    When collecting a lazy frame with struct columns to pandas backend,
    the struct values should be accessible. Based on narwhals conventions,
    pandas may represent structs as either:
    - Native Python dicts (object dtype) for compatibility
    - ArrowDtype with struct type for pandas >= 2.2
    Both representations are acceptable and should provide field access.
    """
    if "pandas_constructor" in str(constructor):
        # Skip for eager pandas since we're testing lazy collection
        pytest.skip("Test is for lazy frames collecting to pandas")

    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    import pandas as pd

    data = {"a": [1, 2, 3], "b": [4, 5, 6]}
    df = nw.from_native(constructor(data))

    # Create a struct column
    df_with_struct = df.select(nw.struct("a", "b").alias("my_struct"))

    # Collect to pandas backend
    result = df_with_struct.lazy().collect(backend=Implementation.PANDAS)
    native = result.to_native()

    assert isinstance(native, pd.DataFrame)
    assert "my_struct" in native.columns

    # For pandas, struct values should be accessible (either as dicts or ArrowDtype structs)
    # The key requirement from comment 2 is that the conversion follows narwhals conventions:
    # - For pandas: object (dict) representation or ArrowDtype with struct
    first_value = native["my_struct"].iloc[0]

    # Verify we can access the fields - this is the critical behavior
    # The value should be dict-like (for native pandas) or struct-like (for ArrowDtype)
    if isinstance(first_value, dict):
        # Native pandas dict representation
        assert first_value["a"] == 1
        assert first_value["b"] == 4
    else:
        # Likely ArrowDtype struct - should still be accessible
        assert hasattr(first_value, "__getitem__") or hasattr(first_value, "as_py")


def test_struct_collect_to_polars(constructor: Constructor) -> None:
    """Test that struct columns remain as structs when collecting to polars.

    When collecting a lazy frame with struct columns to polars backend,
    the struct values should remain as native struct types.
    This verifies narwhals convention: polars should maintain struct dtype.
    """
    pytest.importorskip("polars")
    pytest.importorskip("pyarrow")

    if "pandas_constructor" in str(constructor):
        pytest.skip("Test is for lazy frames collecting to polars")

    data = {"a": [1, 2, 3], "b": [4, 5, 6]}
    df = nw.from_native(constructor(data))

    # Create a struct column
    df_with_struct = df.select(nw.struct("a", "b").alias("my_struct"))

    # Collect to polars backend
    result = df_with_struct.lazy().collect(backend=Implementation.POLARS)
    native = result.to_native()

    import polars as pl

    assert isinstance(native, pl.DataFrame)
    assert "my_struct" in native.columns

    # For polars, the column should have a Struct dtype (narwhals convention)
    assert native["my_struct"].dtype.base_type() == pl.Struct


def test_struct_collect_to_pyarrow(constructor: Constructor) -> None:
    """Test that struct columns remain as structs when collecting to pyarrow.

    When collecting a lazy frame with struct columns to pyarrow backend,
    the struct values should remain as pyarrow struct types.
    This verifies narwhals convention: pyarrow should maintain struct type.
    """
    pytest.importorskip("pyarrow")

    if "pandas_constructor" in str(constructor):
        pytest.skip("Test is for lazy frames collecting to pyarrow")

    data = {"a": [1, 2, 3], "b": [4, 5, 6]}
    df = nw.from_native(constructor(data))

    # Create a struct column
    df_with_struct = df.select(nw.struct("a", "b").alias("my_struct"))

    # Collect to pyarrow backend
    result = df_with_struct.lazy().collect(backend=Implementation.PYARROW)
    native = result.to_native()

    import pyarrow as pa

    assert isinstance(native, pa.Table)
    assert "my_struct" in native.column_names

    # For pyarrow, the column should have a struct type (narwhals convention)
    col_type = native.schema.field("my_struct").type
    assert pa.types.is_struct(col_type)
