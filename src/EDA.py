

from pathlib import Path
from typing import Sequence
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

###############################################################################
# I/O Functions
###############################################################################

def load_dataset(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Load a CSV or Excel dataset."""
    path = Path(path)
    if path.suffix in {".csv"}:
        df = pd.read_csv(path, nrows=nrows)
    elif path.suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path, nrows=nrows)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    return df


###############################################################################
# Visualization Functions
###############################################################################

def plot_numerical_distributions(df: pd.DataFrame, cols: Sequence[str] = None, bins: int = 30) -> None:
    """Plot histograms for numeric columns."""
    if cols is None:
        cols = df.select_dtypes("number").columns
    nrows = (len(cols) + 2) // 3  # Calculate rows needed
    fig, axes = plt.subplots(nrows, 3, figsize=(12, nrows * 3))
    for ax, col in zip(axes.flatten(), cols):
        sns.histplot(df[col].dropna(), bins=bins, ax=ax, kde=True)
        ax.set_title(col)
    plt.tight_layout()
    plt.show()

def plot_categorical_distributions(df: pd.DataFrame, cols: Sequence[str] = None, top_n: int = 15) -> None:
    """Create bar charts for categorical columns."""
    if cols is None:
        cols = df.select_dtypes(exclude="number").columns
    nrows = (len(cols) + 2) // 3  # Calculate rows needed
    fig, axes = plt.subplots(nrows, 3, figsize=(12, nrows * 3))
    for ax, col in zip(axes.flatten(), cols):
        vc = df[col].value_counts().nlargest(top_n)
        sns.barplot(x=vc.values, y=vc.index, ax=ax)
        ax.set_title(col)
    plt.tight_layout()
    plt.show()

def plot_correlational_heatmap(df: pd.DataFrame) -> None:
    """Display a heatmap of numeric variable correlations."""
    corr = df.select_dtypes("number").corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()

def boxplot_outliers(df: pd.DataFrame, cols: Sequence[str] = None) -> None:
    """Create boxplots to inspect outliers in numeric columns."""
    if cols is None:
        cols = df.select_dtypes("number").columns
    nrows = (len(cols) + 2) // 3  # Calculate rows needed
    fig, axes = plt.subplots(nrows, 3, figsize=(12, nrows * 3))
    for ax, col in zip(axes.flatten(), cols):
        sns.boxplot(x=df[col], ax=ax)
        ax.set_title(col)
    plt.tight_layout()
    plt.show()