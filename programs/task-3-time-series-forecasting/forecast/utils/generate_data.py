"""
Generate sample time series data for testing
"""

import pandas as pd
import numpy as np
from pathlib import Path
import typer
from datetime import datetime, timedelta
from rich.console import Console

console = Console()


def generate_sample_data(
    output: Path = typer.Option(Path("data/sample_data.csv"), help="Output CSV path"),
    periods: int = typer.Option(365, help="Number of periods to generate"),
    freq: str = typer.Option("D", help="Frequency: D(daily), H(hourly), W(weekly)"),
    trend: float = typer.Option(0.05, help="Trend magnitude"),
    seasonality: float = typer.Option(10, help="Seasonality amplitude"),
    noise: float = typer.Option(2, help="Noise standard deviation"),
) -> None:
    """
    Generate synthetic time series data for demonstration.
    """
    
    console.print("[bold cyan]📊 Generating Sample Time Series Data[/bold cyan]")
    console.print()
    
    # Generate date range
    dates = pd.date_range(start="2023-01-01", periods=periods, freq=freq)
    
    # Generate components
    trend_component = np.arange(periods) * trend
    seasonality_component = seasonality * np.sin(np.arange(periods) * 2 * np.pi / 365)
    noise_component = np.random.normal(0, noise, periods)
    
    # Combine components
    values = 100 + trend_component + seasonality_component + noise_component
    
    # Create DataFrame
    df = pd.DataFrame({
        'ds': dates,
        'y': values
    })
    
    # Save
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    
    console.print(f"✓ Generated {periods} data points")
    console.print(f"✓ Saved to {output}")
    console.print()
    console.print(f"Data summary:")
    console.print(f"  Mean: {df['y'].mean():.2f}")
    console.print(f"  Std:  {df['y'].std():.2f}")
    console.print(f"  Min:  {df['y'].min():.2f}")
    console.print(f"  Max:  {df['y'].max():.2f}")
