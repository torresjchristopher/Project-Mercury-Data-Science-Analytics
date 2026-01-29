"""
Predict command - generate forecasts from trained models
"""

import typer
from pathlib import Path
from typing import Optional
import pandas as pd
from rich.console import Console
from rich.table import Table
import joblib

console = Console()

def predict(
    model_path: Path = typer.Option(..., help="Path to trained model pickle file"),
    periods: int = typer.Option(30, help="Number of periods to forecast"),
    confidence: float = typer.Option(0.95, help="Confidence level for intervals"),
    output: Optional[Path] = typer.Option(None, help="Output CSV path (optional)"),
) -> None:
    """
    Generate forecasts from a trained model.
    """
    
    console.print("[bold cyan]🔮 Time Series Forecasting CLI - Predict[/bold cyan]")
    console.print()
    
    # Load model
    try:
        model = joblib.load(model_path)
        console.print(f"✓ Loaded model from {model_path}")
    except Exception as e:
        console.print(f"[red]✗ Error loading model: {e}[/red]")
        raise typer.Exit(1)
    
    # Generate forecasts
    with console.status("[bold cyan]Generating forecasts...[/bold cyan]"):
        try:
            forecast_df = model.forecast(periods)
        except Exception as e:
            console.print(f"[red]✗ Error generating forecast: {e}[/red]")
            raise typer.Exit(1)
    
    console.print("✓ Forecast generated")
    console.print()
    
    # Display results
    console.print("[bold]📈 Forecast Results:[/bold]")
    console.print()
    
    forecast_table = Table(title=f"Forecasted {periods} Periods")
    forecast_table.add_column("Period", style="cyan")
    forecast_table.add_column("Forecast", style="green")
    forecast_table.add_column(f"{int(confidence*100)}% Lower", style="yellow")
    forecast_table.add_column(f"{int(confidence*100)}% Upper", style="yellow")
    
    for i, row in forecast_df.iterrows():
        forecast_table.add_row(
            str(i + 1),
            f"{row['yhat']:.2f}",
            f"{row['yhat_lower']:.2f}",
            f"{row['yhat_upper']:.2f}"
        )
    
    console.print(forecast_table)
    console.print()
    
    # Summary statistics
    console.print("[bold]📊 Summary Statistics:[/bold]")
    console.print(f"  Mean forecast: {forecast_df['yhat'].mean():.2f}")
    console.print(f"  Min: {forecast_df['yhat'].min():.2f}")
    console.print(f"  Max: {forecast_df['yhat'].max():.2f}")
    console.print(f"  Std Dev: {forecast_df['yhat'].std():.2f}")
    console.print()
    
    # Save if requested
    if output:
        try:
            forecast_df.to_csv(output, index=False)
            console.print(f"✓ Saved forecast to {output}")
        except Exception as e:
            console.print(f"[red]✗ Error saving forecast: {e}[/red]")
            raise typer.Exit(1)
    
    console.print()
    console.print("[bold green]✅ Prediction complete![/bold green]")
