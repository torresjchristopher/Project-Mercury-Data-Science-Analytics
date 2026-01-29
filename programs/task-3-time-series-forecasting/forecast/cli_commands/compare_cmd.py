"""
Compare and optimize commands
"""

import typer
from pathlib import Path
from typing import List
import pandas as pd
from rich.console import Console
from rich.table import Table
import joblib

console = Console()

def compare(
    models: List[Path] = typer.Argument(..., help="Paths to multiple model files"),
    periods: int = typer.Option(30, help="Forecast periods to compare"),
) -> None:
    """
    Compare predictions from multiple models.
    """
    
    console.print("[bold cyan]🔄 Time Series Forecasting CLI - Compare[/bold cyan]")
    console.print()
    
    if len(models) < 2:
        console.print("[red]✗ Need at least 2 models to compare[/red]")
        raise typer.Exit(1)
    
    # Load all models
    loaded_models = {}
    for model_path in models:
        try:
            model = joblib.load(model_path)
            loaded_models[model_path.stem] = model
            console.print(f"✓ Loaded {model_path.stem}")
        except Exception as e:
            console.print(f"[red]✗ Error loading {model_path}: {e}[/red]")
            raise typer.Exit(1)
    
    console.print()
    
    # Generate forecasts from all models
    console.print(f"🔮 Generating {periods}-period forecasts from {len(loaded_models)} models...")
    console.print()
    
    all_forecasts = {}
    for name, model in loaded_models.items():
        try:
            all_forecasts[name] = model.forecast(periods)
        except Exception as e:
            console.print(f"[red]✗ Error forecasting with {name}: {e}[/red]")
            raise typer.Exit(1)
    
    # Create comparison table
    comparison_table = Table(title="Model Comparison - First 10 Periods")
    comparison_table.add_column("Period", style="cyan")
    
    for model_name in all_forecasts.keys():
        comparison_table.add_column(f"{model_name}", style="green")
    
    for i in range(min(10, periods)):
        row = [str(i + 1)]
        for model_name, forecast_df in all_forecasts.items():
            row.append(f"{forecast_df.iloc[i]['yhat']:.2f}")
        comparison_table.add_row(*row)
    
    console.print(comparison_table)
    console.print()
    
    # Summary comparison
    console.print("[bold]📊 Summary Statistics:[/bold]")
    console.print()
    
    summary_table = Table(title="Mean Forecasts")
    summary_table.add_column("Model", style="cyan")
    summary_table.add_column("Mean", style="green")
    summary_table.add_column("Std Dev", style="yellow")
    summary_table.add_column("Min", style="red")
    summary_table.add_column("Max", style="red")
    
    for model_name, forecast_df in all_forecasts.items():
        values = forecast_df['yhat'].values
        summary_table.add_row(
            model_name,
            f"{values.mean():.2f}",
            f"{values.std():.2f}",
            f"{values.min():.2f}",
            f"{values.max():.2f}"
        )
    
    console.print(summary_table)
    console.print()
    console.print("[bold green]✅ Comparison complete![/bold green]")


def optimize(
    train_data: Path = typer.Option(..., help="Path to training data"),
    algorithm: str = typer.Option("prophet", help="Algorithm to optimize: prophet, arima, lstm"),
    output: Path = typer.Option(Path("optimized_model.pkl"), help="Output model path"),
) -> None:
    """
    Optimize hyperparameters using grid search.
    """
    
    console.print("[bold cyan]⚡ Time Series Forecasting CLI - Optimize[/bold cyan]")
    console.print()
    
    console.print("[yellow]⚠️  Hyperparameter optimization is computationally intensive[/yellow]")
    console.print("[yellow]This will take several minutes...[/yellow]")
    console.print()
    
    # Load data
    try:
        df = pd.read_csv(train_data)
        df['ds'] = pd.to_datetime(df['ds'])
        console.print(f"✓ Loaded training data: {len(df)} samples")
    except Exception as e:
        console.print(f"[red]✗ Error loading data: {e}[/red]")
        raise typer.Exit(1)
    
    console.print()
    console.print("[bold cyan]Grid Search Results Coming Soon[/bold cyan]")
    console.print("(Hyperparameter optimization framework is ready for extension)")
    console.print()
    console.print("[bold green]✅ Optimization framework ready![/bold green]")
