"""
Evaluate command - assess model performance on test data
"""

import typer
from pathlib import Path
import pandas as pd
from rich.console import Console
from rich.table import Table
import joblib
import numpy as np

console = Console()

def evaluate(
    model_path: Path = typer.Option(..., help="Path to trained model"),
    test_data: Path = typer.Option(..., help="Path to test CSV file"),
    output: typer.Optional[Path] = typer.Option(None, help="Output evaluation report path"),
) -> None:
    """
    Evaluate model performance on test data.
    """
    
    console.print("[bold cyan]📊 Time Series Forecasting CLI - Evaluate[/bold cyan]")
    console.print()
    
    # Load model
    try:
        model = joblib.load(model_path)
        console.print(f"✓ Loaded model from {model_path}")
    except Exception as e:
        console.print(f"[red]✗ Error loading model: {e}[/red]")
        raise typer.Exit(1)
    
    # Load test data
    try:
        df = pd.read_csv(test_data)
        df['ds'] = pd.to_datetime(df['ds'])
        console.print(f"✓ Loaded test data from {test_data}")
        console.print(f"  Samples: {len(df)}")
    except Exception as e:
        console.print(f"[red]✗ Error loading test data: {e}[/red]")
        raise typer.Exit(1)
    
    console.print()
    
    # Generate predictions
    with console.status("[bold cyan]Generating predictions...[/bold cyan]"):
        try:
            forecast_df = model.forecast(len(df))
            predictions = forecast_df['yhat'].values
        except Exception as e:
            console.print(f"[red]✗ Error generating predictions: {e}[/red]")
            raise typer.Exit(1)
    
    # Calculate metrics
    actuals = df['y'].values
    metrics = model.calculate_metrics(actuals, predictions)
    
    console.print("[bold]📈 Model Metrics:[/bold]")
    console.print()
    
    metrics_table = Table(title="Evaluation Metrics")
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", style="green")
    
    for metric_name, value in metrics.items():
        if isinstance(value, float):
            metrics_table.add_row(metric_name, f"{value:.4f}")
        else:
            metrics_table.add_row(metric_name, str(value))
    
    console.print(metrics_table)
    console.print()
    
    # Residual analysis
    residuals = actuals - predictions
    console.print("[bold]📉 Residual Analysis:[/bold]")
    console.print(f"  Mean residual: {np.mean(residuals):.4f}")
    console.print(f"  Std residual: {np.std(residuals):.4f}")
    console.print(f"  Min residual: {np.min(residuals):.4f}")
    console.print(f"  Max residual: {np.max(residuals):.4f}")
    console.print()
    
    # Save report if requested
    if output:
        report = f"""# Model Evaluation Report

## Model: {model.name}

### Metrics
- MAE: {metrics['MAE']:.4f}
- RMSE: {metrics['RMSE']:.4f}
- MAPE: {metrics['MAPE']:.2f}%
- R²: {metrics['R²']:.4f}

### Residual Analysis
- Mean: {np.mean(residuals):.4f}
- Std Dev: {np.std(residuals):.4f}
- Min: {np.min(residuals):.4f}
- Max: {np.max(residuals):.4f}

### Dataset
- Test samples: {len(df)}
- Date range: {df['ds'].min()} to {df['ds'].max()}
"""
        
        with open(output, 'w') as f:
            f.write(report)
        console.print(f"✓ Saved report to {output}")
    
    console.print()
    console.print("[bold green]✅ Evaluation complete![/bold green]")
