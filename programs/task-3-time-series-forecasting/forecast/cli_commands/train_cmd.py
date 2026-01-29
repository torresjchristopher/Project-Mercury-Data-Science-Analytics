"""
Train command - train forecasting models on data
"""

import typer
from pathlib import Path
from typing import Optional
import pandas as pd
from rich.console import Console
from rich.table import Table
import joblib

from forecast.models.prophet_model import ProphetModel
from forecast.models.arima_model import ARIMAModel
from forecast.models.lstm_model import LSTMModel
from forecast.models.ensemble_model import EnsembleModel

console = Console()

def train(
    data: Path = typer.Argument(..., help="Path to CSV file with 'ds' and 'y' columns"),
    output: Path = typer.Option(Path("models"), help="Output directory for models"),
    algorithm: str = typer.Option("prophet", help="Algorithm: prophet, arima, lstm, or ensemble"),
    test_size: float = typer.Option(0.2, help="Test set size (0-1)"),
) -> None:
    """
    Train forecasting models on historical data.
    
    Algorithms:
    - prophet: Facebook's Prophet (default, handles seasonality)
    - arima: ARIMA (classical statistical model)
    - lstm: LSTM neural network (deep learning)
    - ensemble: Combination of all three
    """
    
    console.print("[bold cyan]⏱️ Time Series Forecasting CLI - Train[/bold cyan]")
    console.print()
    
    # Load data
    try:
        df = pd.read_csv(data)
        console.print(f"✓ Loaded data from {data}")
        console.print(f"  Shape: {df.shape}")
        console.print(f"  Columns: {', '.join(df.columns)}")
    except Exception as e:
        console.print(f"[red]✗ Error loading data: {e}[/red]")
        raise typer.Exit(1)
    
    # Validate columns
    if 'ds' not in df.columns or 'y' not in df.columns:
        console.print("[red]✗ CSV must have 'ds' (datetime) and 'y' (values) columns[/red]")
        raise typer.Exit(1)
    
    # Convert ds to datetime
    try:
        df['ds'] = pd.to_datetime(df['ds'])
    except Exception as e:
        console.print(f"[red]✗ Error converting 'ds' to datetime: {e}[/red]")
        raise typer.Exit(1)
    
    # Split data
    split_idx = int(len(df) * (1 - test_size))
    train_df = df[:split_idx]
    test_df = df[split_idx:]
    
    console.print(f"✓ Split data:")
    console.print(f"  Train: {len(train_df)} samples")
    console.print(f"  Test: {len(test_df)} samples")
    console.print()
    
    # Create output directory
    output.mkdir(parents=True, exist_ok=True)
    
    # Train model(s)
    with console.status("[bold cyan]Training model...[/bold cyan]"):
        try:
            if algorithm == "prophet":
                model = ProphetModel()
                model.fit(train_df)
                models_to_save = [("prophet", model)]
                
            elif algorithm == "arima":
                model = ARIMAModel(auto=True)
                model.fit(train_df)
                models_to_save = [("arima", model)]
                
            elif algorithm == "lstm":
                model = LSTMModel(epochs=50)
                model.fit(train_df)
                models_to_save = [("lstm", model)]
                
            elif algorithm == "ensemble":
                prophet_model = ProphetModel()
                arima_model = ARIMAModel(auto=True)
                lstm_model = LSTMModel(epochs=50)
                
                ensemble = EnsembleModel([prophet_model, arima_model, lstm_model])
                ensemble.fit(train_df)
                models_to_save = [("ensemble", ensemble)]
                
            else:
                console.print(f"[red]✗ Unknown algorithm: {algorithm}[/red]")
                raise typer.Exit(1)
                
        except Exception as e:
            console.print(f"[red]✗ Error training model: {e}[/red]")
            raise typer.Exit(1)
    
    console.print("✓ Training complete")
    console.print()
    
    # Evaluate on test set
    console.print("[bold]📊 Evaluation on Test Set:[/bold]")
    console.print()
    
    evaluation_table = Table(title="Model Performance")
    evaluation_table.add_column("Model", style="cyan")
    evaluation_table.add_column("MAE", style="magenta")
    evaluation_table.add_column("RMSE", style="magenta")
    evaluation_table.add_column("MAPE %", style="magenta")
    evaluation_table.add_column("R²", style="magenta")
    
    for model_name, model in models_to_save:
        # Get predictions on test set
        forecast_df = model.forecast(len(test_df))
        predictions = forecast_df['yhat'].values
        actuals = test_df['y'].values
        
        # Calculate metrics
        metrics = model.calculate_metrics(actuals, predictions)
        
        evaluation_table.add_row(
            model_name,
            f"{metrics['MAE']:.4f}",
            f"{metrics['RMSE']:.4f}",
            f"{metrics['MAPE']:.2f}",
            f"{metrics['R²']:.4f}"
        )
    
    console.print(evaluation_table)
    console.print()
    
    # Save models
    console.print("[bold]💾 Saving Models:[/bold]")
    for model_name, model in models_to_save:
        model_path = output / f"{model_name}_model.pkl"
        joblib.dump(model, model_path)
        console.print(f"✓ Saved {model_name} to {model_path}")
    
    console.print()
    console.print("[bold green]✅ Training complete![/bold green]")
    console.print()
    console.print("Next steps:")
    console.print(f"  forecast predict --model-path {output}/model.pkl --periods 30")
    console.print(f"  forecast evaluate --model-path {output}/model.pkl --test-data {data}")
