"""
Advanced Time Series Forecasting CLI
Production-grade forecasting with Prophet, ARIMA, LSTM, and ensemble methods
"""

import typer
from pathlib import Path
from typing import Optional
import sys

from forecast.cli_commands import train_cmd, predict_cmd, evaluate_cmd, compare_cmd, optimize_cmd

app = typer.Typer(
    name="forecast",
    help="🔮 Advanced Time Series Forecasting CLI with Multiple Algorithms",
    no_args_is_help=True,
)

# Add commands
app.command(name="train")(train_cmd.train)
app.command(name="predict")(predict_cmd.predict)
app.command(name="evaluate")(evaluate_cmd.evaluate)
app.command(name="compare")(compare_cmd.compare)
app.command(name="optimize")(optimize_cmd.optimize)


@app.command()
def version() -> None:
    """Show version information."""
    typer.echo("🔮 forecast-cli v1.0.0")
    typer.echo("Advanced time series forecasting with Prophet, ARIMA, LSTM, and Ensemble methods")


@app.command()
def demo() -> None:
    """Run a quick demo with sample data."""
    import pandas as pd
    import numpy as np
    from forecast.models.prophet_model import ProphetModel
    
    typer.echo("📊 Running quick demo...")
    
    # Generate sample data
    dates = pd.date_range("2020-01-01", periods=365, freq="D")
    trend = np.arange(len(dates)) * 0.05
    seasonality = 10 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365)
    noise = np.random.normal(0, 2, len(dates))
    values = 100 + trend + seasonality + noise
    
    df = pd.DataFrame({
        "ds": dates,
        "y": values
    })
    
    # Train Prophet
    model = ProphetModel()
    model.fit(df)
    
    # Forecast
    future = model.forecast(periods=30)
    
    typer.echo(f"✅ Demo complete!")
    typer.echo(f"📈 Last 5 forecasted values:")
    print(future[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail())


if __name__ == "__main__":
    app()
