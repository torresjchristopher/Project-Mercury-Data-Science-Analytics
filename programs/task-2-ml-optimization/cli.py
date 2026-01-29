"""
ML Model Optimization CLI
Command-line interface for model optimization, compression, and interpretation
"""

import click
import json
import pandas as pd
from pathlib import Path
from typing import Optional
import pickle
import sys

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from optimizer import (
    HyperparameterOptimizer, ModelCompressor, FeatureSelector,
    ModelInterpreter, PerformanceProfiler
)

console = Console()


@click.group()
def cli():
    """ML Model Optimization Suite - Improve, compress, and interpret models"""
    pass


@cli.command()
@click.option('--data', required=True, help='Training data (CSV with features and target)')
@click.option('--target', default='target', help='Target column name')
@click.option('--model', type=click.Choice(['rf', 'gb', 'lr']), default='rf',
              help='Model type: rf (Random Forest), gb (Gradient Boosting), lr (Logistic Regression)')
@click.option('--trials', default=100, help='Number of optimization trials')
@click.option('--timeout', default=600, help='Timeout in seconds')
@click.option('--output', default='optimized_model.pkl', help='Output model path')
def optimize(data, target, model, trials, timeout, output):
    """
    Optimize model hyperparameters using Bayesian optimization
    
    Example:
        ml-optimizer optimize \\
            --data train.csv \\
            --target target_col \\
            --model rf \\
            --trials 50
    """
    
    console.print(Panel.fit(
        "[bold cyan]🎯 ML Hyperparameter Optimization[/bold cyan]",
        title="Starting Optimization"
    ))
    
    # Load data
    if not Path(data).exists():
        console.print(f"[red]Error: Data file not found: {data}[/red]")
        sys.exit(1)
    
    try:
        df = pd.read_csv(data)
        X = df.drop(columns=[target]).values
        y = df[target].values
        
        console.print(f"[cyan]Data loaded: {X.shape[0]} samples, {X.shape[1]} features[/cyan]")
        
        # Optimize
        optimizer = HyperparameterOptimizer(model_type=model, n_trials=trials, timeout=timeout)
        result = optimizer.optimize(X, y)
        
        # Display results
        table = Table(title="Optimization Results")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in result.best_params.items():
            table.add_row(key, str(value))
        
        table.add_row("Best Score", f"{result.best_score:.4f}")
        table.add_row("Trials Run", str(result.n_trials))
        
        console.print(table)
        
        # Save model
        with open(output, 'wb') as f:
            pickle.dump(result.best_model, f)
        
        console.print(f"[green]✓ Model saved to: {output}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        sys.exit(1)


@cli.command()
@click.option('--model', required=True, help='Trained model path (.pkl)')
@click.option('--test-data', required=True, help='Test data (CSV)')
@click.option('--target', default='target', help='Target column name')
def evaluate(model, test_data, target):
    """
    Evaluate model performance on test set
    
    Example:
        ml-optimizer evaluate \\
            --model optimized_model.pkl \\
            --test-data test.csv
    """
    
    console.print("[cyan]Evaluating model...[/cyan]")
    
    # Load model
    if not Path(model).exists():
        console.print(f"[red]Error: Model not found: {model}[/red]")
        sys.exit(1)
    
    # Load data
    if not Path(test_data).exists():
        console.print(f"[red]Error: Test data not found: {test_data}[/red]")
        sys.exit(1)
    
    try:
        with open(model, 'rb') as f:
            model_obj = pickle.load(f)
        
        df = pd.read_csv(test_data)
        X_test = df.drop(columns=[target]).values
        y_test = df[target].values
        
        metrics = PerformanceProfiler.evaluate_model(model_obj, X_test, y_test)
        
        # Display results
        table = Table(title="Performance Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        for metric, value in metrics.items():
            table.add_row(metric, f"{value:.4f}")
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        sys.exit(1)


@cli.command()
@click.option('--model', required=True, help='Trained model path (.pkl)')
@click.option('--train-data', required=True, help='Training data (CSV)')
@click.option('--target', default='target', help='Target column name')
@click.option('--threshold', default=0.01, help='Importance threshold for pruning')
def compress(model, train_data, target, threshold):
    """
    Compress model through pruning and quantization
    
    Example:
        ml-optimizer compress \\
            --model optimized_model.pkl \\
            --train-data train.csv \\
            --threshold 0.005
    """
    
    console.print("[cyan]Compressing model...[/cyan]")
    
    if not Path(model).exists():
        console.print(f"[red]Error: Model not found: {model}[/red]")
        sys.exit(1)
    
    if not Path(train_data).exists():
        console.print(f"[red]Error: Training data not found: {train_data}[/red]")
        sys.exit(1)
    
    try:
        with open(model, 'rb') as f:
            model_obj = pickle.load(f)
        
        df = pd.read_csv(train_data)
        X_train = df.drop(columns=[target]).values
        
        # Prune features
        compressor = ModelCompressor()
        pruning = compressor.prune_features(model_obj, X_train, threshold=threshold)
        
        # Display results
        table = Table(title="Model Compression Analysis")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Prunable Features", pruning['features_reduction'])
        table.add_row("Features to Remove", str(pruning['n_prunable']))
        table.add_row("Threshold", str(threshold))
        
        console.print(table)
        
        # Show prunable features
        if pruning['prunable_features']:
            console.print("\n[cyan]Prunable Features:[/cyan]")
            for feat in pruning['prunable_features'][:10]:  # Show top 10
                console.print(f"  • {feat}")
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        sys.exit(1)


@cli.command()
@click.option('--model', required=True, help='Trained model path (.pkl)')
@click.option('--train-data', required=True, help='Training data (CSV)')
@click.option('--instance', default=0, help='Instance index to explain')
@click.option('--num-features', default=5, help='Number of features to explain')
def explain(model, train_data, instance, num_features):
    """
    Explain model predictions
    
    Example:
        ml-optimizer explain \\
            --model optimized_model.pkl \\
            --train-data train.csv \\
            --instance 0
    """
    
    console.print("[cyan]Explaining model predictions...[/cyan]")
    
    if not Path(model).exists():
        console.print(f"[red]Error: Model not found: {model}[/red]")
        sys.exit(1)
    
    if not Path(train_data).exists():
        console.print(f"[red]Error: Training data not found: {train_data}[/red]")
        sys.exit(1)
    
    try:
        with open(model, 'rb') as f:
            model_obj = pickle.load(f)
        
        df = pd.read_csv(train_data)
        X_train = df.drop(columns=['target']).values if 'target' in df.columns else df.values
        
        interpreter = ModelInterpreter(model_obj, X_train)
        explanation = interpreter.explain_instance(instance, num_features=num_features)
        
        # Display results
        console.print(f"\n[cyan]Prediction for Instance {instance}:[/cyan]")
        console.print(f"  Class: {explanation['prediction']}")
        console.print(f"  Confidence: {max(explanation['probabilities']):.4f}")
        
        console.print(f"\n[cyan]Top {num_features} Contributing Features:[/cyan]")
        for feat, importance in explanation['explanation'].items():
            direction = "[green]↑[/green]" if importance > 0 else "[red]↓[/red]"
            console.print(f"  {direction} {feat}: {importance:.4f}")
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        sys.exit(1)


@cli.command()
@click.option('--train-data', required=True, help='Training data (CSV)')
@click.option('--target', default='target', help='Target column name')
@click.option('--k', default=10, help='Number of top features to select')
def select_features(train_data, target, k):
    """
    Select top K most important features
    
    Example:
        ml-optimizer select-features \\
            --train-data train.csv \\
            --k 10
    """
    
    console.print("[cyan]Selecting top features...[/cyan]")
    
    if not Path(train_data).exists():
        console.print(f"[red]Error: Training data not found: {train_data}[/red]")
        sys.exit(1)
    
    try:
        df = pd.read_csv(train_data)
        X = df.drop(columns=[target]).values
        y = df[target].values
        feature_names = df.drop(columns=[target]).columns.tolist()
        
        result = FeatureSelector.select_top_k(X, y, k=k, feature_names=feature_names)
        
        # Display results
        console.print(f"\n[cyan]Top {k} Features:[/cyan]")
        table = Table()
        table.add_column("Rank", style="cyan")
        table.add_column("Feature", style="green")
        table.add_column("Score", style="yellow")
        
        sorted_features = sorted(result['feature_scores'].items(), key=lambda x: x[1], reverse=True)
        
        for i, (feature, score) in enumerate(sorted_features[:k], 1):
            table.add_row(str(i), feature, f"{score:.4f}")
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        sys.exit(1)


@cli.command()
def benchmark():
    """
    Show ML optimization benchmarks and recommendations
    """
    
    console.print(Panel.fit(
        "[bold cyan]📊 ML Optimization Benchmarks[/bold cyan]",
        title="Performance & Best Practices"
    ))
    
    benchmarks = [
        {
            "technique": "Hyperparameter Optimization (Optuna)",
            "improvement": "5-25%",
            "time": "1-10 hours (100 trials)",
            "best_for": "All model types"
        },
        {
            "technique": "Feature Selection (Top-K)",
            "improvement": "0-5%",
            "time": "Minutes",
            "best_for": "High-dimensional data"
        },
        {
            "technique": "Model Compression",
            "improvement": "-1 to -5%",
            "time": "Seconds",
            "best_for": "Deployment to edge"
        },
        {
            "technique": "Ensemble Methods",
            "improvement": "3-10%",
            "time": "Varies",
            "best_for": "Final production models"
        },
    ]
    
    table = Table(title="Optimization Techniques")
    table.add_column("Technique", style="cyan")
    table.add_column("Improvement", style="green")
    table.add_column("Time Required", style="yellow")
    table.add_column("Best For", style="magenta")
    
    for b in benchmarks:
        table.add_row(b["technique"], b["improvement"], b["time"], b["best_for"])
    
    console.print(table)


@cli.command()
def version():
    """Show version information"""
    console.print("[cyan]ML Model Optimization Suite v1.0.0[/cyan]")
    console.print("[cyan]Built with scikit-learn, Optuna, SHAP, and LIME[/cyan]")


if __name__ == "__main__":
    cli()
