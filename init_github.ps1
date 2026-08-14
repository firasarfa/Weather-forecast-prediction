#!/usr/bin/env pwsh
# DevSecOps-clean git init script for Weather Forecast LSTM Project
# This script strips old git history, initializes a fresh repo,
# and pushes to a new remote URL with clean, scrubbed files.

# 1. Remove existing .git folder (if any) to strip commit history
if (Test-Path -Path "./.git") {
    Remove-Item -Path "./.git" -Recurse -Force
    Write-Host "[INFO] Removed existing .git folder."
} else {
    Write-Host "[INFO] No existing .git folder found. Proceeding with fresh init."
}

# 2. Initialize new git repository
git init

# 3. Stage all files (respecting .gitignore)
git add .

# 4. Commit all files with initial message
git commit -m "Initial commit: Weather forecast LSTM prediction project

- LSTM multivariate time-series model for hourly temperature forecasting
- Data preprocessing & feature engineering (lags, rolling stats, seasonal flags)
- Model: 2-layer LSTM with L2 regularization, dropout, Huber loss
- Evaluation: MSE/MAE/R² with baseline comparison
- Visualizations: actual vs predicted, residuals, error over time"

# 5. Create new remote and push (replace USERNAME/REPO with your GitHub values)
# NOTE: You must first create the repository on GitHub.com manually,
# then uncomment and run the push command below:
#
# git remote add origin https://github.com/USERNAME/weather-forecast-prediction.git
# git branch -M main
# git push -u origin main

Write-Host ""
Write-Host "[SUCCESS] Git repository initialized and committed locally."
Write-Host "[NEXT] Create the repo on GitHub.com, then run:"
Write-Host "  git remote add origin https://github.com/USERNAME/weather-forecast-prediction.git"
Write-Host "  git branch -M main"
Write-Host "  git push -u origin main"