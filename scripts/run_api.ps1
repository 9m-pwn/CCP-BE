param(
    [string]$EnvName = "poc-env"
)

$ErrorActionPreference = "Stop"

Write-Host "Starting API with conda env '$EnvName'..."
conda run -n $EnvName python run_api.py
