param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs
)

$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$explicitPython = $null
for ($i = 0; $i -lt $RemainingArgs.Count; $i++) {
    if ($RemainingArgs[$i] -eq "--python" -and ($i + 1) -lt $RemainingArgs.Count) {
        $explicitPython = $RemainingArgs[$i + 1]
        break
    }
}

function Test-HealthyPython {
    param(
        [string[]]$CommandParts,
        [switch]$RequireVenv
    )
    try {
        $probe = "import encodings, pip"
        if ($RequireVenv) {
            $probe = "import encodings, pip, venv"
        }
        if ($CommandParts.Count -eq 1) {
            & $CommandParts[0] -c $probe *> $null
        }
        else {
            & $CommandParts[0] $CommandParts[1..($CommandParts.Count - 1)] -c $probe *> $null
        }
        return ($LASTEXITCODE -eq 0)
    }
    catch {
        return $false
    }
}

if ($explicitPython) {
    if (-not (Test-HealthyPython @($explicitPython) -RequireVenv:$false)) {
        throw "Explicit Python interpreter is not usable: $explicitPython"
    }
    & $explicitPython "$RootDir\\scripts\\bootstrap_dev.py" @RemainingArgs
    exit $LASTEXITCODE
}

$candidates = @()
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if ($pythonCmd) { $candidates += ,@($pythonCmd.Source) }
$pyCmd = Get-Command py -ErrorAction SilentlyContinue
if ($pyCmd) {
    foreach ($version in @("-3.12", "-3.11", "-3.10", "")) {
        if ($version) {
            $candidates += ,@($pyCmd.Source, $version)
        }
        else {
            $candidates += ,@($pyCmd.Source)
        }
    }
}

foreach ($candidate in $candidates) {
    if (Test-HealthyPython $candidate -RequireVenv) {
        if ($candidate.Count -eq 1) {
            & $candidate[0] "$RootDir\\scripts\\bootstrap_dev.py" @RemainingArgs
        }
        else {
            & $candidate[0] $candidate[1..($candidate.Count - 1)] "$RootDir\\scripts\\bootstrap_dev.py" @RemainingArgs
        }
        exit $LASTEXITCODE
    }
}

throw "Unable to locate a healthy Python interpreter for bootstrap."
