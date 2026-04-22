#Requires -Version 5.1
<#
.SYNOPSIS
    Bytevion_ local deployment — starts the Byte API server and the frontend demo server.

.DESCRIPTION
    1. Installs / reuses the project venv at .\.venv
    2. Starts byte_server on http://127.0.0.1:8000  (API + gateway)
    3. Starts a Python HTTP server for the frontend on http://localhost:3000
    4. Opens the demo in your default browser
    5. Ctrl-C shuts everything down cleanly

.PARAMETER AdminToken
    Admin token for Byte protected endpoints. Defaults to 'byte-demo-admin'.

.PARAMETER Port
    Port for the Byte API server. Defaults to 8000.

.PARAMETER FrontendPort
    Port for the demo frontend. Defaults to 3000.

.PARAMETER CacheMode
    Byte gateway cache mode: semantic | hybrid | exact | normalized. Defaults to 'hybrid' (recommended).

.PARAMETER GatewayMode
    Byte gateway dispatch mode: adaptive | backend. Defaults to 'adaptive'.

.PARAMETER NoBrowser
    Skip auto-opening the browser.

.EXAMPLE
    .\start.ps1
    .\start.ps1 -AdminToken "my-secret" -CacheMode hybrid
    .\start.ps1 -Port 8080 -FrontendPort 5000 -NoBrowser
#>
param(
    [string]$AdminToken     = "byte-demo-admin",
    [int]   $Port           = 8000,
    [int]   $FrontendPort   = 3000,
    [ValidateSet("semantic","hybrid","exact","normalized")]
    [string]$CacheMode      = "hybrid",
    [ValidateSet("adaptive","backend")]
    [string]$GatewayMode    = "adaptive",
    [string]$OpenAIKey      = "",
    [string]$AnthropicKey   = "",
    [string]$DeepSeekKey    = "",
    [string]$GroqKey        = "",
    [string]$MistralKey     = "",
    [string]$OpenRouterKey  = "",
    [string]$BindHost       = "0.0.0.0",   # bind to all interfaces for external access
    [switch]$NoBrowser
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot

# ── Helpers ────────────────────────────────────────────────────────────────────

function Get-PublicIP {
    # Try each strategy; return the first non-loopback IPv4 we find
    try {
        $ip = (Get-NetIPAddress -AddressFamily IPv4 -ErrorAction Stop |
               Where-Object { $_.IPAddress -notmatch '^(127\.|169\.254\.)' -and
                              $_.PrefixOrigin -notin @('WellKnown','Link') } |
               Sort-Object -Property { $_.IPAddress } |
               Select-Object -First 1).IPAddress
        if ($ip) { return $ip }
    } catch {}
    # Fallback: ask ipify (requires internet)
    try { return (Invoke-RestMethod 'https://api.ipify.org?format=text' -TimeoutSec 4 -ErrorAction Stop).Trim() } catch {}
    return "YOUR_VM_IP"
}

function Write-Banner {
    Write-Host ""
    Write-Host "  >Bytevion_  Deployment" -ForegroundColor Cyan
    Write-Host "  ─────────────────────────────────────────────" -ForegroundColor DarkGray
    Write-Host "  API Server  : http://${BindHost}:$Port" -ForegroundColor White
    Write-Host "  Frontend    : http://${BindHost}:$FrontendPort" -ForegroundColor White
    Write-Host "  Cache Mode  : $CacheMode" -ForegroundColor White
    Write-Host "  Gateway Mode: $GatewayMode" -ForegroundColor White
    Write-Host "  Admin Token : $AdminToken" -ForegroundColor DarkGray
    Write-Host "  ─────────────────────────────────────────────" -ForegroundColor DarkGray
    Write-Host ""
}

function Write-Step([string]$msg) {
    Write-Host "  » $msg" -ForegroundColor Cyan
}

function Write-OK([string]$msg) {
    Write-Host "  ✓ $msg" -ForegroundColor Green
}

function Write-Warn([string]$msg) {
    Write-Host "  ⚠ $msg" -ForegroundColor Yellow
}

function Write-Fail([string]$msg) {
    Write-Host "  ✗ $msg" -ForegroundColor Red
}

# ── Venv & Install ─────────────────────────────────────────────────────────────

$VenvDir    = Join-Path $Root ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$VenvPip    = Join-Path $VenvDir "Scripts\pip.exe"
$ByteServer = Join-Path $VenvDir "Scripts\byte_server.exe"

Write-Banner

if (-not (Test-Path $VenvPython)) {
    Write-Step "Creating virtual environment…"
    python -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) {
        Write-Fail "Failed to create venv. Ensure Python 3.10+ is on PATH."
        exit 1
    }
    Write-OK "Virtual environment created"
}

if (-not (Test-Path $ByteServer)) {
    Write-Step "Installing Byte with server + OpenAI extras (first run — may take ~2 min)…"
    & $VenvPip install -e "$Root\.[server,openai]" --quiet
    if ($LASTEXITCODE -ne 0) {
        Write-Fail "Installation failed. Check network and try again."
        exit 1
    }
    Write-OK "Byte installed"
} else {
    Write-OK "Byte already installed"
}

# ── Port availability check ────────────────────────────────────────────────────

function Test-PortFree([int]$p) {
    $conn = Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue
    return ($null -eq $conn)
}

if (-not (Test-PortFree $Port)) {
    Write-Warn "Port $Port is in use. Byte server may already be running — or choose a different port with -Port."
}
if (-not (Test-PortFree $FrontendPort)) {
    Write-Warn "Port $FrontendPort is in use. Choose a different port with -FrontendPort."
}

# ── Data directory ─────────────────────────────────────────────────────────────

$CacheDir = Join-Path $Root "byte_data"
if (-not (Test-Path $CacheDir)) {
    New-Item -ItemType Directory -Path $CacheDir | Out-Null
    Write-OK "Created cache directory: byte_data"
}

# ── Start Byte API Server ──────────────────────────────────────────────────────

Write-Step "Starting Byte API server on port $Port…"

$AliasFile = Join-Path $Root "byte_aliases.json"

$ByteArgs = @(
    "--host", $BindHost,
    "--port", "$Port",
    "--cache-dir", $CacheDir,
    "--gateway", "True",
    "--gateway-mode", $GatewayMode,
    "--gateway-cache-mode", $CacheMode,
    "--security-admin-token", $AdminToken,
    "--router-aliases-file", $AliasFile,
    "--cors-origins", "*"
)

# Build environment block so the Byte router can reach all configured providers
$ByteEnv = [System.Environment]::GetEnvironmentVariables()
if ($OpenAIKey)     { $ByteEnv["OPENAI_API_KEY"]     = $OpenAIKey     }
if ($AnthropicKey)  { $ByteEnv["ANTHROPIC_API_KEY"]  = $AnthropicKey  }
if ($DeepSeekKey)   { $ByteEnv["DEEPSEEK_API_KEY"]   = $DeepSeekKey   }
if ($GroqKey)       { $ByteEnv["GROQ_API_KEY"]       = $GroqKey       }
if ($MistralKey)    { $ByteEnv["MISTRAL_API_KEY"]    = $MistralKey    }
if ($OpenRouterKey) { $ByteEnv["OPENROUTER_API_KEY"] = $OpenRouterKey }

$ByteProc = Start-Process `
    -FilePath $ByteServer `
    -ArgumentList $ByteArgs `
    -WorkingDirectory $Root `
    -PassThru `
    -NoNewWindow `
    -RedirectStandardOutput (Join-Path $Root "byte_server.log") `
    -RedirectStandardError  (Join-Path $Root "byte_server_err.log")

Write-OK "Byte server started (PID $($ByteProc.Id))"
Write-Host "    Logs → byte_server.log / byte_server_err.log" -ForegroundColor DarkGray

# Wait for server to become ready
Write-Step "Waiting for Byte server to become ready…"
$ready   = $false
$timeout = 30
$waited  = 0
while (-not $ready -and $waited -lt $timeout) {
    Start-Sleep -Seconds 1
    $waited++
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/healthz" -TimeoutSec 2 -ErrorAction Stop
        if ($r.StatusCode -eq 200) { $ready = $true }
    } catch { <# still starting #> }
}

if ($ready) {
    Write-OK "Byte server is ready (${waited}s)"
} else {
    Write-Warn "Byte server did not respond in ${timeout}s — check byte_server_err.log"
    Write-Warn "The frontend will still open; reconnect once the server is up."
}

# ── Frontend HTTP Server ───────────────────────────────────────────────────────

Write-Step "Starting frontend server on port $FrontendPort…"

$FrontendScript = @"
import http.server, os
os.chdir(r'$Root')
class H(http.server.SimpleHTTPRequestHandler):
    def log_message(self, fmt, *a): pass
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()
http.server.HTTPServer(('$BindHost', $FrontendPort), H).serve_forever()
"@

$FrontendProc = Start-Process `
    -FilePath $VenvPython `
    -ArgumentList @("-c", $FrontendScript) `
    -WorkingDirectory $Root `
    -PassThru `
    -NoNewWindow `
    -RedirectStandardOutput (Join-Path $Root "frontend.log") `
    -RedirectStandardError  (Join-Path $Root "frontend_err.log")

Start-Sleep -Seconds 1
Write-OK "Frontend server started (PID $($FrontendProc.Id))"

# ── Open Browser ───────────────────────────────────────────────────────────────

$DemoUrl = "http://localhost:$FrontendPort/demo.html"

if (-not $NoBrowser) {
    Write-Step "Opening browser → $DemoUrl"
    Start-Sleep -Milliseconds 500
    Start-Process $DemoUrl
}

# ── Summary ────────────────────────────────────────────────────────────────────

$PublicIP = Get-PublicIP
$ExternalDemo = "http://${PublicIP}:$FrontendPort/demo.html"
$ExternalAPI  = "http://${PublicIP}:$Port"

Write-Host ""
Write-Host "  ┌──────────────────────────────────────────────────────────┐" -ForegroundColor Cyan
Write-Host "  │  >Bytevion_ is running                                   │" -ForegroundColor Cyan
Write-Host "  │                                                          │" -ForegroundColor Cyan
Write-Host "  │  LOCAL                                                   │" -ForegroundColor Cyan
Write-Host "  │    Frontend  →  http://localhost:$FrontendPort/demo.html" -ForegroundColor White
Write-Host "  │    API       →  http://localhost:$Port" -ForegroundColor White
Write-Host "  │                                                          │" -ForegroundColor Cyan
Write-Host "  │  EXTERNAL (VM public IP)                                 │" -ForegroundColor Cyan
Write-Host "  │    Frontend  →  $ExternalDemo" -ForegroundColor Green
Write-Host "  │    API       →  $ExternalAPI" -ForegroundColor Green
Write-Host "  │                                                          │" -ForegroundColor Cyan
Write-Host "  │  Health   →  http://localhost:$Port/healthz              │" -ForegroundColor DarkGray
Write-Host "  │  Admin token : $AdminToken" -ForegroundColor DarkGray
Write-Host "  │                                                          │" -ForegroundColor Cyan
Write-Host "  │  Press Ctrl-C to stop all services                       │" -ForegroundColor Cyan
Write-Host "  └──────────────────────────────────────────────────────────┘" -ForegroundColor Cyan
Write-Host ""
Write-Host "  NOTE: Ensure VM firewall/NSG allows inbound TCP on ports $Port and $FrontendPort" -ForegroundColor Yellow
Write-Host ""

# ── Shutdown hook ──────────────────────────────────────────────────────────────

try {
    # Block until Ctrl-C
    while ($true) {
        Start-Sleep -Seconds 2

        # Restart crashed processes
        if ($ByteProc.HasExited) {
            Write-Warn "Byte server exited unexpectedly (code $($ByteProc.ExitCode)) — check byte_server_err.log"
        }
        if ($FrontendProc.HasExited) {
            Write-Warn "Frontend server exited unexpectedly — restarting…"
            $FrontendProc = Start-Process `
                -FilePath $VenvPython `
                -ArgumentList @("-c", $FrontendScript) `
                -WorkingDirectory $Root `
                -PassThru -NoNewWindow `
                -RedirectStandardOutput (Join-Path $Root "frontend.log") `
                -RedirectStandardError  (Join-Path $Root "frontend_err.log")
        }
    }
} finally {
    Write-Host ""
    Write-Step "Shutting down…"
    if (-not $ByteProc.HasExited)    { Stop-Process -Id $ByteProc.Id    -Force -ErrorAction SilentlyContinue }
    if (-not $FrontendProc.HasExited){ Stop-Process -Id $FrontendProc.Id -Force -ErrorAction SilentlyContinue }
    Write-OK "All services stopped. Goodbye."
}
