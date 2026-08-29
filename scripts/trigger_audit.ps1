<#
    Fire the Real Data Research Audit from a scheduler that actually fires.

    GitHub's `schedule` event is best-effort and has no SLA: it is delayed under
    load and dropped outright when the queue does not clear. Measured on this
    repository, 29 Aug 2026: four scheduled opportunities, four misses, the last
    one 56 minutes past a target it had 41 minutes to register for. Two
    scheduled runs have ever fired here, both late. Upstream, with an equally
    correct configuration, fired six and missed the same morning.

    `workflow_dispatch` is a direct API call rather than a queued event, and it
    has never failed here -- every manual run succeeded. This script is that
    call, on a timer that belongs to you.

    The GitHub crons stay in place as free backup for nights this machine is
    off. An extra run cannot hurt: the audit's boundary never regresses, so a
    duplicate publishes nothing.

    SETUP (once):
      1. Create a fine-grained personal access token at
         https://github.com/settings/personal-access-tokens/new
           - Repository access : only khd343/Stage
           - Permissions       : Actions -> Read and write
      2. Store it for your user account, NOT in this repository:
           setx GH_AUDIT_TOKEN "github_pat_..."
         Open a new terminal afterwards so the variable is visible.
      3. Register the daily task (runs 19:10 IST, after the 15:30 close):
           schtasks /Create /TN "RS-Stages audit" /SC DAILY /ST 19:10 ^
             /TR "powershell -NoProfile -ExecutionPolicy Bypass -File \"S:\RS-Stages\scripts\trigger_audit.ps1\""

    The token is never written to disk by this script and never enters the
    repository. If it leaks, revoke it at the URL above; its only power is
    starting a workflow in one repository.
#>

$ErrorActionPreference = 'Stop'

$repo = 'khd343/Stage'
$workflow = 'real_data_audit.yml'
$stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'

$token = $env:GH_AUDIT_TOKEN
if ([string]::IsNullOrWhiteSpace($token)) {
    # Fail loudly. A trigger that silently does nothing when unconfigured is
    # worse than no trigger, because it looks like protection that was never
    # wired up -- the exact failure mode upstream's watchdog is stuck in.
    Write-Host "$stamp  FAILED: GH_AUDIT_TOKEN is not set. See the setup notes at the top of this file." -ForegroundColor Red
    exit 1
}

try {
    Invoke-RestMethod -Method Post `
        -Uri "https://api.github.com/repos/$repo/actions/workflows/$workflow/dispatches" `
        -Headers @{
            Authorization          = "Bearer $token"
            Accept                 = 'application/vnd.github+json'
            'X-GitHub-Api-Version' = '2022-11-28'
            'User-Agent'           = 'rs-stages-trigger'
        } `
        -Body '{"ref":"main"}' `
        -ContentType 'application/json' | Out-Null
}
catch {
    Write-Host "$stamp  FAILED to dispatch: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

Write-Host "$stamp  Dispatched $workflow on $repo. It takes about seven minutes." -ForegroundColor Green
