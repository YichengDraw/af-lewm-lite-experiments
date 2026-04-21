$ErrorActionPreference = "Stop"

$stablewmHome = if ($env:STABLEWM_HOME) { $env:STABLEWM_HOME } else { Join-Path $HOME ".stable-wm" }
New-Item -ItemType Directory -Force -Path $stablewmHome | Out-Null

$downloads = @(
    @{
        Name = "pusht_expert_train.h5.zst"
        Url = "https://huggingface.co/datasets/quentinll/lewm-pusht/resolve/main/pusht_expert_train.h5.zst"
        ExpectedBytes = 13136247974
    }
)

foreach ($download in $downloads) {
    $target = Join-Path $stablewmHome $download.Name
    $part = "$target.part"
    Write-Host ""
    Write-Host "Downloading $($download.Name) -> $target"
    if ((Test-Path $target) -and ((Get-Item $target).Length -ge $download.ExpectedBytes)) {
        Write-Host "  SKIP: existing file has expected size"
        continue
    }
    if ((Test-Path $target) -and -not (Test-Path $part)) {
        Move-Item -Force -Path $target -Destination $part
    }
    & curl.exe -L --fail --show-error --retry 5 --retry-delay 5 -C - --output $part $download.Url
    if ($LASTEXITCODE -ne 0) {
        throw "Download failed for $($download.Name)"
    }
    $actualBytes = (Get-Item $part).Length
    if ($actualBytes -lt $download.ExpectedBytes) {
        throw "Download incomplete for $($download.Name): $actualBytes bytes, expected at least $($download.ExpectedBytes)"
    }
    Move-Item -Force -Path $part -Destination $target
}

Write-Host ""
Write-Host "Official PushT dataset download finished."
Write-Host "Next step:"
Write-Host "  python extract_datasets.py"
