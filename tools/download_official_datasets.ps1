$ErrorActionPreference = "Stop"

$stablewmHome = if ($env:STABLEWM_HOME) { $env:STABLEWM_HOME } else { Join-Path $HOME ".stable-wm" }
New-Item -ItemType Directory -Force -Path $stablewmHome | Out-Null

$downloads = @(
    @{
        Name = "pusht_expert_train.h5.zst"
        Url = "https://huggingface.co/datasets/quentinll/lewm-pusht/resolve/main/pusht_expert_train.h5.zst"
    },
    @{
        Name = "tworoom.tar.zst"
        Url = "https://huggingface.co/datasets/quentinll/lewm-tworooms/resolve/main/tworoom.tar.zst"
    },
    @{
        Name = "reacher.tar.zst"
        Url = "https://huggingface.co/datasets/quentinll/lewm-reacher/resolve/main/reacher.tar.zst"
    },
    @{
        Name = "cube_single_expert.tar.zst"
        Url = "https://huggingface.co/datasets/quentinll/lewm-cube/resolve/main/cube_single_expert.tar.zst"
    }
)

foreach ($download in $downloads) {
    $target = Join-Path $stablewmHome $download.Name
    Write-Host ""
    Write-Host "Downloading $($download.Name) -> $target"
    & curl.exe -L -C - --output $target $download.Url
    if ($LASTEXITCODE -ne 0) {
        throw "Download failed for $($download.Name)"
    }
}

Write-Host ""
Write-Host "All dataset downloads finished."
Write-Host "Next step:"
Write-Host "  python extract_datasets.py"
