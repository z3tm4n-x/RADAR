param(
    [Parameter(Mandatory=$true)][string]$Plan,
    [Parameter(Mandatory=$true)][string]$PlanSha256,
    [Parameter(Mandatory=$true)][string]$Cache
)
# Retain public HTTP response bytes through Windows' native TLS transport.
# No certificate bypass, installation, execution of downloaded code or overwrite.
$ErrorActionPreference = 'Stop'
$planPath = (Resolve-Path -LiteralPath $Plan).Path
if ((Get-FileHash -LiteralPath $planPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $PlanSha256) {
    throw 'GSI fetch plan hash mismatch.'
}
$requests = Get-Content -LiteralPath $planPath -Raw | ConvertFrom-Json
if ($requests.urls.Count -ne 36 -or @($requests.urls | Sort-Object -Unique).Count -ne 36) {
    throw 'Exactly four source pages and 32 unique core anchors are required.'
}
$cachePath = [IO.Path]::GetFullPath($Cache)
$workspace = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$allowedRoot = [IO.Path]::GetFullPath((Join-Path $workspace 'outputs')) + [IO.Path]::DirectorySeparatorChar
if (-not $cachePath.StartsWith($allowedRoot, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Evidence cache must be inside this workspace outputs directory.'
}
[IO.Directory]::CreateDirectory($cachePath) | Out-Null
$utf8 = [Text.UTF8Encoding]::new($false, $true)
$latin1 = [Text.Encoding]::GetEncoding('iso-8859-1')
$sourcePaths = @('/~weick/atima/', '/~weick/atima/atima14.html', '/~weick/atima/atima_readme.txt', '/~weick/atima/atima.txt')
$counter = 0
foreach ($address in $requests.urls) {
    $counter++
    $uri = [Uri]$address
    if ($uri.Scheme -ne 'https' -or $uri.Host -ne 'web-docs.gsi.de' -or $uri.Port -ne 443 -or
        $uri.UserInfo -ne '' -or $uri.Fragment -ne '' -or
        ($uri.AbsolutePath -notin $sourcePaths -and $uri.AbsolutePath -ne '/~weick/atima/atima14_script.cgi')) {
        throw 'Unexpected GSI endpoint.'
    }
    $name = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData([Text.Encoding]::ASCII.GetBytes($address))).ToLowerInvariant()
    $destination = Join-Path $cachePath ($name + '.json')
    if (Test-Path -LiteralPath $destination) {
        Write-Output "GSI evidence $counter/36 already retained; Python replay will validate it."
        continue
    }
    $response = $null
    for ($attempt = 0; $attempt -lt 2; $attempt++) {
        try {
            $response = Invoke-WebRequest -Uri $address -TimeoutSec 25 -MaximumRedirection 0 -Headers @{'User-Agent'='RADAR-scientific-validation/0.1'}
            break
        } catch {
            # Do not retry HTTP error responses or alter certificate validation.
            if ($attempt -eq 1 -or $_.Exception.Response) { throw }
            Write-Output "GSI transport failure at $counter/36; one retry after two seconds."
            Start-Sleep -Seconds 2
        }
    }
    $bytes = $response.RawContentStream.ToArray()
    $contentType = ($response.Headers['Content-Type'] -join ',')
    if ($response.StatusCode -ne 200 -or $bytes.Length -gt 131072 -or
        $response.BaseResponse.RequestMessage.RequestUri.AbsoluteUri -ne $address -or
        $contentType -notmatch '^text/(html|plain)') { throw 'Unexpected HTTP response.' }
    $encoding = 'utf-8'
    try { $body = $utf8.GetString($bytes) }
    catch [Text.DecoderFallbackException] { $encoding = 'iso-8859-1'; $body = $latin1.GetString($bytes) }
    $evidence = [ordered]@{
        url=$address; http_status=200; content_type=$contentType
        retrieved_utc=[DateTimeOffset]::UtcNow.ToString('o')
        sha256=[Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($bytes)).ToLowerInvariant()
        raw_base64=[Convert]::ToBase64String($bytes); text_encoding=$encoding; text=$body
    }
    $encoded = $utf8.GetBytes(($evidence | ConvertTo-Json -Depth 10))
    $stream = [IO.File]::Open($destination, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write)
    try { $stream.Write($encoded, 0, $encoded.Length) } finally { $stream.Dispose() }
    Write-Output "Retained GSI evidence $counter/36 ($($bytes.Length) bytes)."
    Start-Sleep -Seconds 1
}
if ((Get-FileHash -LiteralPath $planPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $PlanSha256) {
    throw 'GSI fetch plan changed during retrieval.'
}
