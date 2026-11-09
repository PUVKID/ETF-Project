param([int]$Port = 8879)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$projectPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    throw '프로젝트 가상환경을 먼저 만들고 requirements.txt의 패키지를 설치해 주세요.'
}
# 현재 창에서 서버와 중지 가능한 지속 작업을 함께 실행합니다.
& $projectPython -B -X utf8 server.py --port $Port --continuous
exit $LASTEXITCODE
