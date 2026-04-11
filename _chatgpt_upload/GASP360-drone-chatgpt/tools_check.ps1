Write-Host "Python:"
python --version

Write-Host "`nPip:"
pip --version

Write-Host "`nFFmpeg:"
.\tools\ffmpeg\bin\ffmpeg.exe -version

Write-Host "`nFFmpeg v360 filter:"
.\tools\ffmpeg\bin\ffmpeg.exe -filters | Select-String v360

Write-Host "`nCOLMAP:"
.\COLMAP\COLMAP.bat -h