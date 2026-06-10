# GitHub 업로드 스크립트
# PowerShell에서 실행: .\deploy.ps1

cd "C:\Users\User\Desktop\건축AI 모듈2 7조"

git init
git add .
git commit -m "건축AI 자재관리 대시보드 초기 배포"
git branch -M main
git remote add origin https://github.com/hyunsoo1222/architecture-ai-dashboard.git
git push -u origin main

Write-Host "✅ GitHub 업로드 완료" -ForegroundColor Green
Write-Host "👉 다음 단계: https://share.streamlit.io 접속" -ForegroundColor Cyan
