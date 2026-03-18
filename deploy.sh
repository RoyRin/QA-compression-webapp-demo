#!/usr/bin/env bash
set -euo pipefail

# Sync public index.html to api directory
cp code/public/index.html code/api/index.html

# Deploy to Vercel production
cd code
npx vercel --prod
