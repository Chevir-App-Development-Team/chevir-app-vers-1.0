#!/usr/bin/env bash
# Saytı lokal işə salır (Linux / macOS). Node.js 20+ lazımdır.
cd "$(dirname "$0")" || exit 1
[ -d node_modules ] || npm install
exec npm run dev -- --open
