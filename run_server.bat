@echo off
cd pokemon-showdown
git pull
call npm install
echo 0 > logs/lastbattle.txt
node pokemon-showdown start --no-security