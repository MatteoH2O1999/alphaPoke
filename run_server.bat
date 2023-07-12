@echo off
cd pokemon-showdown
copy config\config-example.js config\config.js
call npm install
echo 0 > logs/lastbattle.txt
node pokemon-showdown start --no-security