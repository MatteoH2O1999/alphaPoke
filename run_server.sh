git submodule update --remote
cd pokemon-showdown
cp config/config-example.js config/config.js
sed -i 's/exports.repl = true;/exports.repl = false;/' ./config/config.js
npm install
printf 0 > logs/lastbattle.txt
node pokemon-showdown start --no-security