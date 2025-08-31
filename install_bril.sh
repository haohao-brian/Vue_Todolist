#1. install deno and show version
curl -fsSL https://deno.land/install.sh | sh
export PATH="~/.deno/bin:$PATH"
deno --version

#2. install brili
deno install -g ./brili.ts

#3. install flit
sudo apt install flit

#4. cd into /bril/bril-txt; install uv tool and then brili2txt and brili2json
cd bril
cd brili-txt
sudo snap install astral-uv --classic
uv tool install .
