FROM node:boron

RUN apt-get update
RUN apt-get -y install curl build-essential git pkg-config
RUN apt-get -y install libcairo2-dev libjpeg-dev libgif-dev

RUN npm install -g nodeunit esprima grunt browserify uglifyjs mongodb
RUN npm install -g probmods/webppl#show-js-stack
# f17cb407c7126895bc65e18825354799fc213dbe

RUN mkdir -p ~/.webppl
RUN npm install --prefix ~/.webppl webppl-timeit webppl-dp webppl-json
RUN npm install --prefix ~/.webppl null-a/webppl-nn#115b6b6413f8ee44a8fa3cce73b1d539895f7d82
RUN npm install --prefix ~/.webppl null-a/webppl-fs#1801afea46148aceace9cf3fd983ae928747df04

RUN mkdir /data
RUN mkdir /data/logs
RUN mkdir /data/config
RUN mkdir /data/results

ENV project_git_url https://github.com/stuhlmueller/neural-nets.git
ENV experiment_cmd "WEBPPL_PARAM_PATH=/data/results/ node --stack-size=100000 `which webppl` models/bimodal-2.wppl --require webppl-nn --require webppl-fs --param-store file --param-id bimodal"

CMD if [ ! -d "/project" ]; then \
      git clone ${project_git_url} /project; cd /project/models/common; npm install; \
    fi && \
    cd /project && \
    if [ -w /data/ ]; then \
      git log -1 > /data/config/git.txt && \
      echo "$OPTIONS" > /data/config/options.txt; \
    fi && \
    eval "$experiment_cmd $OPTIONS"
