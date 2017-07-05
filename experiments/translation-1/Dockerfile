FROM continuumio/anaconda3

RUN conda install pytorch torchvision -c soumith -y
RUN conda install jupyter -y

RUN apt-get install -y gcc
RUN apt-get install -y g++

RUN conda install -y nltk
RUN conda install -y spacy
RUN conda install -y tqdm
RUN conda install -y requests
RUN conda install -y flake8
RUN conda install -y pytest
RUN conda install -y pytest-cov
RUN conda install unidecode -y

RUN pip install codecov
RUN pip install pytest-pythonpath

RUN cd /tmp && git clone https://github.com/pytorch/text.git && cd /tmp/text && python setup.py install

RUN python -m spacy.en.download

RUN mkdir /data
RUN mkdir /data/logs
RUN mkdir /data/config
RUN mkdir /data/results

ENV project_git_url https://github.com/stuhlmueller/neural-nets.git
ENV experiment_cmd "python models/pytorch/translation.py"

CMD if [ ! -d "/project" ]; then \
      git clone ${project_git_url} /project; \
    fi && \
    cd /project && \
    if [ -w /data/ ]; then \
      git log -1 > /data/config/git.txt && \
      echo "$OPTIONS" > /data/config/options.txt; \
    fi && \
    eval "$experiment_cmd $OPTIONS"
