FROM joeranbosma/picai_swinnnunetr:latest

RUN apt-get update && \
      apt-get -y install sudo

RUN pip install jupyter matplotlib seaborn

ENTRYPOINT /bin/bash /scripts/interactive_script.sh
