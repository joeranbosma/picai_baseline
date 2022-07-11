FROM joeranbosma/picai_swinnnunetr:latest

RUN apt-get update && \
      apt-get -y install sudo

RUN pip3 install jupyter

ENTRYPOINT /bin/bash /scripts/interactive_script.sh
