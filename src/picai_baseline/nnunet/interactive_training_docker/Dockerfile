FROM joeranbosma/picai_swinnnunetr:latest

USER root

RUN apt-get update && \
      apt-get -y install sudo

RUN pip3 install jupyter jupyter_contrib_nbextensions
RUN jupyter contrib nbextension install --user

USER user

WORKDIR /workdir/

ENTRYPOINT /bin/bash /scripts/interactive_script.sh
