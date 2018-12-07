FROM alpine:3.6
RUN echo "http://dl-cdn.alpinelinux.org/alpine/latest-stable/main" > /etc/apk/repositories
RUN echo "http://dl-cdn.alpinelinux.org/alpine/latest-stable/community" >> /etc/apk/repositories
RUN apk --no-cache --update-cache add gcc gfortran python3 python3-dev py3-pip build-base wget freetype-dev libpng-dev openblas-dev git linux-headers ca-certificates libstdc++ cmake g++ make musl-dev

RUN ln -s /usr/include/locale.h /usr/include/xlocale.h

RUN git clone https://github.com/tbenthompson/tectosaur.git
WORKDIR /tectosaur

RUN python3 -V
RUN pip3 install numpy
RUN pip3 install .
RUN pip3 install -U pip
RUN pip3 install jupyterlab
RUN pip3 install pycuda

ENTRYPOINT /usr/bin/jupyter lab --no-browser --ip=0.0.0.0 --allow-root --port 9999
EXPOSE 9999
