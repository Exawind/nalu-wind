ARG BASE_IMG=exawind/exawind-dev:latest
FROM ${BASE_IMG} AS base

WORKDIR /workspace
COPY . /workspace

ARG NUM_PROCS=16
RUN (\
    cmake \
      -Bbuild \
      -DCMAKE_BUILD_TYPE=RELEASE \
      -DCMAKE_PREFIX_PATH=/opt/exawind \
      -DCMAKE_INSTALL_PREFIX=/opt/exawind \
      -DBUILD_SHARED_LIBS=ON \
      -DENABLE_HYPRE=ON -DENABLE_TIOGA=ON -DENABLE_OPENFAST=ON . \
    && cd build \
    && make -j${NUM_PROCS} \
    && make install \
    && export LD_LIBRARY_PATH=/opt/exawind:/usr/local/lib:${LD_LIBRARY_PATH} \
    && ./unittestX --gtest_filter=-Actuator*.* \
    )

RUN (\
    echo "/opt/exawind/lib" > /etc/ld.so.conf.d/exawind.conf \
    && ldconfig \
    )

ENV PATH /opt/exawind/bin:${PATH}
WORKDIR /run
