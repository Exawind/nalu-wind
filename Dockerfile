FROM exawind/exw-tioga:latest AS tioga
FROM exawind/exw-openfast:latest AS openfast
FROM exawind/exw-trilinos:latest AS trilinos
FROM exawind/exw-dev-deps:latest AS base

COPY --from=tioga /opt/exawind /opt/exawind
COPY --from=openfast /opt/exawind /opt/exawind
COPY --from=trilinos /opt/exawind /opt/exawind

WORKDIR /nalu-build
COPY . /nalu-build

ARG ENABLE_OPENMP=OFF
ARG ENABLE_CUDA=OFF
RUN (\
    cmake \
      -Bbuild \
      -DCMAKE_BUILD_TYPE=RELEASE \
      -DCMAKE_PREFIX_PATH=/opt/exawind \
      -DCMAKE_INSTALL_PREFIX=/opt/exawind \
      -DBUILD_SHARED_LIBS=ON \
      -DENABLE_OPENMP=${ENABLE_OPENMP} \
      -DENABLE_CUDA=${ENABLE_CUDA} \
      -DENABLE_HYPRE=ON -DENABLE_TIOGA=ON -DENABLE_OPENFAST=ON \
      -G Ninja    . \
    && cd build \
    && ninja -j$(nproc) \
    && export LD_LIBRARY_PATH=/opt/exawind:/usr/local/lib:${LD_LIBRARY_PATH} \
    && ./unittestX --gtest_filter=-Actuator*.* \
    && ninja install \
    )

RUN (\
    echo "/opt/exawind/lib" > /etc/ld.so.conf.d/exawind.conf \
    && ldconfig \
    )
ENV PATH /opt/exawind/bin:${PATH}
WORKDIR /workspace
