FROM exawind/exawind-dev as base

WORKDIR /workspace
COPY . /workspace

ARG NUM_PROCS=16
RUN (\
    cmake \
      -Bbuild \
      -DCMAKE_PREFIX_PATH=/opt/exawind \
      -DCMAKE_INSTALL_PREFIX=/opt/exawind \
      -DBUILD_SHARED_LIBS=ON \
      -DENABLE_HYPRE=ON -DENABLE_TIOGA=ON -DENABLE_OPENFAST=ON . \
    && cd build \
    && make -j${NUM_PROCS} \
    && export LD_LIBRARY_PATH=/opt/exawind:/usr/local/lib:${LD_LIBRARY_PATH} \
    && ./unittestX --gtest_filter=-Actuator*.* \
    )
