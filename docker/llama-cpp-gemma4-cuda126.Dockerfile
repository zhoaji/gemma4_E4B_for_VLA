FROM ghcr.io/nvidia-ai-iot/llama_cpp@sha256:de16bf712e9614ae5fe6e7230c3807cbe6aba94a949a3aaf2294c2aee1b9ccd1

# Pinned upstream revision with support for the current Gemma 4 E4B GGUF layout.
ARG LLAMA_CPP_REF=d7a2074112d27649303fa107eb8c94db1ee435f3
ARG BUILD_JOBS=4

RUN git -C /opt/llama.cpp fetch --depth 1 origin "$LLAMA_CPP_REF" \
    && git -C /opt/llama.cpp checkout --detach --force FETCH_HEAD \
    && test "$(git -C /opt/llama.cpp rev-parse HEAD)" = "$LLAMA_CPP_REF" \
    && cmake -S /opt/llama.cpp -B /tmp/llama-build \
        -DGGML_CUDA=ON \
        -DLLAMA_BUILD_TESTS=OFF \
        -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /tmp/llama-build --target llama-server -j"$BUILD_JOBS" \
    && install -m 0755 /tmp/llama-build/bin/llama-server /usr/local/bin/llama-server \
    && cp -a /tmp/llama-build/bin/*.so* /usr/local/lib/ \
    && ldconfig \
    && rm -rf /tmp/llama-build
