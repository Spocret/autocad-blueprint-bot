# ─── Stage 1: build dxf2dwg from LibreDWG source ────────────────────────────
FROM python:3.11-slim AS libredwg-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    autoconf \
    automake \
    libtool \
    pkg-config \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q "https://ftp.gnu.org/gnu/libredwg/libredwg-0.13.3.tar.xz" \
    && tar xf libredwg-0.13.3.tar.xz \
    && cd libredwg-0.13.3 \
    && ./configure --prefix=/usr/local \
    && make -j"$(nproc)" \
    && make install \
    && rm -rf /libredwg-0.13.3 /libredwg-0.13.3.tar.xz

# ─── Stage 2: production image ───────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy only the dxf2dwg binary and its shared library from the build stage
COPY --from=libredwg-builder /usr/local/bin/dxf2dwg /usr/local/bin/dxf2dwg
COPY --from=libredwg-builder /usr/local/lib/libdwg* /usr/local/lib/
RUN ldconfig

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
