# 手話Physical AI 最終課題 開発環境（Ubuntu 22.04 / Python 3.10）
# Apple Silicon(arm64)でもIntelでも動く。GPUなしのためMuJoCoはosmesa(ソフトウェア描画)を使う。
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Tokyo \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MUJOCO_GL=osmesa \
    PYOPENGL_PLATFORM=osmesa

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-venv python3-dev \
      build-essential git curl ca-certificates ffmpeg \
      libosmesa6 libosmesa6-dev libgl1 libegl1 libgles2 libglu1-mesa libglib2.0-0 libsm6 libxext6 libxrender1 \
      fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

# 顔の合成（three.js + three-vrm を headless Chromium で描画）用の Node.js 22
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# 非rootユーザ（ホストのUID/GIDに合わせると、出力ファイルの所有者がずれない）
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} dev 2>/dev/null || true \
    && useradd -m -u ${UID} -g ${GID} -s /bin/bash dev 2>/dev/null || useradd -m -s /bin/bash dev

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv ${VIRTUAL_ENV} && chown -R ${UID}:${GID} ${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

WORKDIR /workspace
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && pip install -r /tmp/requirements.txt \
    && chown -R ${UID}:${GID} ${VIRTUAL_ENV}

# Chromium（Playwright）。顔の合成の描画に使う。ブラウザは共有パスに置く
ENV PLAYWRIGHT_BROWSERS_PATH=/opt/pw-browsers
RUN pip install playwright opencv-python-headless Pillow \
    && playwright install --with-deps chromium \
    && chmod -R a+rX /opt/pw-browsers \
    && rm -rf /var/lib/apt/lists/*

# ROS 2 Humble（ros-base + colcon + rosbag2）。仮想環境のPythonでノードを動かす
RUN apt-get update && apt-get install -y --no-install-recommends gnupg lsb-release \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu jammy main" > /etc/apt/sources.list.d/ros2.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends ros-humble-ros-base python3-colcon-common-extensions ros-humble-ros2bag ros-humble-rosbag2-storage-default-plugins \
    && rm -rf /var/lib/apt/lists/*
ENV ROS_DISTRO=humble

USER dev
CMD ["bash"]
