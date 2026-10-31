export UID := $(shell id -u)
export GID := $(shell id -g)
RUN ?= demo
T ?= 手を振る|指で1、2、3を示す

.PHONY: setup build shell models ros-build ros-demo demo video demo-motion test down
setup: build models ros-build  ## 初回の準備（イメージのビルド、モデルの取得、ROS 2パッケージのビルド）
build:   ## Dockerイメージをビルド
	docker compose build
shell:   ## コンテナに入る
	docker compose run --rm dev bash
models:  ## モデルを取得（Shadow Hand、MediaPipe、顔のVRM）して、描画用のnpmパッケージを入れる
	docker compose run --rm dev bash scripts/fetch_models.sh
ros-build: ## ROS 2パッケージ(sign_avatar)をビルド
	docker compose run --rm dev bash -lc "source /opt/ros/humble/setup.bash && cd ros2_ws && colcon build --symlink-install"
ros-demo:  ## ROS 2の4ノードで再生・録画（例: make ros-demo T="手を振る|指で1、2、3を示す" RUN=demo）
	docker compose run --rm dev bash -lc "source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash && ros2 launch sign_avatar demo.launch.py texts:='$(T)' run_name:=$(RUN)"
demo:    ## 撮影不要のデモ（手書きの動き）を再生・録画 -> outputs/ros_runs/demo/final.mp4
	$(MAKE) ros-demo RUN=demo
video:   ## 自分の動画からアバター動画を作る（例: make video V="data/my_sign.mp4"）
	docker compose run --rm dev python video2avatar.py $(V)
demo-motion: ## 手書きデモの動き（demo/）を作り直す
	docker compose run --rm dev python tools_make_demo_motion.py
test:    ## 環境のスモークテスト（MuJoCo描画）
	docker compose run --rm dev python src/smoke_test.py
down:
	docker compose down
