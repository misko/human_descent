#!/usr/bin/env bash
# Run Python checks without exposing the checkout or its score database to writes.
set -euo pipefail

task_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
task_dir=${HUDES_TEST_RUN_DIR:-$(mktemp -d /tmp/hudes-python-XXXXXXXX)}
task_owns_dir=0
[[ -n ${HUDES_TEST_RUN_DIR:-} ]] || task_owns_dir=1
task_container="hudes-test-$(id -u)-$$-${RANDOM}"
task_child=""

cleanup() {
    # Stop the owned container before waiting for its CLI so shell teardown
    # cannot leave a listener behind.
    if [[ -z ${HUDES_PYTHON:-} ]]; then
        docker stop --time 1 "$task_container" >/dev/null 2>&1 || true
    fi
    if [[ -n $task_child ]]; then
        kill "$task_child" 2>/dev/null || true
        wait "$task_child" 2>/dev/null || true
    fi
    if [[ $task_owns_dir == 1 ]]; then
        rm -rf -- "$task_dir"
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mkdir -p -- "$task_dir"
export HIGH_SCORES_PATH="$task_dir/high-scores.sqlite3"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$task_repo${PYTHONPATH:+:$PYTHONPATH}"
export SDL_VIDEODRIVER=${SDL_VIDEODRIVER:-dummy}
export SDL_AUDIODRIVER=${SDL_AUDIODRIVER:-dummy}
export PYGAME_HIDE_SUPPORT_PROMPT=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

if [[ -n ${HUDES_PYTHON:-} ]]; then
    task_python=$(command -v -- "$HUDES_PYTHON")
    task_native_args=()
    for task_argument in "$@"; do
        if [[ $task_argument == /app/* ]]; then
            task_native_args+=("$task_repo/${task_argument#/app/}")
        else
            task_native_args+=("$task_argument")
        fi
    done
    cd -- "$task_dir"
    "$task_python" "${task_native_args[@]}" &
else
    task_args=(run --rm --init --name "$task_container" --read-only --tmpfs /tmp --shm-size=1g
        --user "$(id -u):$(id -g)"
        -v "$task_repo:/app:ro" -v "$task_dir:$task_dir" -w "$task_dir"
        -e PYTHONPATH=/app -e PYTHONDONTWRITEBYTECODE=1
        -e "HIGH_SCORES_PATH=$HIGH_SCORES_PATH"
        -e SDL_VIDEODRIVER -e SDL_AUDIODRIVER -e PYGAME_HIDE_SUPPORT_PROMPT
        -e OMP_NUM_THREADS -e MKL_NUM_THREADS
        -e "MPLCONFIGDIR=$task_dir/matplotlib")
    for task_key in HUDES_PORT HUDES_HEALTH_PORT HUDES_ADMIN_TOKEN HUDES_SPEED_RUN_SECONDS HUDES_MULTIPLAYER_TEST_SECONDS; do
        if [[ -v $task_key ]]; then task_args+=(-e "$task_key"); fi
    done
    if [[ -d $task_repo/mnist_data ]]; then
        task_args+=(-v "$task_repo/mnist_data:$task_dir/mnist_data:ro")
    fi
    for task_port in ${HUDES_TEST_PORTS:-}; do
        if [[ ! $task_port =~ ^[0-9]+$ ]] || (( task_port < 1024 || task_port > 65535 || task_port == 10000 || task_port == 10001 )); then
            echo "Refusing unsafe test port: $task_port" >&2
            exit 2
        fi
        task_args+=(-p "127.0.0.1:$task_port:$task_port")
    done
    docker "${task_args[@]}" "${HUDES_TEST_IMAGE:-hudes-test}" python "$@" &
fi
task_child=$!
wait "$task_child"
task_child=""
