#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SERVICE_NAME="hailo-llm"
SERVICE_USER="hailo-llm"
SERVICE_GROUP="hailo-llm"
SERVICE_DIR="/opt/hailo-llm"
DATA_DIR="/var/lib/hailo-llm"
VENV_DIR="${SERVICE_DIR}/venv"
UNIT_DEST="/etc/systemd/system/hailo-llm.service"
ETC_HAILO_CONFIG="/etc/hailo/hailo-llm.yaml"
ETC_XDG_DIR="/etc/xdg/hailo-llm"
JSON_CONFIG="${ETC_XDG_DIR}/hailo-llm.json"
ENV_FILE="/etc/hailo/hailo-llm.env"
SOCKET_PATH="${HAILO_DEVICE_SOCKET:-/run/hailo/device.sock}"
DEFAULT_PORT="11441"

log() { echo "[hailo-llm] $*"; }
error() { echo "[hailo-llm] ERROR: $*" >&2; }

usage() {
    cat <<'EOF'
Usage: sudo ./install.sh

Install the Hailo LLM API service. Model HEF files must be installed at the
paths listed in /etc/hailo/hailo-llm.yaml before requesting those models.
EOF
}

require_root() {
    if [[ ${EUID} -ne 0 ]]; then
        error "Run this installer as root: sudo ./install.sh"
        exit 1
    fi
}

parse_args() {
    if [[ $# -gt 0 ]]; then
        if [[ "$1" == "-h" || "$1" == "--help" ]]; then
            usage
            exit 0
        fi
        error "Unknown option: $1"
        usage
        exit 1
    fi
}

preflight() {
    for command in python3 systemctl curl; do
        if ! command -v "${command}" >/dev/null 2>&1; then
            error "Missing required command: ${command}"
            exit 1
        fi
    done
    if [[ ! -S "${SOCKET_PATH}" ]]; then
        error "Device-manager socket not found at ${SOCKET_PATH}; install and start hailo-device-manager first"
        exit 1
    fi
}

create_user_group() {
    if ! getent group "${SERVICE_GROUP}" >/dev/null; then
        groupadd --system "${SERVICE_GROUP}"
    fi
    if ! id "${SERVICE_USER}" >/dev/null 2>&1; then
        useradd -r -s /usr/sbin/nologin -d "${DATA_DIR}" -g "${SERVICE_GROUP}" "${SERVICE_USER}"
    fi

    local socket_group
    socket_group="$(stat -c '%G' "${SOCKET_PATH}")"
    if [[ -z "${socket_group}" ]] || ! getent group "${socket_group}" >/dev/null; then
        error "Could not determine the device-manager socket group"
        exit 1
    fi
    usermod -aG "${socket_group}" "${SERVICE_USER}"
}

install_files() {
    install -d -m 0755 "${SERVICE_DIR}" "${DATA_DIR}/models"
    install -d -m 0755 /etc/hailo "${ETC_XDG_DIR}"
    install -m 0644 "${SCRIPT_DIR}/hailo_llm_server.py" "${SERVICE_DIR}/hailo_llm_server.py"
    install -m 0644 "${SCRIPT_DIR}/config.py" "${SERVICE_DIR}/config.py"
    install -m 0644 "${SCRIPT_DIR}/render_config.py" "${SERVICE_DIR}/render_config.py"
    install -m 0644 "${REPO_ROOT}/device_manager/device_client.py" "${SERVICE_DIR}/device_client.py"
    install -m 0644 "${SCRIPT_DIR}/requirements.txt" "${SERVICE_DIR}/requirements.txt"
    install -m 0644 "${SCRIPT_DIR}/hailo-llm.service" "${UNIT_DEST}"
    if [[ ! -f "${ETC_HAILO_CONFIG}" ]]; then
        install -m 0644 "${SCRIPT_DIR}/config.yaml" "${ETC_HAILO_CONFIG}"
    else
        log "Preserving existing config ${ETC_HAILO_CONFIG}"
    fi
    if [[ ! -f "${ENV_FILE}" ]]; then
        cat >"${ENV_FILE}" <<EOF
# Environment overrides for hailo-llm
HAILO_DEVICE_SOCKET=${SOCKET_PATH}
# NPU_MAX_GENERATED_TOKENS=128
EOF
        chmod 0644 "${ENV_FILE}"
    fi
    chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "${DATA_DIR}"
    chown "${SERVICE_USER}:${SERVICE_GROUP}" "${ETC_XDG_DIR}"
    chmod 0750 "${ETC_XDG_DIR}"
    chown -R root:root "${SERVICE_DIR}"
    chmod 0755 "${SERVICE_DIR}/hailo_llm_server.py"
}

install_python_environment() {
    python3 -m venv "${VENV_DIR}"
    "${VENV_DIR}/bin/pip" install --disable-pip-version-check -r "${SERVICE_DIR}/requirements.txt"
}

render_config() {
    "${VENV_DIR}/bin/python3" "${SERVICE_DIR}/render_config.py" \
        --input "${ETC_HAILO_CONFIG}" --output "${JSON_CONFIG}"
    chown root:"${SERVICE_GROUP}" "${ETC_HAILO_CONFIG}"
    chmod 0640 "${ETC_HAILO_CONFIG}"
    chown "${SERVICE_USER}:${SERVICE_GROUP}" "${JSON_CONFIG}"
    chmod 0640 "${JSON_CONFIG}"
}

warn_if_port_in_use() {
    local port="$1"
    if command -v ss >/dev/null 2>&1 && ss -lnt 2>/dev/null | awk '{print $4}' | grep -q ":${port}$"; then
        log "WARNING: port ${port} is already in use; adjust ${ETC_HAILO_CONFIG} if needed"
    fi
}

configured_port() {
    "${VENV_DIR}/bin/python3" - "${ETC_HAILO_CONFIG}" <<'PY'
import sys
import yaml
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    print(int((yaml.safe_load(handle) or {}).get("server", {}).get("port", 11441)))
PY
}

start_service() {
    systemctl daemon-reload
    systemctl enable --now "${SERVICE_NAME}.service"
    local port
    port="$(configured_port)"
    local attempt
    for attempt in {1..15}; do
        if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
            log "Health check succeeded at http://127.0.0.1:${port}/health"
            return
        fi
        sleep 1
    done
    error "Service did not become ready; inspect: journalctl -u ${SERVICE_NAME}.service -n 100 --no-pager"
    return 1
}

main() {
    parse_args "$@"
    require_root
    preflight
    create_user_group
    install_files
    install_python_environment
    render_config
    warn_if_port_in_use "$(configured_port)"
    start_service
    log "Installed. Edit ${ETC_HAILO_CONFIG}; add HEFs under /var/lib/hailo-llm/models."
}

main "$@"
