#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="hailo-llm"
SERVICE_USER="hailo-llm"
SERVICE_GROUP="hailo-llm"
REMOVE_USER=false
PURGE_DATA=false

usage() {
    cat <<'EOF'
Usage: sudo ./uninstall.sh [--remove-user] [--purge-data]

  --remove-user  Remove the hailo-llm system user and group
  --purge-data   Remove model data and operator/runtime configuration
EOF
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --remove-user) REMOVE_USER=true ;;
            --purge-data) PURGE_DATA=true ;;
            -h|--help) usage; exit 0 ;;
            *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
        esac
        shift
    done
}

main() {
    if [[ ${EUID} -ne 0 ]]; then
        echo "Run this uninstaller as root: sudo ./uninstall.sh" >&2
        exit 1
    fi
    parse_args "$@"
    systemctl disable --now "${SERVICE_NAME}.service" 2>/dev/null || true
    rm -f /etc/systemd/system/hailo-llm.service
    rm -rf /opt/hailo-llm
    systemctl daemon-reload

    if [[ "${PURGE_DATA}" == true ]]; then
        rm -rf /var/lib/hailo-llm /etc/hailo/hailo-llm.yaml /etc/hailo/hailo-llm.env /etc/xdg/hailo-llm
    fi
    if [[ "${REMOVE_USER}" == true ]]; then
        userdel hailo-llm 2>/dev/null || true
        groupdel hailo-llm 2>/dev/null || true
    fi
    echo "hailo-llm uninstalled"
}

main "$@"
