#!/bin/bash
# =============================================================================
# SSH Key Setup Script for PEST++ Agent Deployment
# =============================================================================
# This script helps set up SSH key-based authentication to remote machines,
# which is required for the deploy-agents.sh script to work.
#
# Usage:
#   ./setup-ssh-keys.sh [user@]host [user@]host ...
#
# Or to set up from agents.conf:
#   ./setup-ssh-keys.sh --from-config
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/agents.conf"

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

setup_ssh_key() {
    local target=$1

    # Parse user@host
    if [[ "$target" == *"@"* ]]; then
        local user="${target%@*}"
        local host="${target#*@}"
    else
        local user="$USER"
        local host="$target"
    fi

    echo ""
    log_info "Setting up SSH key for ${user}@${host}..."

    # Check if we can already connect without password
    if ssh -o BatchMode=yes -o ConnectTimeout=5 "${user}@${host}" "echo ok" &>/dev/null; then
        log_success "SSH key already set up for ${user}@${host}"
        return 0
    fi

    # Generate SSH key if it doesn't exist
    if [[ ! -f ~/.ssh/id_rsa ]] && [[ ! -f ~/.ssh/id_ed25519 ]]; then
        log_info "No SSH key found. Generating new key..."
        ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N "" -q
        log_success "SSH key generated: ~/.ssh/id_ed25519"
    fi

    # Copy SSH key to remote
    log_info "Copying SSH key to ${user}@${host}..."
    log_warn "You will be prompted for the password for ${user}@${host}"

    if ssh-copy-id -o ConnectTimeout=10 "${user}@${host}"; then
        log_success "SSH key copied to ${user}@${host}"

        # Verify
        if ssh -o BatchMode=yes -o ConnectTimeout=5 "${user}@${host}" "echo ok" &>/dev/null; then
            log_success "Verified: Can now connect without password"
            return 0
        else
            log_error "Verification failed - still requires password"
            return 1
        fi
    else
        log_error "Failed to copy SSH key to ${user}@${host}"
        return 1
    fi
}

print_usage() {
    echo "Usage: $0 [user@]host [[user@]host ...]"
    echo "       $0 --from-config"
    echo ""
    echo "Sets up SSH key-based authentication to remote machines."
    echo ""
    echo "Options:"
    echo "  --from-config    Read machines from agents.conf"
    echo "  --help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 user@192.168.1.101 user@192.168.1.102"
    echo "  $0 --from-config"
}

# =============================================================================
# Main
# =============================================================================

echo -e "${BLUE}"
echo "=============================================="
echo "  SSH Key Setup for PEST++ Agents"
echo "=============================================="
echo -e "${NC}"

if [[ $# -eq 0 ]] || [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    print_usage
    exit 0
fi

targets=()

if [[ "$1" == "--from-config" ]]; then
    if [[ ! -f "$CONFIG_FILE" ]]; then
        log_error "Config file not found: ${CONFIG_FILE}"
        exit 1
    fi

    log_info "Reading machines from ${CONFIG_FILE}..."

    while IFS=' ' read -r host user num_agents || [[ -n "$host" ]]; do
        [[ "$host" =~ ^#.*$ ]] && continue
        [[ -z "$host" ]] && continue

        user=${user:-$USER}
        targets+=("${user}@${host}")
    done < "$CONFIG_FILE"
else
    targets=("$@")
fi

if [[ ${#targets[@]} -eq 0 ]]; then
    log_error "No targets specified"
    print_usage
    exit 1
fi

log_info "Will set up SSH keys for ${#targets[@]} machine(s):"
for target in "${targets[@]}"; do
    echo "  - ${target}"
done
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    log_info "Aborted"
    exit 0
fi

# Set up keys for each target
success_count=0
fail_count=0

for target in "${targets[@]}"; do
    if setup_ssh_key "$target"; then
        success_count=$((success_count + 1))
    else
        fail_count=$((fail_count + 1))
    fi
done

# Summary
echo ""
echo -e "${BLUE}=============================================="
echo "  Summary"
echo "==============================================${NC}"
echo ""
log_info "Successful: ${success_count}"
if [[ "$fail_count" -gt 0 ]]; then
    log_error "Failed: ${fail_count}"
fi
echo ""

if [[ "$success_count" -gt 0 ]]; then
    log_success "You can now run deploy-agents.sh to deploy PEST++ agents"
fi
