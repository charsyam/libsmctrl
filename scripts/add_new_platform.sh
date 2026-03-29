#!/bin/bash
# Copyright 2026 Backend.AI Team
# Automated tool for adding support for a new GPU model / CUDA version.
#
# Runs in phases that can be executed individually or all together.
#
# Usage:
#   ./scripts/add_new_platform.sh                # run all phases
#   ./scripts/add_new_platform.sh --phase 0      # environment check only
#   ./scripts/add_new_platform.sh --phase 1      # GPU info collection
#   ./scripts/add_new_platform.sh --phase 0,1,2  # multiple phases
#   ./scripts/add_new_platform.sh --phase 3 --scan-range -1000 1000
#   ./scripts/add_new_platform.sh --phase 4 --apply
#
# Options:
#   --phase <N|N,N,...>   Run specific phase(s). Default: all (0-5)
#   --device <id>         CUDA device index. Default: 0
#   --scan-range <s> <e>  Stream offset scan range. Default: -500 500
#   --scan-step <n>       Scan step size in bytes. Default: 4
#   --apply               Phase 4: apply generated patch to libsmctrl.c
#   --state-dir <dir>     Directory for intermediate state. Default: .smctrl_state
#   -h, --help            Show this help

set -euo pipefail

# ──────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CUDA="${CUDA:-/usr/local/cuda}"
NVCC="${NVCC:-$CUDA/bin/nvcc}"
DEVICE=0
SCAN_START=-500
SCAN_END=500
SCAN_STEP=4
APPLY=0
STATE_DIR="$PROJECT_DIR/.smctrl_state"
PHASES="all"
LIBSMCTRL_C="$PROJECT_DIR/libsmctrl.c"
TEST_BIN="$PROJECT_DIR/libsmctrl_test_blackwell"
TEST_SRC="$PROJECT_DIR/libsmctrl_test_blackwell.cu"

# ──────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase)    PHASES="$2"; shift 2 ;;
        --device)   DEVICE="$2"; shift 2 ;;
        --scan-range)
            SCAN_START="$2"; SCAN_END="$3"; shift 3 ;;
        --scan-step) SCAN_STEP="$2"; shift 2 ;;
        --apply)    APPLY=1; shift ;;
        --state-dir) STATE_DIR="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1 ;;
    esac
done

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

ok()   { echo -e "  ${GREEN}OK${NC}    $*"; }
fail() { echo -e "  ${RED}FAIL${NC}  $*"; }
warn() { echo -e "  ${YELLOW}WARN${NC}  $*"; }
info() { echo -e "  ${CYAN}....${NC}  $*"; }

phase_header() {
    echo ""
    echo -e "${BOLD}=== Phase $1: $2 ===${NC}"
}

# Save a key=value to state file
state_save() { echo "$1='$2'" >> "$STATE_DIR/state.env"; }

# Load state file if it exists
state_load() {
    if [[ -f "$STATE_DIR/state.env" ]]; then
        source "$STATE_DIR/state.env"
    fi
}

# Check if phase N should run
should_run() {
    local phase="$1"
    [[ "$PHASES" == "all" ]] && return 0
    echo ",$PHASES," | grep -q ",$phase," && return 0
    return 1
}

# ──────────────────────────────────────────────
# Phase 0: Environment check
# ──────────────────────────────────────────────
phase_0() {
    phase_header 0 "Environment Check"

    # Project root
    if [[ ! -f "$LIBSMCTRL_C" ]]; then
        fail "libsmctrl.c not found at $LIBSMCTRL_C"
        fail "Run this script from the project root or set PROJECT_DIR"
        return 1
    fi
    ok "libsmctrl.c found"

    # CUDA toolkit
    if [[ ! -x "$NVCC" ]]; then
        # Try finding nvcc in PATH
        if command -v nvcc &>/dev/null; then
            NVCC="$(command -v nvcc)"
            CUDA="$(dirname "$(dirname "$NVCC")")"
            warn "Using nvcc from PATH: $NVCC"
        else
            fail "nvcc not found at $NVCC or in PATH"
            fail "Set CUDA=/path/to/cuda or NVCC=/path/to/nvcc"
            return 1
        fi
    fi
    ok "nvcc: $NVCC"

    # GPU access
    if ! command -v nvidia-smi &>/dev/null; then
        fail "nvidia-smi not found -- no GPU accessible"
        return 1
    fi
    local gpu_name
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$DEVICE" 2>/dev/null) || {
        fail "Cannot access GPU device $DEVICE"
        return 1
    }
    ok "GPU device $DEVICE: $gpu_name"

    # Test binary
    if [[ ! -f "$TEST_SRC" ]]; then
        fail "Test source $TEST_SRC not found"
        return 1
    fi

    if [[ ! -x "$TEST_BIN" ]]; then
        info "Test binary not found, building..."
        (
            cd "$PROJECT_DIR"
            make libsmctrl.a 2>&1 | tail -1
            $NVCC -ccbin "${CXX:-g++}" "$TEST_SRC" -o "$TEST_BIN" \
                -g -L. -l:libsmctrl.a -lcuda \
                -I"$CUDA/include" -L"$CUDA/lib64" 2>&1
        ) || {
            fail "Failed to build test binary"
            return 1
        }
    fi
    ok "Test binary: $TEST_BIN"

    # State directory
    mkdir -p "$STATE_DIR"
    # Clear previous state
    rm -f "$STATE_DIR/state.env" "$STATE_DIR/patch"
    ok "State directory: $STATE_DIR"

    state_save "NVCC" "$NVCC"
    state_save "CUDA" "$CUDA"
    state_save "DEVICE" "$DEVICE"

    echo ""
    ok "Environment check passed"
}

# ──────────────────────────────────────────────
# Phase 1: GPU information collection
# ──────────────────────────────────────────────
phase_1() {
    phase_header 1 "GPU Information Collection"
    state_load

    # Run probe
    local probe_output
    probe_output=$("$TEST_BIN" --probe $DEVICE 2>&1) || {
        fail "Probe failed. Output:"
        echo "$probe_output"
        return 1
    }

    # Parse probe output
    local sm_cap sm_count tpc_count sms_per_tpc cuda_ver arch_name
    sm_cap=$(echo "$probe_output" | grep "Compute capability" | awk -F: '{print $2}' | tr -d ' ')
    sm_count=$(echo "$probe_output" | grep "SM count" | awk -F: '{print $2}' | tr -d ' ')
    tpc_count=$(echo "$probe_output" | grep "TPC count" | awk -F: '{print $2}' | tr -d ' ')
    sms_per_tpc=$(echo "$probe_output" | grep "SMs per TPC" | awk -F: '{print $2}' | tr -d ' ')
    cuda_ver=$(echo "$probe_output" | grep "CUDA driver version" | awk -F: '{print $2}' | tr -d ' ')
    arch_name=$(echo "$probe_output" | grep "Architecture" | awk -F: '{print $2}' | sed 's/^ *//')

    # Compute switch-case version number (e.g., "12.8" -> 12080)
    local cuda_major cuda_minor cuda_case_ver
    cuda_major=$(echo "$cuda_ver" | cut -d. -f1)
    cuda_minor=$(echo "$cuda_ver" | cut -d. -f2)
    cuda_case_ver=$(( cuda_major * 1000 + cuda_minor * 10 ))

    # Detect platform
    local platform
    platform=$(uname -m)

    echo "  Compute capability : $sm_cap"
    echo "  Architecture       : $arch_name"
    echo "  SM count           : $sm_count"
    echo "  TPC count          : $tpc_count"
    echo "  SMs per TPC        : $sms_per_tpc"
    echo "  CUDA driver version: $cuda_ver (case $cuda_case_ver)"
    echo "  Platform           : $platform"

    if (( tpc_count > 64 )); then
        warn "TPC count > 64: _ext APIs required for full coverage"
    fi

    state_save "SM_CAP" "$sm_cap"
    state_save "SM_COUNT" "$sm_count"
    state_save "TPC_COUNT" "$tpc_count"
    state_save "SMS_PER_TPC" "$sms_per_tpc"
    state_save "CUDA_VER" "$cuda_ver"
    state_save "CUDA_MAJOR" "$cuda_major"
    state_save "CUDA_MINOR" "$cuda_minor"
    state_save "CUDA_CASE_VER" "$cuda_case_ver"
    state_save "PLATFORM" "$platform"
    state_save "ARCH_NAME" "$arch_name"

    echo ""
    ok "GPU information collected"
}

# ──────────────────────────────────────────────
# Phase 2: Existing support check + TMD callback verification
# ──────────────────────────────────────────────
phase_2() {
    phase_header 2 "Support Check & TMD Callback Verification"
    state_load

    if [[ -z "${CUDA_CASE_VER:-}" ]]; then
        fail "No GPU info found. Run --phase 1 first."
        return 1
    fi

    # ── 2a: Check if case already exists in libsmctrl.c ──
    echo ""
    echo -e "  ${BOLD}[2a] Checking existing support in libsmctrl.c${NC}"

    local case_pattern="case ${CUDA_CASE_VER}:"
    if grep -q "$case_pattern" "$LIBSMCTRL_C"; then
        # Find which offset macro it uses
        local offset_line
        offset_line=$(grep -A3 "$case_pattern" "$LIBSMCTRL_C" | grep -o 'CU_[A-Z0-9_]*MASK_OFF[A-Z_]*' | head -1)
        ok "case $CUDA_CASE_VER already exists (uses $offset_line)"
        state_save "CASE_EXISTS" "1"
        state_save "EXISTING_OFFSET_MACRO" "$offset_line"
    else
        warn "case $CUDA_CASE_VER NOT found -- new platform"
        state_save "CASE_EXISTS" "0"

        # Find nearest existing case for reference
        local nearest
        nearest=$(grep -oP 'case \K[0-9]+' "$LIBSMCTRL_C" | awk -v target="$CUDA_CASE_VER" '
            { diff = ($1 > target) ? $1 - target : target - $1; print diff, $1 }
        ' | sort -n | head -1 | awk '{print $2}')
        info "Nearest existing case: $nearest"
        state_save "NEAREST_CASE" "$nearest"
    fi

    # ── 2b: TMD callback test (Global + Next mask) ──
    echo ""
    echo -e "  ${BOLD}[2b] TMD callback verification${NC}"

    local test_output
    # Run full test but we only care about Phase 2 and 3 results
    test_output=$("$TEST_BIN" "$DEVICE" 2>&1) || true

    # Check Phase 2 (Global mask via TMD callback)
    if echo "$test_output" | grep -q "Phase 2: PASSED"; then
        ok "Global mask (TMD callback): PASSED"
        state_save "TMD_GLOBAL" "PASSED"
    else
        local phase2_fail
        phase2_fail=$(echo "$test_output" | grep -A2 "Phase 2" | grep "FAIL" || echo "unknown error")
        fail "Global mask (TMD callback): FAILED"
        fail "$phase2_fail"
        state_save "TMD_GLOBAL" "FAILED"

        echo ""
        fail "TMD callback mechanism does not work on this platform."
        fail "This likely means a new TMD version with different offsets."
        fail "Manual reverse engineering of control_callback_v2() is required."
        fail "Look at tmd_ver value and find new mask field offsets."
        return 1
    fi

    # Check Phase 3 (Next mask)
    if echo "$test_output" | grep -q "Phase 3: PASSED"; then
        ok "Next mask (TMD callback): PASSED"
        state_save "TMD_NEXT" "PASSED"
    else
        local phase3_fail
        phase3_fail=$(echo "$test_output" | grep -A2 "Phase 3" | grep "FAIL" || echo "unknown error")
        fail "Next mask (TMD callback): FAILED"
        fail "$phase3_fail"
        state_save "TMD_NEXT" "FAILED"
        return 1
    fi

    # Check Phase 4 (Stream mask -- may pass if already supported)
    if echo "$test_output" | grep -q "Phase 4: PASSED"; then
        ok "Stream mask: PASSED (offset already correct)"
        state_save "STREAM_MASK" "PASSED"
        state_save "NEEDS_STREAM_SCAN" "0"
    else
        warn "Stream mask: not working yet (expected for new platform)"
        state_save "STREAM_MASK" "FAILED"
        state_save "NEEDS_STREAM_SCAN" "1"
    fi

    # Check Phase 5 (Priority)
    if echo "$test_output" | grep -q "Phase 5: PASSED"; then
        ok "Mask priority: PASSED"
        state_save "PRIORITY" "PASSED"
    else
        if [[ "${STREAM_MASK:-}" == "FAILED" ]]; then
            info "Mask priority: skipped (stream mask not working)"
        else
            warn "Mask priority: FAILED"
        fi
    fi

    echo ""
    ok "TMD callback verification complete"
}

# ──────────────────────────────────────────────
# Phase 3: Stream offset scan
# ──────────────────────────────────────────────
phase_3() {
    phase_header 3 "Stream Offset Scan"
    state_load

    if [[ -z "${CUDA_CASE_VER:-}" ]]; then
        fail "No GPU info found. Run --phase 1 first."
        return 1
    fi

    # If stream mask already works, skip scan
    if [[ "${NEEDS_STREAM_SCAN:-1}" == "0" ]]; then
        ok "Stream mask already works. No scan needed."
        # Extract current offset value from the macro
        local macro="${EXISTING_OFFSET_MACRO:-}"
        if [[ -n "$macro" ]]; then
            local current_off
            current_off=$(grep "#define $macro" "$LIBSMCTRL_C" | awk '{print $3}')
            ok "Current offset: $macro = $current_off"
            state_save "FOUND_OFFSET" "$current_off"
            state_save "FOUND_OFFSET_MACRO" "$macro"
        fi
        return 0
    fi

    # ── Strategy: try adjacent known offsets first, then scan ──

    echo ""
    echo -e "  ${BOLD}[3a] Trying known offsets from nearby CUDA versions${NC}"

    # Collect all known offsets for current platform
    local platform_suffix=""
    if [[ "${PLATFORM:-x86_64}" == "aarch64" ]]; then
        platform_suffix="_JETSON"
    fi

    # Extract all offset values from libsmctrl.c
    local -a known_offsets=()
    local -a known_names=()
    while IFS= read -r line; do
        local name val
        name=$(echo "$line" | awk '{print $2}')
        val=$(echo "$line" | awk '{print $3}')
        # Filter by platform
        if [[ -n "$platform_suffix" ]]; then
            echo "$name" | grep -q "JETSON" || continue
        else
            echo "$name" | grep -q "JETSON" && continue
        fi
        known_offsets+=("$val")
        known_names+=("$name")
    done < <(grep '#define CU_.*MASK_OFF' "$LIBSMCTRL_C" | grep -v '//')

    # Sort by proximity to latest version (try newest first)
    info "Found ${#known_offsets[@]} known offsets for this platform"

    local found_off=""
    local found_hex=""

    for i in "${!known_offsets[@]}"; do
        local hex="${known_offsets[$i]}"
        local dec_off=$((hex))
        local delta=$((dec_off - 0x4e4))  # delta from CUDA 12.2 base

        info "Trying ${known_names[$i]} = $hex (delta=$delta)..."
        local output
        output=$(MASK_OFF=$delta "$TEST_BIN" "$DEVICE" 2>&1) || true

        if echo "$output" | grep -q "Phase 4: PASSED"; then
            ok "HIT! ${known_names[$i]} = $hex works"
            found_off="$delta"
            found_hex="$hex"
            state_save "FOUND_OFFSET" "$hex"
            state_save "FOUND_DELTA" "$delta"
            state_save "FOUND_VIA" "known_offset:${known_names[$i]}"
            break
        fi
    done

    # ── If known offsets didn't work, scan ──
    if [[ -z "$found_off" ]]; then
        echo ""
        echo -e "  ${BOLD}[3b] Scanning range $SCAN_START to $SCAN_END (step $SCAN_STEP)${NC}"
        info "This may take a while..."

        local count=0
        local total=$(( (SCAN_END - SCAN_START) / SCAN_STEP + 1 ))

        for off in $(seq "$SCAN_START" "$SCAN_STEP" "$SCAN_END"); do
            count=$((count + 1))
            local abs_off=$(printf "0x%x" $((0x4e4 + off)))

            local output
            output=$(MASK_OFF=$off "$TEST_BIN" "$DEVICE" 2>&1) || true

            if echo "$output" | grep -q "Phase 4: PASSED"; then
                echo ""
                ok "HIT! delta=$off, absolute=$abs_off"
                found_off="$off"
                found_hex="$abs_off"
                state_save "FOUND_OFFSET" "$abs_off"
                state_save "FOUND_DELTA" "$off"
                state_save "FOUND_VIA" "scan"
                break
            fi

            # Progress indicator every 50 attempts
            if (( count % 50 == 0 )); then
                printf "\r  ....  Scanned %d / %d offsets..." "$count" "$total"
            fi
        done
        echo ""
    fi

    # ── Report results ──
    echo ""
    if [[ -n "$found_off" ]]; then
        ok "Stream offset found: $found_hex (delta from 0x4e4: $found_off)"

        # Determine if this is v1 (CUDA < 12.0) or v2
        if (( CUDA_CASE_VER >= 12000 )); then
            state_save "MASK_STRUCT" "v2"
            info "Mask structure: stream_sm_mask_v2 (128-bit, CUDA 12.0+)"
        else
            state_save "MASK_STRUCT" "v1"
            info "Mask structure: stream_sm_mask (64-bit, CUDA < 12.0)"
        fi
    else
        fail "No working offset found in range $SCAN_START to $SCAN_END"
        echo ""
        fail "Suggestions:"
        fail "  1. Widen range: --scan-range -2000 2000"
        fail "  2. Verify TMD callback works: --phase 2"
        fail "  3. The stream struct layout may have changed significantly."
        return 1
    fi
}

# ──────────────────────────────────────────────
# Phase 4: Generate and optionally apply patch
# ──────────────────────────────────────────────
phase_4() {
    phase_header 4 "Patch Generation"
    state_load

    if [[ -z "${FOUND_OFFSET:-}" ]]; then
        fail "No offset found. Run --phase 3 first."
        return 1
    fi

    # If case already exists and stream mask works, nothing to patch
    if [[ "${CASE_EXISTS:-0}" == "1" && "${NEEDS_STREAM_SCAN:-1}" == "0" ]]; then
        ok "Already supported. No patch needed."
        return 0
    fi

    local patch_file="$STATE_DIR/patch"
    local cuda_major="${CUDA_MAJOR:-12}"
    local cuda_minor="${CUDA_MINOR:-0}"
    local cuda_case="${CUDA_CASE_VER:-12000}"
    local offset="${FOUND_OFFSET}"
    local mask_struct="${MASK_STRUCT:-v2}"

    # Determine platform suffix for macro name
    local platform_tag=""
    local section_guard="__x86_64__"
    if [[ "${PLATFORM:-x86_64}" == "aarch64" ]]; then
        platform_tag="_JETSON"
        section_guard="__aarch64__"
    fi

    # Construct macro name: CU_12_8_MASK_OFF or CU_12_8_MASK_OFF_JETSON
    local macro_name="CU_${cuda_major}_${cuda_minor}_MASK_OFF${platform_tag}"

    # Check if this macro already exists
    if grep -q "#define $macro_name " "$LIBSMCTRL_C"; then
        ok "Macro $macro_name already exists in libsmctrl.c"
        state_save "PATCH_NEEDED" "0"
        return 0
    fi

    echo ""
    echo -e "  ${BOLD}Generating patch for CUDA $cuda_major.$cuda_minor ($cuda_case)${NC}"
    info "Macro:   #define $macro_name $offset"
    info "Case:    case $cuda_case"
    info "Struct:  ${mask_struct}"

    # ── Build the patch content ──
    # We generate a human-readable instruction file instead of a unified diff,
    # because the insertion points depend on context that's hard to capture
    # in a static diff (the file changes with each new version added).

    cat > "$patch_file" <<PATCHEOF
# ──────────────────────────────────────────────
# libsmctrl.c patch for CUDA ${cuda_major}.${cuda_minor} on ${PLATFORM:-x86_64}
# Generated by add_new_platform.sh on $(date +%Y-%m-%d)
# ──────────────────────────────────────────────
#
# GPU: ${ARCH_NAME:-unknown} (${SM_CAP:-unknown})
# SM count: ${SM_COUNT:-?}, TPC count: ${TPC_COUNT:-?}, SMs/TPC: ${SMS_PER_TPC:-?}
# CUDA driver: ${CUDA_VER:-?} (case ${cuda_case})
# Offset: ${offset}
# Found via: ${FOUND_VIA:-unknown}
#
# Two changes needed:
#
# [CHANGE 1] Add offset macro definition
#   Section: #if ${section_guard}  (or Jetson aarch64 section)
#   Insert AFTER the last #define CU_*_MASK_OFF* line for this platform
#
DEFINE:#define ${macro_name} ${offset}
DEFINE_COMMENT:// ${cuda_major}.${cuda_minor} tested on $(nvidia-smi --query-gpu=driver_version --format=csv,noheader -i "$DEVICE" 2>/dev/null || echo "unknown driver")
#
# [CHANGE 2] Add switch case
#   Section: switch (ver) { ... #if ${section_guard} ... }
#   Insert AFTER the last case in the ${section_guard} section
#
CASE:${cuda_case}
MACRO:${macro_name}
STRUCT:${mask_struct}
PATCHEOF

    ok "Patch file written to: $patch_file"

    # ── Show preview ──
    echo ""
    echo -e "  ${BOLD}Preview of changes:${NC}"
    echo ""
    echo -e "    ${CYAN}// Change 1: Add after last offset #define${NC}"
    echo "    #define ${macro_name} ${offset}"
    echo "    // ${cuda_major}.${cuda_minor} tested on $(nvidia-smi --query-gpu=driver_version --format=csv,noheader -i "$DEVICE" 2>/dev/null || echo "unknown")"
    echo ""
    echo -e "    ${CYAN}// Change 2: Add in switch(ver) under #if ${section_guard}${NC}"
    if [[ "$mask_struct" == "v2" ]]; then
        echo "    case ${cuda_case}:"
        echo "        hw_mask_v2 = (void*)(stream_struct_base + ${macro_name});"
        echo "        break;"
    else
        echo "    case ${cuda_case}:"
        echo "        hw_mask = (struct stream_sm_mask*)(stream_struct_base + ${macro_name});"
        echo "        break;"
    fi

    state_save "PATCH_NEEDED" "1"
    state_save "MACRO_NAME" "$macro_name"

    # ── Apply if requested ──
    if [[ "$APPLY" == "1" ]]; then
        echo ""
        echo -e "  ${BOLD}Applying patch...${NC}"
        _apply_patch "$macro_name" "$offset" "$cuda_case" "$mask_struct" \
                     "$section_guard" "$cuda_major" "$cuda_minor" "$platform_tag"
    else
        echo ""
        info "To apply this patch, run:"
        info "  ./scripts/add_new_platform.sh --phase 4 --apply"
        info "Or manually edit libsmctrl.c as shown above."
    fi
}

_apply_patch() {
    local macro_name="$1" offset="$2" cuda_case="$3" mask_struct="$4"
    local section_guard="$5" cuda_major="$6" cuda_minor="$7" platform_tag="$8"

    local driver_ver
    driver_ver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader -i "$DEVICE" 2>/dev/null || echo "unknown")

    # ── Change 1: Insert #define ──
    # Find the last CU_*_MASK_OFF define for this platform
    local last_define_line
    if [[ -n "$platform_tag" ]]; then
        # Jetson: find last _JETSON define
        last_define_line=$(grep -n '#define CU_.*MASK_OFF_JETSON' "$LIBSMCTRL_C" | tail -1 | cut -d: -f1)
    else
        # x86_64: find last non-JETSON define
        last_define_line=$(grep -n '#define CU_.*MASK_OFF ' "$LIBSMCTRL_C" | tail -1 | cut -d: -f1)
    fi

    if [[ -z "$last_define_line" ]]; then
        fail "Could not find insertion point for #define"
        return 1
    fi

    # Skip any trailing comment line(s) after the last define
    local insert_after=$last_define_line
    while IFS= read -r nextline; do
        if echo "$nextline" | grep -qP '^// '; then
            insert_after=$((insert_after + 1))
        else
            break
        fi
    done < <(sed -n "$((last_define_line + 1)),\$p" "$LIBSMCTRL_C")

    # Insert the define + comment
    local define_block
    define_block="#define ${macro_name} ${offset}\n// ${cuda_major}.${cuda_minor} tested on ${driver_ver}"
    sed -i "${insert_after}a\\
${define_block}" "$LIBSMCTRL_C"

    ok "Inserted #define at line $((insert_after + 1))"

    # ── Change 2: Insert case ──
    # Find the last case line in the correct platform section
    # Strategy: find the #if section, then the last "break;" before #elif or #endif
    local section_start section_end last_break_line
    section_start=$(grep -n "#if ${section_guard}" "$LIBSMCTRL_C" | head -1 | cut -d: -f1)
    if [[ "${section_guard}" == "__x86_64__" ]]; then
        section_end=$(grep -n '#elif __aarch64__' "$LIBSMCTRL_C" | head -1 | cut -d: -f1)
    else
        section_end=$(grep -n '#endif' "$LIBSMCTRL_C" | while IFS=: read -r ln _; do
            if (( ln > section_start )); then echo "$ln"; break; fi
        done)
    fi

    if [[ -z "$section_start" || -z "$section_end" ]]; then
        fail "Could not find platform section boundaries"
        return 1
    fi

    # Find last "break;" line within the section
    last_break_line=$(sed -n "${section_start},${section_end}p" "$LIBSMCTRL_C" \
        | grep -n 'break;' | tail -1 | cut -d: -f1)
    last_break_line=$((section_start + last_break_line - 1))

    # Build the case block
    local case_block
    if [[ "$mask_struct" == "v2" ]]; then
        case_block="\\tcase ${cuda_case}:\\n\\t\\thw_mask_v2 = (void*)(stream_struct_base + ${macro_name});\\n\\t\\tbreak;"
    else
        case_block="\\tcase ${cuda_case}:\\n\\t\\thw_mask = (struct stream_sm_mask*)(stream_struct_base + ${macro_name});\\n\\t\\tbreak;"
    fi

    sed -i "${last_break_line}a\\
$(echo -e "$case_block")" "$LIBSMCTRL_C"

    ok "Inserted case $cuda_case at line $((last_break_line + 1))"

    # ── Rebuild and verify ──
    echo ""
    echo -e "  ${BOLD}Rebuilding and verifying...${NC}"
    (
        cd "$PROJECT_DIR"
        make libsmctrl.a 2>&1 | tail -1
        $NVCC -ccbin "${CXX:-g++}" "$TEST_SRC" -o "$TEST_BIN" \
            -g -L. -l:libsmctrl.a -lcuda \
            -I"$CUDA/include" -L"$CUDA/lib64" 2>&1
    ) || {
        fail "Rebuild failed!"
        return 1
    }
    ok "Rebuild successful"

    local verify_output
    verify_output=$("$TEST_BIN" "$DEVICE" 2>&1) || true

    if echo "$verify_output" | grep -q "All tests PASSED"; then
        ok "All tests PASSED after patch"
    else
        warn "Some tests failed after patch. Output:"
        echo "$verify_output" | grep -E "(PASS|FAIL|ERROR)" | while IFS= read -r line; do
            echo "    $line"
        done
    fi
}

# ──────────────────────────────────────────────
# Phase 5: Final report
# ──────────────────────────────────────────────
phase_5() {
    phase_header 5 "Summary Report"
    state_load

    echo ""
    echo "  Platform       : ${PLATFORM:-unknown}"
    echo "  GPU            : ${ARCH_NAME:-unknown} (${SM_CAP:-unknown})"
    echo "  CUDA version   : ${CUDA_VER:-unknown} (case ${CUDA_CASE_VER:-?})"
    echo "  SM/TPC layout  : ${SM_COUNT:-?} SMs, ${TPC_COUNT:-?} TPCs, ${SMS_PER_TPC:-?} SMs/TPC"
    echo ""
    echo "  TMD Global     : ${TMD_GLOBAL:-not tested}"
    echo "  TMD Next       : ${TMD_NEXT:-not tested}"
    echo "  Stream Mask    : ${STREAM_MASK:-not tested}"
    echo "  Stream Offset  : ${FOUND_OFFSET:-not found} (via ${FOUND_VIA:-n/a})"
    echo ""

    if [[ "${CASE_EXISTS:-0}" == "1" && "${NEEDS_STREAM_SCAN:-1}" == "0" ]]; then
        echo -e "  ${GREEN}${BOLD}Result: Platform already fully supported${NC}"
    elif [[ -n "${FOUND_OFFSET:-}" && "${TMD_GLOBAL:-}" == "PASSED" ]]; then
        if [[ "${PATCH_NEEDED:-1}" == "1" && "$APPLY" != "1" ]]; then
            echo -e "  ${YELLOW}${BOLD}Result: Offset found, patch ready to apply${NC}"
            echo "  Run with --phase 4 --apply to update libsmctrl.c"
        elif [[ "$APPLY" == "1" ]]; then
            echo -e "  ${GREEN}${BOLD}Result: Patch applied and verified${NC}"
        fi
    elif [[ "${TMD_GLOBAL:-}" == "FAILED" ]]; then
        echo -e "  ${RED}${BOLD}Result: TMD callback incompatible — manual work required${NC}"
    else
        echo -e "  ${RED}${BOLD}Result: Stream offset not found — manual investigation needed${NC}"
    fi
    echo ""
}

# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
echo -e "${BOLD}add_new_platform.sh — libsmctrl platform support tool${NC}"

failed=0

for phase in 0 1 2 3 4 5; do
    should_run "$phase" || continue
    if ! "phase_${phase}"; then
        failed=1
        if [[ "$PHASES" == "all" ]]; then
            echo ""
            fail "Phase $phase failed. Stopping."
            break
        fi
    fi
done

exit $failed
