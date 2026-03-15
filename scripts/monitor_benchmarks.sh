#!/bin/bash
# Monitor both benchmark tiers, auto-restart on failure.
# Usage: bash scripts/monitor_benchmarks.sh
# Log: results/benchmark/monitor.log

set -uo pipefail

LOGFILE="results/benchmark/monitor.log"
HTCF_HOST="shandley@login.htcf.wustl.edu"
HTCF_DIR="/scratch/sahlab/shandley/virosense/ViroSense"
CHECK_INTERVAL=3600  # 1 hour

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') | $*" | tee -a "$LOGFILE"; }

check_40b() {
    local last_line
    last_line=$(tail -1 results/benchmark/40b/run.log 2>/dev/null)

    # Check if process is running
    if ! pgrep -f "benchmark_unified.*40b" >/dev/null 2>&1; then
        # Check if it completed successfully
        if grep -q "Benchmark complete" results/benchmark/40b/run.log 2>/dev/null; then
            log "40B: COMPLETE"
            return 0
        fi
        log "40B: CRASHED — restarting"
        set -a && source .env && set +a
        nohup uv run python scripts/benchmark_unified.py run \
            --manifest results/benchmark/manifest/ \
            --classifier results/classifiers/40b/classifier.joblib \
            --backend nim --model evo2_40b \
            --layer blocks.28.mlp.l3 \
            --cache-dir results/benchmark/cache_40b/ \
            --output results/benchmark/40b/ \
            >> results/benchmark/40b/run.log 2>&1 &
        log "40B: Restarted (PID $!)"
        return 1
    fi

    local cached
    cached=$(grep "Checkpoint saved" results/benchmark/40b/run.log | tail -1 | grep -o '[0-9]*/13417' || echo "?")
    log "40B: Running — $cached cached"
    return 0
}

check_7b() {
    # Check if benchmark completed
    local last_checkpoint
    last_checkpoint=$(ssh "$HTCF_HOST" "grep 'Benchmark complete' $HTCF_DIR/results/benchmark/7b/run.log 2>/dev/null" 2>/dev/null)
    if [ -n "$last_checkpoint" ]; then
        log "7B: COMPLETE"
        return 0
    fi

    # Check if screen session exists
    local screen_check
    screen_check=$(ssh "$HTCF_HOST" "screen -ls 2>/dev/null | grep benchmark" 2>/dev/null)

    if [ -z "$screen_check" ]; then
        log "7B: No screen session — checking NIM server"

        # Check NIM server
        local nim_status
        nim_status=$(ssh "$HTCF_HOST" "curl -s -o /dev/null -w '%{http_code}' http://n099:8000/v1/health/ready 2>/dev/null" 2>/dev/null)

        if [ "$nim_status" != "200" ]; then
            # Check if NIM job exists
            local job_check
            job_check=$(ssh "$HTCF_HOST" "squeue -u shandley -n nim-evo2 -h 2>/dev/null" 2>/dev/null)

            if [ -z "$job_check" ]; then
                log "7B: NIM server down — submitting new job"
                ssh "$HTCF_HOST" "cd $HTCF_DIR && sbatch htcf/start_nim_server.sbatch" 2>/dev/null
                log "7B: NIM job submitted — will launch benchmark on next check"
                return 1
            else
                log "7B: NIM job queued/starting — waiting"
                return 1
            fi
        fi

        # NIM is ready, launch benchmark
        log "7B: NIM ready — launching benchmark"
        ssh "$HTCF_HOST" bash -l << 'REMOTE_EOF'
cd /scratch/sahlab/shandley/virosense/ViroSense
mkdir -p results/benchmark/7b
screen -dmS benchmark bash -c "
export PATH=\$HOME/.local/bin:\$PATH
cd /scratch/sahlab/shandley/virosense/ViroSense
uv run python scripts/benchmark_unified.py run \
    --manifest results/benchmark/manifest/ \
    --classifier /scratch/sahlab/shandley/virosense/model_7b/classifier.joblib \
    --backend nim --model evo2_7b --nim-url http://n099:8000 \
    --layer blocks.28.mlp.l3 \
    --cache-dir results/benchmark/cache_7b/ \
    --output results/benchmark/7b/ \
    2>&1 | tee -a results/benchmark/7b/run.log
"
REMOTE_EOF
        log "7B: Benchmark launched in screen"
        return 1
    fi

    # Screen exists — check progress
    local cached
    cached=$(ssh "$HTCF_HOST" "grep 'Checkpoint saved' $HTCF_DIR/results/benchmark/7b/run.log 2>/dev/null | tail -1 | grep -o '[0-9]*/13417'" 2>/dev/null || echo "?")
    log "7B: Running — $cached cached"
    return 0
}

# Main loop
log "========================================="
log "Benchmark monitor started (checking every ${CHECK_INTERVAL}s)"
log "========================================="

while true; do
    log "--- Check ---"
    check_40b
    r40=$?
    check_7b
    r7=$?

    # Exit if both complete
    if grep -q "Benchmark complete" results/benchmark/40b/run.log 2>/dev/null && \
       ssh "$HTCF_HOST" "grep -q 'Benchmark complete' $HTCF_DIR/results/benchmark/7b/run.log 2>/dev/null" 2>/dev/null; then
        log "BOTH BENCHMARKS COMPLETE!"
        break
    fi

    sleep "$CHECK_INTERVAL"
done
