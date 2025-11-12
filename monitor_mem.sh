#!/usr/bin/env bash
# monitor_mem.sh
# Usage: monitor_mem.sh [interval_seconds] [samples] [max_lines]
# Example: monitor_mem.sh 1 0 30   # sample every 1s indefinitely, show top 30
# Shows absolute memory (RSS) in MiB, sorted by RSS descending.

INTERVAL=${1:-1}
SAMPLES=${2:-0}   # 0 means run forever
MAX_LINES=${3:-20}

# optional: if an extra arg is provided and it's a pid, print pid-tree summary
PID_SUMMARY_PID=""
if [ ! -z "$4" ]; then
  PID_SUMMARY_PID=$4
fi

print_header() {
  printf "%s\n" "Timestamp: $(date -Is)"
  printf "%6s %6s %9s %9s %s\n" PID PPID "RSS(MiB)" "VSZ(MiB)" CMD
}

print_top() {
  # ps outputs RSS and VSZ in KB on most systems; convert to MiB
  # Fields: pid,ppid,rss,vsz,cmd
  ps -eo pid,ppid,rss,vsz,cmd --sort=-rss --no-headers | \
    awk -v max="$MAX_LINES" 'NR<=max{printf "%6s %6s %9.2f %9.2f %s\n", $1, $2, $3/1024, $4/1024, substr($0, index($0,$5))}'
}

sum_pid_tree() {
  local rootpid=$1
  # collect all child pids using ps --no-headers -o pid --ppid recursively
  # fallback: use pstree/pstree -p if available
  if ! kill -0 "$rootpid" 2>/dev/null; then
    echo "PID $rootpid not running"
    return
  fi
  # get all descendant pids (including root) using a loop
  pids="$rootpid"
  idx=1
  while [ $idx -le $(echo "$pids" | wc -w) ]; do
    current=$(echo "$pids" | cut -d' ' -f$idx)
    children=$(ps --no-headers -o pid --ppid $current 2>/dev/null | awk '{print $1}')
    for c in $children; do
      case " $pids " in
        *" $c "*) ;;
        *) pids="$pids $c";;
      esac
    done
    idx=$((idx+1))
  done
  # sum RSS and VSZ
  awk_cmd="BEGIN{rss=0;vsz=0} {rss+=\$1; vsz+=\$2} END{printf \"PID-tree %d procs total RSS=%.2f MiB VSZ=%.2f MiB\n\", NR, rss/1024, vsz/1024}"
  ps -o rss= -o vsz= -p $pids 2>/dev/null | awk "$awk_cmd"
}

# main loop
count=0
while true; do
  print_header
  print_top
  if [ -n "$PID_SUMMARY_PID" ]; then
    echo "--- PID tree summary for $PID_SUMMARY_PID ---"
    sum_pid_tree "$PID_SUMMARY_PID"
  fi
  echo
  if [ "$SAMPLES" -gt 0 ]; then
    count=$((count+1))
    if [ $count -ge $SAMPLES ]; then
      break
    fi
  fi
  sleep "$INTERVAL"
done
