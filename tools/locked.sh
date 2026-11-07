#!/usr/bin/env bash
# Throttle heavy work on this shared Mac.
#   tools/locked.sh val   <cmd...>   one at a time: any run that writes matching/out/<phase>, classification/out or
#                                    integration/out/<phase> (full pipeline, matching/run.py, stage runs on val/live)
#   tools/locked.sh flash <cmd...>   at most 2 Gemini Flash workers at once (wrap tools/agy/flash.sh)
# Waits for a free slot, then runs the command and returns its exit code.
set -uo pipefail
kind="${1:?val|flash}"; shift
case "$kind" in
  val)   exec lockf /tmp/bankrec_val.lock "$@" ;;
  flash) while :; do
           for i in 1 2; do
             lockf -t 0 "/tmp/bankrec_flash_$i.lock" "$@"; rc=$?
             [ $rc -ne 75 ] && exit $rc     # 75 = slot busy (EX_TEMPFAIL); anything else is the command's code
           done
           sleep 15
         done ;;
  *) echo "usage: $0 val|flash cmd..." >&2; exit 2 ;;
esac
