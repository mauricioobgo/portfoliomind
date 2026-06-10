#!/usr/bin/env bash
# Register the PortfolioMind scheduler under the portfoliomind Hermes profile.
#
# This script is the operator-facing entry point for the card-8 cron
# registration step. It sources the canonical morning-cron expression
# from DEFAULT_MORNING_CRON (so the in-process scheduler and the
# hermes-level cron wrapper always agree), builds the right
# `hermes cron create` command, and (with --apply) actually
# registers the job.
#
# Usage:
#   bash scripts/register_cron.sh                    # dry-run, print command
#   bash scripts/register_cron.sh --morning-cron "..."  # override morning cron
#   bash scripts/register_cron.sh --apply           # actually register
#   bash scripts/register_cron.sh --workdir /path   # use a different workdir
#   bash scripts/register_cron.sh --name myname     # use a different job name
#
# The script never embeds secrets; the portfoliomind profile env
# supplies GOOGLE_SERVICE_ACCOUNT_JSON, INVESTINGPRO_*, XTB_* at
# runtime, not here.

set -euo pipefail

# --- Argument parsing ------------------------------------------------------

APPLY=0
MORNING_CRON=""
WORKDIR=""
NAME="portfoliomind-scheduler"
PROFILE="portfoliomind"
PYTHON_BIN="uv run python"

usage() {
  cat <<EOF
Usage: bash scripts/register_cron.sh [--apply] [--morning-cron CRON] [--workdir DIR] [--name NAME]

Options:
  --apply              Actually run \`hermes cron create\`. Default is dry-run.
  --morning-cron CRON  Override the morning trigger cron expression
                       (default: DEFAULT_MORNING_CRON from the codebase)
  --workdir DIR        Set the workdir for the cron job (default: $(pwd))
  --name NAME          Set the cron job name (default: portfoliomind-scheduler)
  --profile NAME       Set the Hermes profile (default: portfoliomind)
  --help               Show this help text

Examples:
  # Print the command without running it.
  bash scripts/register_cron.sh

  # Register the cron job under the portfoliomind profile.
  bash scripts/register_cron.sh --apply

  # Use a different morning cron (e.g. move to 14:00 UTC for US summer DST).
  bash scripts/register_cron.sh --morning-cron "0 14 * * 1-5" --apply
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply) APPLY=1; shift ;;
    --morning-cron) MORNING_CRON="$2"; shift 2 ;;
    --workdir) WORKDIR="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

# --- Resolve defaults from the codebase -----------------------------------

# Pull DEFAULT_MORNING_CRON from the codebase so the registration
# command and the in-process scheduler always agree. This is a
# single source of truth — don't hardcode the cron string in
# shell.
DEFAULT_MORNING_CRON="$(
  cd "$(dirname "$0")/.." && \
  ${PYTHON_BIN} -c "from portfoliomind.scheduler.loop import DEFAULT_MORNING_CRON; print(DEFAULT_MORNING_CRON)"
)"

if [[ -z "$MORNING_CRON" ]]; then
  MORNING_CRON="$DEFAULT_MORNING_CRON"
fi

if [[ -z "$WORKDIR" ]]; then
  WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"
fi

# --- Sanity checks ---------------------------------------------------------

if ! command -v hermes >/dev/null 2>&1; then
  echo "ERROR: \`hermes\` CLI not found in PATH. Install Hermes Agent first." >&2
  exit 3
fi

if [[ ! -d "$WORKDIR/src/portfoliomind" ]]; then
  echo "WARNING: $WORKDIR does not look like the portfoliomind repo (no src/portfoliomind). Continuing anyway." >&2
fi

# --- Build the command -----------------------------------------------------

# The --no-agent flag is critical: the cron wrapper should not spin
# up an LLM loop. The script is deterministic — its exit code is
# the truth. The --workdir is set so the script's relative paths
# (uv.lock, .env, src/) resolve correctly.
# The morning cron expression is the re-launch schedule (in UTC,
# container time). The actual morning job inside the daemon fires
# at 08:30 Bogota = 13:30 UTC Mon-Fri via APScheduler's
# America/Bogota timezone. The 0 3 * * * re-launch covers any
# unexpected daemon exit; the daemon itself runs forever under
# that parent.
COMMAND=(
  hermes cron create
  --name "$NAME"
  --schedule "$MORNING_CRON"
  --workdir "$WORKDIR"
  --profile "$PROFILE"
  --no-agent
  --script run_scheduler.py --daemon
)

# --- Run or print ----------------------------------------------------------

if [[ "$APPLY" -eq 1 ]]; then
  echo "Registering cron job..." >&2
  "${COMMAND[@]}"
  echo
  echo "Done. The job is now visible in \`hermes cron list\`." >&2
  echo "To disable: \`hermes cron pause $NAME\`" >&2
  echo "To remove:  \`hermes cron remove $NAME\`" >&2
else
  echo "DRY-RUN. The following command would be run (pass --apply to actually run it):" >&2
  printf '  %q ' "${COMMAND[@]}"
  echo
  echo
  echo "Notes:" >&2
  echo "  * morning cron:  $MORNING_CRON (UTC, default from DEFAULT_MORNING_CRON)" >&2
  echo "  * workdir:       $WORKDIR" >&2
  echo "  * profile:       $PROFILE" >&2
  echo "  * job name:      $NAME" >&2
  echo
  echo "The morning cron in this command is the *re-launch* schedule" >&2
  echo "(the daily check that the daemon is still running). The actual" >&2
  echo "morning job inside the daemon fires at 08:30 Bogota = 13:30 UTC" >&2
  echo "Mon-Fri via APScheduler's America/Bogota timezone." >&2
fi
