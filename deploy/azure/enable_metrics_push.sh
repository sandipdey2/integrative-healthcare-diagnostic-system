#!/usr/bin/env bash
# Turn the in-container metrics push of ihds-app ON (or OFF), from Azure Cloud Shell (Bash), repo root.
# Option A (Grafana Cloud) uses it directly; deploy_monitoring_aca.sh (option B) calls it with an internal URL.
#
# How it works: the app image contains a Prometheus binary. When GRAFANA_RW_URL is set, scripts/start_all.sh
# starts it in agent mode (scripts/metrics_agent.py): it scrapes 127.0.0.1:8000/metrics inside the container
# and remote_writes the ihds_* series OUT. Port 8000 stays internal; nothing new is exposed. Unset = off.
#
# The three values are stored as Container Apps secrets and mapped to env vars with secretref:
#   secret grafana-rw-url   -> GRAFANA_RW_URL    (Grafana Cloud "Remote Write Endpoint", .../api/prom/push)
#   secret grafana-rw-user  -> GRAFANA_RW_USER   (numeric Prometheus instance ID; optional for option B)
#   secret grafana-rw-token -> GRAFANA_RW_TOKEN  (access-policy token with metrics:write, glc_...)
#
# Option A, Grafana Cloud:
#   export RG=Regroup_9vxmyzpemb ACR=<your-acr> TAG=v3
#   export GRAFANA_RW_URL='https://prometheus-prod-XX-prod-<region>.grafana.net/api/prom/push'
#   export GRAFANA_RW_USER='<instance ID>'
#   read -r -s -p "Grafana Cloud token: " GRAFANA_RW_TOKEN && export GRAFANA_RW_TOKEN && echo
#   bash deploy/azure/enable_metrics_push.sh
# (If GRAFANA_RW_TOKEN is not exported the script prompts for it with hidden input.)
#
# BUILD=1 (default) rebuilds ihds:$TAG in ACR first (the image must contain the agent; use a NEW tag so
# Container Apps starts a new revision). BUILD=0 reuses the current image (e.g. only rotating the token;
# the script then restarts the active revision so the new secret value is read).
# Turn OFF:  RG=... DISABLE=1 bash deploy/azure/enable_metrics_push.sh
set -euo pipefail

: "${RG:?set RG to the resource group of ihds-app}"
APP="${APP:-ihds-app}"
CONTAINER="${CONTAINER:-$APP}"
az extension add --name containerapp --upgrade --only-show-errors >/dev/null || true
az provider register --namespace Microsoft.App --only-show-errors >/dev/null || true

restart_active() {
  local rev
  rev=$(az containerapp revision list -n "$APP" -g "$RG" --query "[?properties.active] | [-1].name" -o tsv)
  az containerapp revision restart -n "$APP" -g "$RG" --revision "$rev" --only-show-errors -o none
  echo "   restarted revision $rev"
}

if [[ "${DISABLE:-0}" == "1" ]]; then
  echo ">> Disabling metrics push on $APP"
  az containerapp update -n "$APP" -g "$RG" --container-name "$CONTAINER" \
    --remove-env-vars GRAFANA_RW_URL GRAFANA_RW_USER GRAFANA_RW_TOKEN --only-show-errors -o none
  az containerapp secret remove -n "$APP" -g "$RG" --secret-names grafana-rw-url grafana-rw-user grafana-rw-token \
    --only-show-errors -o none 2>/dev/null || true
  echo ">> Done: new revision without the agent."
  exit 0
fi

: "${GRAFANA_RW_URL:?set GRAFANA_RW_URL (Grafana Cloud remote write endpoint)}"
case "$GRAFANA_RW_URL" in
  https://*) ;;
  http://*)  # plain HTTP only for an in-environment target without credentials (option B)
    if [[ -n "${GRAFANA_RW_USER:-}" ]]; then echo "refusing to send credentials over http://"; exit 1; fi ;;
  *) echo "GRAFANA_RW_URL must start with https://"; exit 1 ;;
esac
DEPLOYMENT_LABEL="${DEPLOYMENT_LABEL:-azure-container-apps}"

SECRETS=("grafana-rw-url=$GRAFANA_RW_URL")
ENV_ARGS=("GRAFANA_RW_URL=secretref:grafana-rw-url" "IHDS_METRICS_DEPLOYMENT=$DEPLOYMENT_LABEL")
if [[ -n "${GRAFANA_RW_USER:-}" ]]; then
  if [[ -z "${GRAFANA_RW_TOKEN:-}" ]]; then
    read -r -s -p "Grafana Cloud access-policy token (glc_..., input hidden): " GRAFANA_RW_TOKEN; echo
  fi
  [[ -n "$GRAFANA_RW_TOKEN" ]] || { echo "empty token"; exit 1; }
  SECRETS+=("grafana-rw-user=$GRAFANA_RW_USER" "grafana-rw-token=$GRAFANA_RW_TOKEN")
  ENV_ARGS+=("GRAFANA_RW_USER=secretref:grafana-rw-user" "GRAFANA_RW_TOKEN=secretref:grafana-rw-token")
fi

echo ">> 1/3 Storing Container Apps secrets (values not shown)"
az containerapp secret set -n "$APP" -g "$RG" --secrets "${SECRETS[@]}" --only-show-errors -o none
unset GRAFANA_RW_TOKEN SECRETS

IMAGE_ARGS=()
if [[ "${BUILD:-1}" == "1" ]]; then
  : "${ACR:?set ACR to the registry that holds the ihds image}"
  : "${TAG:?set TAG to a NEW image tag, e.g. v3}"
  echo ">> 2/3 Building ihds:$TAG in $ACR (includes the Prometheus agent binary)"
  az acr build -r "$ACR" -t "ihds:$TAG" . --only-show-errors -o none
  IMAGE_ARGS=(--image "$ACR.azurecr.io/ihds:$TAG")
else
  echo ">> 2/3 BUILD=0: keeping the current image"
fi

echo ">> 3/3 Updating $APP"
if [[ -z "${GRAFANA_RW_USER:-}" ]]; then
  az containerapp update -n "$APP" -g "$RG" --container-name "$CONTAINER" \
    --remove-env-vars GRAFANA_RW_USER GRAFANA_RW_TOKEN --only-show-errors -o none 2>/dev/null || true
fi
az containerapp update -n "$APP" -g "$RG" --container-name "$CONTAINER" "${IMAGE_ARGS[@]}" \
  --set-env-vars "${ENV_ARGS[@]}" --only-show-errors -o none
if [[ "${BUILD:-1}" != "1" ]]; then restart_active; fi   # secrets are read at container start

REV=$(az containerapp revision list -n "$APP" -g "$RG" --query "[?properties.active] | [-1].name" -o tsv)
echo ">> Done. Active revision: $REV"
echo "   Check:  az containerapp logs show -n $APP -g $RG --tail 80 | grep -i 'metrics agent'"
