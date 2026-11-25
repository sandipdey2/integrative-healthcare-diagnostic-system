#!/usr/bin/env bash
# Option B (lower priority; option A, Grafana Cloud, is the chosen path): self-hosted Prometheus + Grafana as two extra Azure Container Apps next to ihds-app.
# Run from the repo root in Azure Cloud Shell (Bash), after deploy_aca.sh has created ihds-app.
#
#   export RG=Regroup_9vxmyzpemb ACR=<your-acr>       # ENVNAME defaults to ihds-env
#   bash deploy/azure/deploy_monitoring_aca.sh         # prompts for a Grafana admin password (hidden)
#
# Design (simplest reliable option on Container Apps):
#   ihds-app (8501 public, 8000 internal to the container)
#     └─ Prometheus agent inside the container scrapes 127.0.0.1:8000/metrics and remote_writes to
#   ihds-prometheus  (internal ingress :9090 -> http://ihds-prometheus, reachable only inside ihds-env,
#                     --web.enable-remote-write-receiver, 15 d retention on the replica's disk)
#   ihds-grafana     (external ingress :3000, anonymous read-only Viewer, admin password = secret,
#                     datasource http://ihds-prometheus, IHDS dashboard provisioned)
# Prometheus does not scrape ihds-app directly because a Container App exposes one HTTP ingress port (8501 here).
#
# Optional: GRAFANA_ADMIN_PASSWORD (else prompted), PROM_APP, GRAFANA_APP, MON_TAG (default v1), TAG (new app image tag, v3),
#           ANONYMOUS=0 to require login for viewers.
# Lab subscriptions: provider registration may be denied (ignored); `az acr build` may be blocked by policy
# (TasksOperationsNotAllowed) - see deploy/azure/README.md. Two extra apps at min 1 replica use lab credit.
# Remove:  az containerapp delete -n ihds-grafana -g $RG -y; az containerapp delete -n ihds-prometheus -g $RG -y
#          DISABLE=1 bash deploy/azure/enable_metrics_push.sh
set -euo pipefail

: "${RG:?set RG to the existing resource group}"
: "${ACR:?set ACR to the registry used by deploy_aca.sh}"
APP="${APP:-ihds-app}"
ENVNAME="${ENVNAME:-ihds-env}"
PROM_APP="${PROM_APP:-ihds-prometheus}"
GRAFANA_APP="${GRAFANA_APP:-ihds-grafana}"
MON_TAG="${MON_TAG:-v1}"
ANON="true"; [[ "${ANONYMOUS:-1}" == "0" ]] && ANON="false"

echo ">> Using resource group $RG, environment $ENVNAME"
az extension add --name containerapp --upgrade --only-show-errors >/dev/null || true
az provider register --namespace Microsoft.App --only-show-errors >/dev/null || true
az containerapp env show -n "$ENVNAME" -g "$RG" -o none --only-show-errors \
  || { echo "Container Apps environment $ENVNAME not found: run deploy/azure/deploy_aca.sh first"; exit 1; }
az containerapp show -n "$APP" -g "$RG" -o none --only-show-errors \
  || { echo "$APP not found: run deploy/azure/deploy_aca.sh first"; exit 1; }

GF_PASS="${GRAFANA_ADMIN_PASSWORD:-}"
if [[ -z "$GF_PASS" ]]; then
  read -r -s -p "Choose a Grafana admin password (input hidden, min 12 chars): " GF_PASS; echo
fi
[[ ${#GF_PASS} -ge 12 ]] || { echo "Grafana admin password must be at least 12 characters"; exit 1; }

echo ">> 1/4 Building monitoring images in $ACR"
az acr build -r "$ACR" -t "ihds-prometheus:$MON_TAG" -f deploy/monitoring/prometheus/Dockerfile . --only-show-errors -o none
az acr build -r "$ACR" -t "ihds-grafana:$MON_TAG" -f deploy/monitoring/grafana/Dockerfile . --only-show-errors -o none

ACR_USER=$(az acr credential show -n "$ACR" --query username -o tsv)
ACR_PASS=$(az acr credential show -n "$ACR" --query 'passwords[0].value' -o tsv)   # never printed
REG=(--registry-server "$ACR.azurecr.io" --registry-username "$ACR_USER" --registry-password "$ACR_PASS")

echo ">> 2/4 $PROM_APP (internal ingress only)"
if az containerapp show -n "$PROM_APP" -g "$RG" -o none --only-show-errors 2>/dev/null; then
  az containerapp update -n "$PROM_APP" -g "$RG" --image "$ACR.azurecr.io/ihds-prometheus:$MON_TAG" --only-show-errors -o none
else
  az containerapp create -n "$PROM_APP" -g "$RG" --environment "$ENVNAME" \
    --image "$ACR.azurecr.io/ihds-prometheus:$MON_TAG" "${REG[@]}" \
    --ingress internal --target-port 9090 --transport http --allow-insecure \
    --cpu 0.5 --memory 1.0Gi --min-replicas 1 --max-replicas 1 \
    --only-show-errors -o none
fi

echo ">> 3/4 $GRAFANA_APP (public, anonymous viewer=$ANON)"
GF_ENV=("PROMETHEUS_URL=http://$PROM_APP" "GF_SECURITY_ADMIN_PASSWORD=secretref:grafana-admin-password"
        "GF_AUTH_ANONYMOUS_ENABLED=$ANON" "GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer" "GF_USERS_ALLOW_SIGN_UP=false")
if az containerapp show -n "$GRAFANA_APP" -g "$RG" -o none --only-show-errors 2>/dev/null; then
  az containerapp secret set -n "$GRAFANA_APP" -g "$RG" --secrets "grafana-admin-password=$GF_PASS" --only-show-errors -o none
  az containerapp update -n "$GRAFANA_APP" -g "$RG" --image "$ACR.azurecr.io/ihds-grafana:$MON_TAG" \
    --set-env-vars "${GF_ENV[@]}" --only-show-errors -o none
else
  az containerapp create -n "$GRAFANA_APP" -g "$RG" --environment "$ENVNAME" \
    --image "$ACR.azurecr.io/ihds-grafana:$MON_TAG" "${REG[@]}" \
    --ingress external --target-port 3000 \
    --secrets "grafana-admin-password=$GF_PASS" --env-vars "${GF_ENV[@]}" \
    --cpu 0.5 --memory 1.0Gi --min-replicas 1 --max-replicas 1 \
    --only-show-errors -o none
fi
unset GF_PASS GRAFANA_ADMIN_PASSWORD ACR_PASS
# Note: Grafana reads GF_SECURITY_ADMIN_PASSWORD only when its database is first created; on an existing
# replica change the password in the Grafana UI (or restart the revision after `secret set`).

echo ">> 4/4 Turning on the metrics push in $APP -> http://$PROM_APP/api/v1/write"
GRAFANA_RW_URL="http://$PROM_APP/api/v1/write" GRAFANA_RW_USER="" GRAFANA_RW_TOKEN="" \
  DEPLOYMENT_LABEL="azure-container-apps" APP="$APP" RG="$RG" ACR="$ACR" TAG="${TAG:-v3}" \
  bash deploy/azure/enable_metrics_push.sh

GF_FQDN=$(az containerapp show -n "$GRAFANA_APP" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv)
echo ">> Grafana: https://$GF_FQDN  (dashboard 'IHDS API'; admin login: admin / the password you chose)"
echo "   Data appears ~1 min after the app's new revision starts and only for /assess calls made after that."
