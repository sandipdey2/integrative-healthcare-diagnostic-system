#!/usr/bin/env bash
# Alternative: Azure Container Instances (single container, public IP + DNS label).
# Requires the image already in ACR (run deploy_aca.sh steps 1-2, or: az acr build -r $ACR -t ihds:v1 .)
#   export RG=<existing-resource-group> ACR=<acr-name>
#   bash deploy/azure/deploy_aci.sh
# Port 8000 is public here, so the Prometheus /metrics endpoint is switched off (IHDS_METRICS=0).
set -euo pipefail
: "${RG:?}"; : "${ACR:?}"
NAME="${NAME:-ihds-aci}"
TAG="${TAG:-v1}"
DNS="${DNS:-ihds-$RANDOM}"
LOCATION="${LOCATION:-$(az group show -n "$RG" --query location -o tsv)}"
az provider register --namespace Microsoft.ContainerInstance --only-show-errors >/dev/null || true
ACR_USER=$(az acr credential show -n "$ACR" --query username -o tsv)
ACR_PASS=$(az acr credential show -n "$ACR" --query 'passwords[0].value' -o tsv)
az container create -g "$RG" -n "$NAME" -l "$LOCATION" \
  --image "$ACR.azurecr.io/ihds:$TAG" \
  --registry-login-server "$ACR.azurecr.io" --registry-username "$ACR_USER" --registry-password "$ACR_PASS" \
  --os-type Linux --cpu 1 --memory 2 \
  --ports 8501 8000 --ip-address Public --dns-name-label "$DNS" \
  --environment-variables IHDS_METRICS=0 \
  --only-show-errors -o none
unset ACR_PASS
FQDN=$(az container show -g "$RG" -n "$NAME" --query ipAddress.fqdn -o tsv)
echo ">> Deployed: http://$FQDN:8501 (Streamlit)  http://$FQDN:8000/docs (FastAPI)"
