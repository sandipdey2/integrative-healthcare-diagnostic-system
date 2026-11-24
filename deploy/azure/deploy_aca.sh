#!/usr/bin/env bash
# Deploy the IHDS Streamlit app to Azure Container Apps from Azure Cloud Shell (Bash).
#
# Usage (from the repo root in Cloud Shell):
#   export RG=<existing-resource-group>          # Vocareum labs: use the pre-created Regroup_* group
#   export ACR=<globally-unique-acr-name>        # 5-50 lowercase letters/digits, e.g. ihdsacr$RANDOM
#   bash deploy/azure/deploy_aca.sh
#
# Optional:
#   export LOCATION=eastus                       # default: the resource group's location
#   export WITH_OPENAI=1                         # prompts (hidden input) for an OpenAI key, stored as a Container Apps secret
#
# Notes for Udacity/Vocareum lab subscriptions:
#   * You usually cannot create resource groups -> this script never creates one.
#   * If `az acr build` fails with "TasksOperationsNotAllowed", see deploy/azure/README.md (fallback options).
#   * If the token expires mid-run: `az login --use-device-code`, then re-run (steps are idempotent-ish).
set -euo pipefail

: "${RG:?set RG to an existing resource group}"
: "${ACR:?set ACR to a unique registry name}"
APP="${APP:-ihds-app}"
ENVNAME="${ENVNAME:-ihds-env}"
TAG="${TAG:-v1}"
LOCATION="${LOCATION:-$(az group show -n "$RG" --query location -o tsv)}"
IMAGE="$ACR.azurecr.io/ihds:$TAG"

echo ">> Using resource group $RG in $LOCATION"
az extension add --name containerapp --upgrade --only-show-errors >/dev/null
az provider register --namespace Microsoft.App --only-show-errors >/dev/null || true
az provider register --namespace Microsoft.OperationalInsights --only-show-errors >/dev/null || true

echo ">> 1/4 Container registry $ACR"
if ! az acr show -n "$ACR" -g "$RG" >/dev/null 2>&1; then
  az acr create -g "$RG" -n "$ACR" --sku Basic --admin-enabled true -l "$LOCATION" --only-show-errors -o none
fi

echo ">> 2/4 Cloud build of the image (az acr build, no local Docker needed)"
az acr build -r "$ACR" -t "ihds:$TAG" . --only-show-errors

echo ">> 3/4 Container Apps environment $ENVNAME"
if ! az containerapp env show -n "$ENVNAME" -g "$RG" >/dev/null 2>&1; then
  az containerapp env create -n "$ENVNAME" -g "$RG" -l "$LOCATION" --only-show-errors -o none
fi

echo ">> 4/4 Container app $APP"
ACR_USER=$(az acr credential show -n "$ACR" --query username -o tsv)
ACR_PASS=$(az acr credential show -n "$ACR" --query 'passwords[0].value' -o tsv)   # kept in a variable, never printed

EXTRA=()
if [[ "${WITH_OPENAI:-0}" == "1" ]]; then
  read -r -s -p "OpenAI API key (input hidden): " OPENAI_KEY; echo
  EXTRA=(--secrets "openai-key=$OPENAI_KEY" --env-vars "OPENAI_API_KEY=secretref:openai-key")
fi

if az containerapp show -n "$APP" -g "$RG" >/dev/null 2>&1; then
  az containerapp update -n "$APP" -g "$RG" --image "$IMAGE" --only-show-errors -o none
else
  az containerapp create -n "$APP" -g "$RG" \
    --environment "$ENVNAME" \
    --image "$IMAGE" \
    --registry-server "$ACR.azurecr.io" --registry-username "$ACR_USER" --registry-password "$ACR_PASS" \
    --target-port 8501 --ingress external \
    --cpu 1.0 --memory 2.0Gi \
    --min-replicas 1 --max-replicas 2 \
    "${EXTRA[@]}" \
    --only-show-errors -o none
fi
unset ACR_PASS OPENAI_KEY

FQDN=$(az containerapp show -n "$APP" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv)
echo ">> Deployed: https://$FQDN"
echo ">> Health:   curl -s https://$FQDN/_stcore/health   (expect: ok)"
