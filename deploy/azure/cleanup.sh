#!/usr/bin/env bash
# Delete only the resources created by the deploy scripts (never the resource group itself).
set -euo pipefail
: "${RG:?}"; : "${ACR:?}"
az containerapp delete -n "${APP:-ihds-app}" -g "$RG" --yes --only-show-errors || true
az containerapp env delete -n "${ENVNAME:-ihds-env}" -g "$RG" --yes --only-show-errors || true
az container delete -n "${NAME:-ihds-aci}" -g "$RG" --yes --only-show-errors || true
az acr delete -n "$ACR" -g "$RG" --yes --only-show-errors || true
echo "Cleanup requested. A Log Analytics workspace created by the environment may remain: az monitor log-analytics workspace list -g \$RG -o table"
