# Deploying to Azure Container Apps (Cloud Shell)

The container runs `scripts/start_all.sh`, which starts three processes: the MCP tool server
(`mcp_server/server.py`, streamable HTTP on 127.0.0.1:8765, not exposed), the FastAPI service (`api/main.py`,
port 8000; its LangGraph agents get their tools from the MCP server) and Streamlit (port 8501; it calls the API
through `IHDS_API_URL`). Container Apps ingress exposes 8501 only; the ACI variant exposes 8501 and 8000.
The OpenAI/Vocareum key is passed to the API and Streamlit processes, not to the MCP server.
The image build runs `python -m ihds.pipeline`,
so the Evidently reports in `monitoring/reports/` are already present when the app starts; the
**Monitoring (Evidently)** tab can also regenerate them on demand.

## 1. Open Cloud Shell (Bash) and get the code

```bash
git clone https://github.com/sandipdey2/integrative-healthcare-diagnostic-system.git
cd integrative-healthcare-diagnostic-system
az account show --query name -o tsv          # confirm the lab subscription (do not paste IDs anywhere public)
az group list --query "[].{name:name,location:location}" -o table   # find your existing Regroup_* group
```

## 2. Deploy to Azure Container Apps

```bash
export RG=<existing-resource-group>      # e.g. the Regroup_* group the lab created
export ACR=ihdsacr$RANDOM                # must be globally unique, lowercase letters/digits only
echo $ACR                                # note it down for cleanup
bash deploy/azure/deploy_aca.sh
```

Optional live LLM writer (gpt-4o-mini through Vocareum; otherwise template mode):

```bash
export WITH_OPENAI=1
export OPENAI_API_BASE=https://openai.vocareum.com/v1
read -r -s OPENAI_API_KEY && export OPENAI_API_KEY      # paste the voc-... key; input hidden, not in history
bash deploy/azure/deploy_aca.sh
```

The key is stored as the Container Apps secret `openai-key` and exposed as `OPENAI_API_KEY=secretref:openai-key`;
`OPENAI_API_BASE` is a plain environment variable. The script also works on an existing app (it sets the secret
and env vars on update). Check: the app sidebar shows "LLM key detected · endpoint: https://openai.vocareum.com/v1".

## 3. Verify

```bash
FQDN=$(az containerapp show -n ihds-app -g $RG --query properties.configuration.ingress.fqdn -o tsv)
curl -s https://$FQDN/_stcore/health          # -> ok
az containerapp logs show -n ihds-app -g $RG --tail 50 | grep -i -E "uvicorn|mcp"   # API and MCP server started
az containerapp revision list -n ihds-app -g $RG -o table
az containerapp logs show -n ihds-app -g $RG --tail 50
```

Open `https://$FQDN`, run a record on **Risk assessment**, then open **Monitoring (Evidently)**.
Suggested screenshots for the submission/deck: Container App overview (Running, URL), the app's
Risk assessment result, the Monitoring tab table, one Evidently HTML report, the `curl` health output.

## Alternative: Azure Container Instances

```bash
bash deploy/azure/deploy_aci.sh                # uses the image already built in $ACR
curl -s http://<fqdn>:8000/health              # {"status":"ok","tools_transport":"mcp-http",...}
```

Re-deploying a changed image to an existing app: `export TAG=v2` before `bash deploy/azure/deploy_aca.sh`.

## Troubleshooting (lab subscriptions)

| Symptom | Fix |
|---|---|
| `AuthorizationFailed` creating a resource group | Do not create one; use the existing `Regroup_*` group (`export RG=...`). |
| `TasksOperationsNotAllowed` on `az acr build` | ACR Tasks are disabled on some subscriptions. Build locally with Docker Desktop: `docker build -t $ACR.azurecr.io/ihds:v1 . && az acr login -n $ACR && docker push $ACR.azurecr.io/ihds:v1`, then re-run the script (the build step can be commented out). |
| Region quota / `not available in location` | `export LOCATION=westus2` (or another allowed region) and re-run. A previous lab had 0 App Service quota in East US; Container Apps quota is separate. |
| `MissingSubscriptionRegistration` for Microsoft.App | `az provider register -n Microsoft.App --wait` |
| Token expired / `AADSTS` errors | `az login --use-device-code` |
| App shows "Please wait..." forever | Streamlit needs WebSockets; Container Apps ingress supports them. Check `az containerapp logs show`. |

## Cleanup (after grading)

```bash
bash deploy/azure/cleanup.sh       # deletes the app, environment, ACI and ACR; never the resource group
```
