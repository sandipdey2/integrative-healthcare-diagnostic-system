# Option B Grafana image with the IHDS datasource + dashboard baked in. Build context: repo root.
#   az acr build -r $ACR -t ihds-grafana:v1 -f deploy/monitoring/grafana/Dockerfile .
# Runtime settings come from Container App env vars / secrets (deploy/azure/deploy_monitoring_aca.sh):
#   PROMETHEUS_URL (e.g. http://ihds-prometheus), GF_SECURITY_ADMIN_PASSWORD (secretref).
FROM grafana/grafana:12.1.0
COPY monitoring/grafana/provisioning /etc/grafana/provisioning
COPY monitoring/grafana/dashboards /var/lib/grafana/dashboards
ENV GF_AUTH_ANONYMOUS_ENABLED=true \
    GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer \
    GF_USERS_ALLOW_SIGN_UP=false \
    GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH=/var/lib/grafana/dashboards/ihds_api.json \
    GF_ANALYTICS_REPORTING_ENABLED=false \
    GF_ANALYTICS_CHECK_FOR_UPDATES=false \
    PROMETHEUS_URL=http://prometheus:9090
