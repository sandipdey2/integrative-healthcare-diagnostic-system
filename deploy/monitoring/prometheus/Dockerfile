# Option B Prometheus image (receives remote_write from ihds-app). Build context: repo root.
#   az acr build -r $ACR -t ihds-prometheus:v1 -f deploy/monitoring/prometheus/Dockerfile .
FROM prom/prometheus:v3.5.0
COPY deploy/monitoring/prometheus/prometheus.yml /etc/prometheus/prometheus.yml
# Storage is the container's disk: it survives restarts of the process but not a new revision/replica.
CMD ["--config.file=/etc/prometheus/prometheus.yml", "--storage.tsdb.path=/prometheus", \
     "--storage.tsdb.retention.time=15d", "--web.enable-remote-write-receiver"]
