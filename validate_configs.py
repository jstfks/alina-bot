import yaml
import json

files = [
    'alerts.yml',
    'prometheus.yml',
    'alertmanager.yml.tmpl',
    'docker-compose.observability.yml',
    'grafana/provisioning/datasources/datasources.yml',
    'grafana/provisioning/dashboards/dashboards.yml',
    'grafana/provisioning/dashboards/alina-bot-overview.json'
]

for f in files:
    with open(f, encoding='utf-8') as fh:
        if f.endswith('.json'):
            json.load(fh)
        else:
            yaml.safe_load(fh)
    print(f + ': OK')

print('All config files valid!')