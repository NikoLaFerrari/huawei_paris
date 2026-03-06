import yaml

with open('config.yaml', 'r') as f:
    raw = yaml.safe_load(f)

with open('../../../../../home/pl/telecom/results_telecom_swap/bench_60fa0500cef8/config_60fa0500cef8.yaml', 'r') as g:
    raw1 = yaml.safe_load(g)

#print(raw.get('model_config'))
#print(raw.get('runner_config'))
mc = raw.get('model_config;')
print(mc['pp_interleave_num'])
print(raw.get('model_config').get('pp_interleave_num'))
