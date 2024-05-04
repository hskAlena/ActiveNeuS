output_dir='outputs'
method_name='active_uncert_neus-acc'

exp_name='uncert_neus-acc-frontier-lr5e4-chair'
timestamp='2023-08-26_110214'
config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
ns-eval --load-config="${config_path}" \
        --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"

exp_name='uncert_neus-acc-frontier-lr5e4-drums'
timestamp='2023-08-26_123536'
config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
ns-eval --load-config="${config_path}" \
        --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"

exp_name='uncert_neus-acc-frontier-lr5e4-ficus'
timestamp='2023-08-26_122233'
config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
ns-eval --load-config="${config_path}" \
        --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"

exp_name='uncert_neus-acc-frontier-lr5e4-hotdog'
timestamp='2023-08-26_114554'
config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
ns-eval --load-config="${config_path}" \
        --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"

exp_name='uncert_neus-acc-frontier-lr5e4-mic'
timestamp='2023-08-26_115122'
config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
ns-eval --load-config="${config_path}" \
        --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"

exp_name='uncert_neus-acc-frontier-lr5e4-lego'
timestamp='2023-08-26_125215'
config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
ns-eval --load-config="${config_path}" \
        --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"

exp_name='uncert_neus-acc-frontier-lr5e4-materials'
timestamp='2023-08-26_225430'
config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
ns-eval --load-config="${config_path}" \
        --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"

exp_name='uncert_neus-acc-frontier-lr5e4-ship'
timestamp='2023-08-26_110222'
config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
ns-eval --load-config="${config_path}" \
        --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"
