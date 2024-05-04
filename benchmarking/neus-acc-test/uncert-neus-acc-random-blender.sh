output_dir='outputs'
method_name='active_uncert_neus-acc'

exp_name='uncert_neus-acc-random-lr5e4-chair'
timestamp='2023-08-26_111443'
config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
ns-eval --load-config="${config_path}" \
        --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"

exp_name='uncert_neus-acc-random-lr5e4-drums'
timestamp='2023-08-27_164309'
config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
ns-eval --load-config="${config_path}" \
        --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"

exp_name='uncert_neus-acc-random-lr5e4-ficus'
timestamp='2023-08-26_130818'
config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
ns-eval --load-config="${config_path}" \
        --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"

exp_name='uncert_neus-acc-random-lr5e4-hotdog'
timestamp='2023-08-26_121609'
config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
ns-eval --load-config="${config_path}" \
        --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"

exp_name='uncert_neus-acc-random-lr5e4-mic'
timestamp='2023-08-27_143340'
config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
ns-eval --load-config="${config_path}" \
        --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"

exp_name='uncert_neus-acc-random-lr5e4-lego'
timestamp='2023-08-27_143712'
config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
ns-eval --load-config="${config_path}" \
        --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"

exp_name='uncert_neus-acc-random-lr5e4-materials'
timestamp='2023-08-27_200154'
config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
ns-eval --load-config="${config_path}" \
        --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"

exp_name='uncert_neus-acc-random-lr5e4-ship'
timestamp='2023-08-26_125514'
config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
ns-eval --load-config="${config_path}" \
        --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"