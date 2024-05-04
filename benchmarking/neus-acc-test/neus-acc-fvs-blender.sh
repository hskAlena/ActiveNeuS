output_dir='outputs'
method_name='active_neus-acc'

exp_name='neus-acc-fvs-neusfix-5e4-mic'
timestamp='2023-09-16_004614'
config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
ns-eval --load-config="${config_path}" --test_mode val --no-image_output \
        --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"

exp_name='neus-acc-fvs-neusfix-5e4-drums'
timestamp='2023-09-16_045216'
config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
ns-eval --load-config="${config_path}" --test_mode val --no-image_output \
        --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"

exp_name='neus-acc-fvs-neusfix-5e4-ficus'
timestamp='2023-09-16_013100'
config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
ns-eval --load-config="${config_path}" --test_mode val --no-image_output \
        --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"

exp_name='neus-acc-fvs-neusfix-5e4-hotdog'
timestamp='2023-09-16_030538'
config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
ns-eval --load-config="${config_path}" --test_mode val --no-image_output \
        --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"

exp_name='neus-acc-fvs-neusfix-5e4-chair'
timestamp='2023-09-16_021700'
config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
ns-eval --load-config="${config_path}" --test_mode val --no-image_output \
        --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"

exp_name='neus-acc-fvs-neusfix-lr5e4-lego'
timestamp='2023-09-16_064519'
config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
ns-eval --load-config="${config_path}" --test_mode val --no-image_output \
        --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"

exp_name='neus-acc-fvs-neusfix-lr5e4-materials'
timestamp='2023-09-16_035712'
config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
ns-eval --load-config="${config_path}" --test_mode val --no-image_output \
        --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"

exp_name='neus-acc-fvs-neusfix-lr5e4-ship'
timestamp='2023-09-16_054517'
config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
ns-eval --load-config="${config_path}" --test_mode val --no-image_output \
        --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"
