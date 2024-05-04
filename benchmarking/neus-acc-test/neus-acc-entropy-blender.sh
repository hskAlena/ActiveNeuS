output_dir='outputs'
method_name='active_neus-acc'

output_dir='outputs'
method_name='active_neus-acc'

exp_name='neus-acc-entropy-topk-xgrid-200k-mic'
timestamp='2023-10-16_002829'
config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
ns-eval --load-config="${config_path}" --test_mode val --no-image_output \
        --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"

exp_name='neus-acc-entropy-topk-xgrid-200k-drums'
timestamp='2023-10-18_001116'
config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
ns-eval --load-config="${config_path}" --test_mode val --no-image_output \
        --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"

exp_name='neus-acc-entropy-topk-xgrid-200k-ficus'
timestamp='2023-10-16_113151'
config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
ns-eval --load-config="${config_path}" --test_mode val --no-image_output \
        --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"

exp_name='neus-acc-entropy-topk-xgrid-200k-hotdog'
timestamp='2023-10-16_171633'
config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
ns-eval --load-config="${config_path}" --test_mode val --no-image_output \
        --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"

exp_name='neus-acc-entropy-topk-xgrid-200k-chair'
timestamp='2023-10-15_172246'
config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
ns-eval --load-config="${config_path}" --test_mode val --no-image_output \
        --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"

exp_name='neus-acc-entropy-topk-xgrid-200k-lego'
timestamp='2023-10-17_114216'
config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
ns-eval --load-config="${config_path}" --test_mode val --no-image_output \
        --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"

exp_name='neus-acc-entropy-topk-xgrid-200k-materials'
timestamp='2023-10-17_114815'
config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
ns-eval --load-config="${config_path}" --test_mode val --no-image_output \
        --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"

exp_name='neus-acc-entropy-topk-xgrid-200k-ship'
timestamp='2023-10-18_073531'
config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
ns-eval --load-config="${config_path}" --test_mode val --no-image_output \
        --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"

##########################################################
exp_name='neus-acc-entropy-neusfix-5e4-mic'
timestamp='2023-09-16_004651'
config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
ns-eval --load-config="${config_path}" --test_mode val --no-image_output \
        --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"

exp_name='neus-acc-entropy-neusfix-5e4-drums'
timestamp='2023-09-16_044607'
config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
ns-eval --load-config="${config_path}" --test_mode val --no-image_output \
        --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"

exp_name='neus-acc-entropy-neusfix-5e4-ficus'
timestamp='2023-09-16_013037'
config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
ns-eval --load-config="${config_path}" --test_mode val --no-image_output \
        --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"

exp_name='neus-acc-entropy-neusfix-5e4-hotdog'
timestamp='2023-09-16_030223'
config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
ns-eval --load-config="${config_path}" --test_mode val --no-image_output \
        --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"

exp_name='neus-acc-entropy-neusfix-5e4-chair'
timestamp='2023-09-16_021511'
config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
ns-eval --load-config="${config_path}" --test_mode val --no-image_output \
        --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"

exp_name='neus-acc-entropy-neusfix-5e4-lego'
timestamp='2023-09-16_064059'
config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
ns-eval --load-config="${config_path}" --test_mode val --no-image_output \
        --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"

exp_name='neus-acc-entropy-neusfix-5e4-materials'
timestamp='2023-09-16_035253'
config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
ns-eval --load-config="${config_path}" --test_mode val --no-image_output \
        --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"

exp_name='neus-acc-entropy-neusfix-5e4-ship'
timestamp='2023-09-16_053733'
config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
ns-eval --load-config="${config_path}" --test_mode val --no-image_output \
        --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"
