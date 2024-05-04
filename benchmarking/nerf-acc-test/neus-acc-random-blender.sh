output_dir='outputs'
method_name='active_neus-acc'

exp_name='neus-acc-random-neusfix-5e4-mic'
timestamp='2023-09-15_153028'
config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
ns-eval --load-config="${config_path}" --test_mode val --no-image_output \
        --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"

exp_name='neus-acc-random-neusfix-5e4-drums'
timestamp='2023-09-15_184339'
config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
ns-eval --load-config="${config_path}" --test_mode val --no-image_output \
        --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"

exp_name='neus-acc-random-neusfix-5e4-ficus'
timestamp='2023-09-15_160358'
config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
ns-eval --load-config="${config_path}" --test_mode val --no-image_output \
        --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"

exp_name='neus-acc-random-neusfix-5e4-hotdog'
timestamp='2023-09-15_171516'
config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
ns-eval --load-config="${config_path}" --test_mode val --no-image_output \
        --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"

exp_name='neus-acc-random-neusfix-5e4-chair'
timestamp='2023-09-15_163746'
config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
ns-eval --load-config="${config_path}" --test_mode val --no-image_output \
        --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"

exp_name='neus-acc-random-neusfix-lr5e4-lego'
timestamp='2023-09-15_201925'
config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
ns-eval --load-config="${config_path}" --test_mode val --no-image_output \
        --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"

exp_name='neus-acc-random-neusfix-lr5e4-materials'
timestamp='2023-09-15_175556'
config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
ns-eval --load-config="${config_path}" --test_mode val --no-image_output \
        --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"

exp_name='neus-acc-random-neusfix-lr5e4-ship'
timestamp='2023-09-15_192533'
config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
ns-eval --load-config="${config_path}" --test_mode val --no-image_output \
        --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"

# exp_name='neus-acc-random-5e4-chair'
# timestamp='2023-08-27_171641'
# config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
# ns-eval --load-config="${config_path}" \
#         --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"

# exp_name='neus-acc-random-5e4-drums'
# timestamp='2023-08-27_154922'
# config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
# ns-eval --load-config="${config_path}" \
#         --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"

# exp_name='neus-acc-random-5e4-ficus'
# timestamp='2023-08-27_165114'
# config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
# ns-eval --load-config="${config_path}" \
#         --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"

# exp_name='neus-acc-random-5e4-hotdog'
# timestamp='2023-08-27_150943'
# config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
# ns-eval --load-config="${config_path}" \
#         --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"

# exp_name='neus-acc-random-5e4-mic'
# timestamp='2023-08-27_193029'
# config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
# ns-eval --load-config="${config_path}" \
#         --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"

# exp_name='neus-acc-random-lr5e4-lego'
# timestamp='2023-08-27_164503'
# config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
# ns-eval --load-config="${config_path}" \
#         --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"

# exp_name='neus-acc-random-lr5e4-materials'
# timestamp='2023-08-27_181604'
# config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
# ns-eval --load-config="${config_path}" \
#         --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"

# exp_name='neus-acc-random-lr5e4-ship'
# timestamp='2023-08-27_172506'
# config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
# ns-eval --load-config="${config_path}" \
#         --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"
