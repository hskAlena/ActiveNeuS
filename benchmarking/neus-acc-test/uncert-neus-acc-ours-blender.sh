output_dir='outputs'
method_name='active_uncert_neus-acc'

exp_name='uncert_neus-acc-frontier-dist-ent-xgrid-200k-chair'
timestamp='2023-10-18_205724'
config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
ns-eval --load-config="${config_path}" --test_mode val --no-image_output \
        --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"

exp_name='uncert_neus-acc-frontier-dist-ent-xgrid-200k-ficus'
timestamp='2023-10-18_034338'
config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
ns-eval --load-config="${config_path}" --test_mode val --no-image_output \
        --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"

exp_name='uncert_neus-acc-frontier-dist-ent-xgrid-200k-mic'
timestamp='2023-10-17_134553'
config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
ns-eval --load-config="${config_path}" --test_mode val --no-image_output \
        --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"

exp_name='uncert_neus-acc-frontier-dist-ent-xgrid-200k-hotdog'
timestamp='2023-10-19_144200'
config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
ns-eval --load-config="${config_path}" --test_mode val --no-image_output \
        --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"

###########################################################################
exp_name='uncert_neus-acc-frontier-dist-ent-200k-mic'
timestamp='2023-10-16_113748'
config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
ns-eval --load-config="${config_path}" --test_mode val --no-image_output \
        --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"

exp_name='uncert_neus-acc-frontier-dist-ent-200k-ship'
timestamp='2023-10-16_133715'
config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
ns-eval --load-config="${config_path}" --test_mode val --no-image_output \
        --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"
########################################################################################
exp_name='uncert_neus-acc-frontier-dist-ent-xgrid-10image-20k-chair'
timestamp='2023-10-18_214102'
config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
ns-eval --load-config="${config_path}" --test_mode val --no-image_output \
        --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"

exp_name='uncert_neus-acc-frontier-dist-ent-xgrid-10image-20k-drums'
timestamp='2023-10-19_045626'
config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
ns-eval --load-config="${config_path}" --test_mode val --no-image_output \
        --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"

exp_name='uncert_neus-acc-frontier-dist-ent-xgrid-10image-20k-ficus'
timestamp='2023-10-18_200440'
config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
ns-eval --load-config="${config_path}" --test_mode val --no-image_output \
        --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"

exp_name='uncert_neus-acc-frontier-dist-ent-xgrid-10image-20k-hotdog'
timestamp='2023-10-18_233908'
config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
ns-eval --load-config="${config_path}" --test_mode val --no-image_output \
        --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"

exp_name='uncert_neus-acc-frontier-dist-ent-xgrid-10image-20k-lego'
timestamp='2023-10-19_100403'
config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
ns-eval --load-config="${config_path}" --test_mode val --no-image_output \
        --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"

exp_name='uncert_neus-acc-frontier-dist-ent-xgrid-10image-20k-materials'
timestamp='2023-10-19_024621'
config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
ns-eval --load-config="${config_path}" --test_mode val --no-image_output \
        --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"

exp_name='uncert_neus-acc-frontier-dist-ent-xgrid-10image-20k-mic'
timestamp='2023-10-18_181038'
config_path="${output_dir}/${exp_name}/${method_name}/${timestamp}/config.yml"
ns-eval --load-config="${config_path}" --test_mode val --no-image_output \
        --output-path="${output_dir}/${exp_name}/${method_name}/${timestamp}/output.json"

######################################################3
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
