find sim_data/balance_exp -type f -name metadata.json -exec jq -r --arg var_name 'dist_var' '.[$var_name] // empty | [input_filename, .]' {} +
