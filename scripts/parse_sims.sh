key_name=$2
target_val=$3
jq --arg kn $key_name --argjson tv $target_val -n 'inputs | select(.[$kn]==$tv) | input_filename' $1/**/metadata.json
