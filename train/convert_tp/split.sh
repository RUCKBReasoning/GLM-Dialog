# 将单卡转换成N卡并行
run_cmd="python convert_tp.py --input-folder $1 --output-folder $2 --target-tp $3"
echo ${run_cmd}
eval ${run_cmd}
