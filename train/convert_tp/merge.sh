# 将N个TP并行的转换成单卡的
run_cmd="python convert_tp.py --input-folder $1 --output-folder $2 --target-tp 1"
echo ${run_cmd}
eval ${run_cmd}