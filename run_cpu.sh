LOG_DIR=${LOG_DIR:-'hf_oob_log'}
if [ ! -d ${LOG_DIR} ]; then
    mkdir -p ${LOG_DIR}
fi

./run_all_task_cpu.sh --model_dtype bfloat16 --warm_up_steps 10 --run_steps 10 > ${LOG_DIR}/cpu_output_OOB.log 2>&1
./run_all_task_cpu.sh --model_dtype bfloat16 --warm_up_steps 10 --run_steps 10 --torch_compile True > ${LOG_DIR}/cpu_output_compile.log 2>&1

python analyse_logs.py --file_names ${LOG_DIR}/cpu_output_OOB.log,${LOG_DIR}/cpu_output_compile.log --out_name ${LOG_DIR}/cpu_benchmark.log
