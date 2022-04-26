: ${datasets_dir:="./data"}
: ${window_size:=16}
: ${window_stride:=4}
: ${sim_method:="fix_ma"}
: ${local_sample_size:=224}
: ${remote_sample_size:=224}
: ${output_prefix:="fusion"}
: ${start_filter_interval:=1}
: ${max_filter_interval:=11}
: ${end_filter_interval:=${max_filter_interval}}
: ${remote_lag:=1}
: ${siminet_path:="none"}
: ${local_model:="resnet-18"}
: ${remote_model:="resnext-101"}
: ${split:="cross_subject_background"}

if [ -z ${PRUN_CPU_RANK+x} ]; then
  my_rank=0
  echo "Not running in parallel.."
else
  echo "Running in parallel.."
  my_rank=${PRUN_CPU_RANK}
  # per_node_delay=$(( ${max_filter_interval} / ${NUM_NODES} + 1))
  per_node_delay=$(( ${max_filter_interval} / ${NUM_NODES} ))
  start_filter_interval=$(( ${my_rank} * per_node_delay + 1  ))
  end_filter_interval=$(( ${start_filter_interval} + ${per_node_delay} ))
  if (( ${end_filter_interval} > (${max_filter_interval} + 1) )); then
    end_filter_interval=$(( ${max_filter_interval} + 1 ))
  fi
  echo "Total nodes: $NUM_NODES on rank ${PRUN_CPU_RANK}"
  echo "Filter interval: $start_filter_interval to $end_filter_interval"
fi

output_file="${output_prefix}_${sim_method}_window_${window_size}_${window_stride}_models_${local_sample_size}_${remote_sample_size}_rank_${my_rank}_lag_${remote_lag}_background.log"

rm ${output_file}
echo "Log file: ${output_file}"

for (( i=${start_filter_interval}; i<${end_filter_interval}; i++ ))
do
python3 main.py --datasets_dir ${datasets_dir} --local_scores_dir ${datasets_dir}/PKUMMD/scores_dump/${local_model}/sample_duration_${window_size}/image_size_${local_sample_size}/window_stride_${window_stride}/${split}/val/ --remote_scores_dir ${datasets_dir}/PKUMMD/scores_dump/${remote_model}/sample_duration_${window_size}/image_size_${remote_sample_size}/window_stride_${window_stride}/${split}/val/ --filter_interval $i --remote_lag ${remote_lag} --sim_method ${sim_method} --window_stride ${window_stride} --window_size ${window_size} --siminet_path ${siminet_path} --n_classes 52 >> ${output_file}
done
