: ${datasets_dir:="./data"}
: ${window_size:=16}
: ${window_stride:=4}
: ${sim_method:="fix_ma"}
: ${local_sample_size:=224}
: ${remote_sample_size:=224}
: ${output_prefix:="visualize"}
: ${filter_interval:=5}
: ${remote_lag:=1}
: ${siminet_path:="none"}
: ${local_model:="resnet-18"}
: ${remote_model:="resnext-101"}
: ${split:="cross_subject_background"}

output_file="${output_prefix}_${sim_method}_window_${window_size}_${window_stride}_models_${local_sample_size}_${remote_sample_size}_rank_${my_rank}_lag_${remote_lag}_background.log"

rm ${output_file}
echo "Log file: ${output_file}"

python3 visualize.py \
  --datasets_dir ${datasets_dir} \
  --local_scores_dir ${datasets_dir}/PKUMMD/scores_dump/${local_model}/sample_duration_${window_size}/image_size_${local_sample_size}/window_stride_${window_stride}/${split}/val/ \
  --remote_scores_dir ${datasets_dir}/PKUMMD/scores_dump/${remote_model}/sample_duration_${window_size}/image_size_${remote_sample_size}/window_stride_${window_stride}/${split}/val/ \
  --filter_interval ${filter_interval} \
  --remote_lag ${remote_lag} \
  --sim_method ${sim_method} \
  --window_stride ${window_stride} \
  --window_size ${window_size} \
  --siminet_path ${siminet_path} \
  --n_classes 52 >> ${output_file} 
