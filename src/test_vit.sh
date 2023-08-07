#经过测试的全部能用的代码（LWM除外）####先测试不带样本的，然后再测试带样本200的一批
#最佳参数：
#不带样本回放的：alpha=0.85， scale=1
nepochs=10  # Set the value of nepochs here

# Generate 33 random numbers in each of the specified ranges
alpha_values1=$(awk -v min=0.001 -v max=0.2 'BEGIN{srand(); for(i=1;i<=10;i++) print min+rand()*(max-min)}')
alpha_values2=$(awk -v min=0.01 -v max=0.09 'BEGIN{srand(); for(i=1;i<=10;i++) print min+rand()*(max-min)}')
alpha_values3=$(awk -v min=0.2 -v max=3.2 'BEGIN{srand(); for(i=1;i<=10;i++) print min+rand()*(max-min)}')

# Combine all alpha values into one array
alpha_values="$alpha_values1 $alpha_values2 $alpha_values3"

for dataset in mnist
do
    for num_tasks in 5 3
    do
        results_path="../ViT_results/test_VIT_${dataset}_${num_tasks}"
        # python main_incremental.py --exp-name vit_dataset_${dataset}_tasks_${num_tasks}_epochs_${nepochs} --approach lwf --nepochs $nepochs --num-tasks $num_tasks --datasets $dataset --results-path $results_path     
        #$alpha_values
        for alpha in $alpha_values
        do
            for scale_factor in 1.0
            do
                exp_name="vitRD_${alpha}_scale_${scale_factor}_dataset_${dataset}_tasks_${num_tasks}_epochs_${nepochs}"
                echo "Running experiment with alpha = $alpha, scale_factor = $scale_factor"
                python main_incremental.py --exp-name $exp_name --approach olwf_asym_original --distance_metric renyi --alpha $alpha --scale_factor $scale_factor --nepochs $nepochs --num-tasks $num_tasks --datasets $dataset --results-path $results_path
                echo "Running experiment with alpha = $alpha, scale_factor = $scale_factor"
            done
        done
        python main_incremental.py --exp-name oewc_dataset_${dataset}_tasks_${num_tasks}_epochs_${nepochs} --approach oewc --nepochs $nepochs --num-tasks $num_tasks --datasets $dataset --results-path $results_path
        python main_incremental.py --exp-name icarl_exampler_dataset_${dataset}_tasks_${num_tasks}_epochs_${nepochs} --approach icarl --num-exemplars 200 --lamb 1 --nepochs $nepochs --num-tasks $num_tasks --datasets $dataset --results-path $results_path
        python main_incremental.py --exp-name finetuning_dataset_${dataset}_tasks_${num_tasks}_epochs_${nepochs} --approach finetuning --nepochs $nepochs --num-tasks $num_tasks --datasets $dataset --results-path $results_path
        python main_incremental.py --exp-name mas_dataset_${dataset}_tasks_${num_tasks}_epochs_${nepochs} --approach mas --nepochs $nepochs --num-tasks $num_tasks --datasets $dataset --results-path $results_path
        python main_incremental.py --exp-name r_walk_noexampler_dataset_${dataset}_tasks_${num_tasks}_epochs_${nepochs} --approach r_walk --nepochs $nepochs --num-tasks $num_tasks --datasets $dataset --results-path $results_path
        # Rest of your script...
    done
done
