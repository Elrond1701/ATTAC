#20230806
nepochs=10  # Set the value of nepochs here
num_exemplars=200
# Generate 33 random numbers in each of the specified ranges
alpha_values1=$(awk -v min=0.1 -v max=0.5 'BEGIN{srand(); for(i=1;i<=5;i++) print min+rand()*(max-min)}')
alpha_values2=$(awk -v min=0.5 -v max=0.8 'BEGIN{srand(); for(i=1;i<=5;i++) print min+rand()*(max-min)}')
alpha_values3=$(awk -v min=0.8 -v max=1.2 'BEGIN{srand(); for(i=1;i<=5;i++) print min+rand()*(max-min)}')

# Combine all alpha values into one array
alpha_values="$alpha_values1 $alpha_values2 $alpha_values3"
for dataset in food101 
do
    for num_tasks in 5 10 15 20
    do
        results_path="../ViT_results/test_VIT_${dataset}_${num_tasks}"    
        # for alpha in 0.85 0.01 0.06
        # do
        #     for scale_factor in 1.0
        #     do
        #     #自己方法无样本回放
        #         exp_name="ViTrenyi_${alpha}_scale_${scale_factor}_dataset_${dataset}_tasks_${num_tasks}_epochs_${nepochs}"
        #         echo "Running experiment with alpha = $alpha, scale_factor = $scale_factor"
        #         python main_incremental.py --exp-name $exp_name --approach olwf_asym_original --distance_metric renyi --alpha $alpha --scale_factor $scale_factor --nepochs $nepochs --num-tasks $num_tasks --datasets $dataset --results-path $results_path 
        #         echo "Running experiment with alpha = $alpha, scale_factor = $scale_factor"
                
        #     #自己方法有样本回放
        #         for exemplar_selection in clustering agglomerative distance
        #         do
        #             exp_name="ViT+Replay_${alpha}_scale_${scale_factor}_dataset_${dataset}_tasks_${num_tasks}_epochs_${nepochs}_${exemplar_selection}"
        #             echo "Running experiment with alpha = $alpha, scale_factor = $scale_factor, exemplar_selection = $exemplar_selection"
        #             python main_incremental.py --exp-name $exp_name --approach olwf_asym_original --distance_metric renyi --alpha $alpha --scale_factor $scale_factor --nepochs $nepochs --num-tasks $num_tasks --datasets $dataset --results-path $results_path --num-exemplars $num_exemplars --exemplar-selection $exemplar_selection
        #             echo "Finished experiment with alpha = $alpha, scale_factor = $scale_factor, exemplar_selection = $exemplar_selection"
        #         done
        #     done
        # done

#基线方法：
        python main_incremental.py --exp-name finetuning_${dataset}_tasks_${num_tasks}_epochs_${nepochs} --approach finetuning --nepochs $nepochs --num-tasks $num_tasks --datasets $dataset --results-path $results_path
        python main_incremental.py --exp-name freezing_${dataset}_tasks_${num_tasks}_epochs_${nepochs} --approach freezing --nepochs $nepochs --num-tasks $num_tasks --datasets $dataset --results-path $results_path
        python main_incremental.py --exp-name joint_${dataset}_tasks_${num_tasks}_epochs_${nepochs} --approach joint --nepochs $nepochs --num-tasks 1 --datasets $dataset --results-path $results_path
#有样本回放方法
        python main_incremental.py --exp-name bic --approach bic --num-exemplars 200 --exemplar-selection random --nepochs $nepochs --num-tasks $num_tasks --datasets $dataset --results-path $results_path 
        python main_incremental.py --exp-name il2m --approach il2m --num-exemplars 200 --exemplar-selection random --nepochs $nepochs --num-tasks $num_tasks --datasets $dataset --results-path $results_path
        python main_incremental.py --exp-name icarl --approach icarl --num-exemplars 200 --lamb 1 --nepochs $nepochs --num-tasks $num_tasks --datasets $dataset --results-path $results_path               
        python main_incremental.py --exp-name eeil_${dataset}_tasks_${num_tasks}_epochs_${nepochs} --approach eeil --nepochs $nepochs --num-tasks $num_tasks --datasets $dataset --results-path $results_path --num-exemplars 200 --exemplar-selection random
        
无样本回放方法：        
        python main_incremental.py --exp-name dmc_${dataset}_tasks_${num_tasks}_epochs_${nepochs} --approach dmc --nepochs $nepochs --num-tasks $num_tasks --datasets $dataset --results-path $results_path
        python main_incremental.py --exp-name r_walk_${dataset}_tasks_${num_tasks}_epochs_${nepochs} --approach r_walk --nepochs $nepochs --num-tasks $num_tasks --datasets $dataset --results-path $results_path
        python main_incremental.py --exp-name path_integral_${dataset}_tasks_${num_tasks}_epochs_${nepochs} --approach path_integral --nepochs $nepochs --num-tasks $num_tasks --datasets $dataset --results-path $results_path
        python main_incremental.py --exp-name mas_${dataset}_tasks_${num_tasks}_epochs_${nepochs} --approach mas --nepochs $nepochs --num-tasks $num_tasks --datasets $dataset --results-path $results_path
        python main_incremental.py --exp-name lucir_${dataset}_tasks_${num_tasks}_epochs_${nepochs} --approach lucir --nepochs $nepochs --num-tasks $num_tasks --datasets $dataset --results-path $results_path
        python main_incremental.py --exp-name LwF${dataset}_tasks_${num_tasks}_epochs_${nepochs} --approach lwf --nepochs $nepochs --num-tasks $num_tasks --datasets $dataset --results-path $results_path
        python main_incremental.py --exp-name EWC_${dataset}_tasks_${num_tasks}_epochs_${nepochs} --approach oewc --nepochs $nepochs --num-tasks $num_tasks --datasets $dataset --results-path $results_path


        
    done
done
