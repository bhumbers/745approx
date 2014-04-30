
close all

%LIN (Linear)
data_lin = csvimport('lin.csv');

%SOG (Sum of Gaussians)
data_sog = csvimport('sog_1_gaussians.csv');
names = data_sog(2:end,1)

%MDP (Markov Decision Process)
data_mdp = csvimport('mdp_1_rewards.csv');

%MED (Median filter)
data_med = csvimport('med_I20x20_K3x3.csv');

addpath('./sdf');
addpath('./export_fig');

group_labels = {'LIN', 'SOG', 'MDP', 'MED'};
titles = {'RMSE', 'Gradient Error', 'Training Time', 'Run Time', 'Calls'}
result_codes = {'rmse', 'grad_rmse', 'run_time', 'train_time', 'instructions'}

result_idx = 1
for i = 2:6
    figure;
    
    lin_rmse = cell2mat(data_lin(2:end,i))';
    sog_rmse = cell2mat(data_sog(2:end,i))';
    mdp_rmse = cell2mat(data_mdp(2:end,i))';
    med_rmse = cell2mat(data_med(2:end,i))';
    rmses = [lin_rmse; sog_rmse; mdp_rmse; med_rmse];

    %RMSE plot
    bar(rmses, 'BarLayout', 'grouped');
    legend(names, 'location', 'NorthWest');
    ylabel(titles(i-1));
    set(gca,'XTickLabel', group_labels);
    title(titles(i-1));
    filename = ['results_', result_codes{result_idx}, '.pdf']
    set(gcf,'color','w');
    box off
    set(gcf, 'Position', [100, 100, 1000, 300]);
    set(findall(gcf,'type','text'),'fontSize',14,'fontWeight','bold')
    %Use log scale for runtime, train time, and call counts
    if i >= 4
        
    end
    
    export_fig(filename);
    
    result_idx = result_idx + 1
end

%sdf('10_701'); %apply a style (if available... which is not on most machines other than Ben's)
