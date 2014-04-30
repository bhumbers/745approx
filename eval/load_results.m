

%LIN (Linear)
data_lin = csvimport('lin.csv');

%SOG (Sum of Gaussians)
data_sog = csvimport('sog_1_gaussians.csv');
names = data(2:end,1)

%MDP (Markov Decision Process)
data_mdp = csvimport('mdp_1_rewards.csv');

%MED (Median filter)
data_med = csvimport('med_I20x20_K3x3.csv');

addpath('./sdf');
addpath('./export_fig');

group_labels = {'LIN', 'SOG', 'MDP', 'MED'};
titles = {'RMSE', 'Gradient Error', 'Training Time', 'Run Time', 'Calls'}

for i = 2:6
    figure;
    
    lin_rmse = cell2mat(data_lin(2:end,i))';
    sog_rmse = cell2mat(data_sog(2:end,i))';
    mdp_rmse = cell2mat(data_mdp(2:end,i))';
    med_rmse = cell2mat(data_med(2:end,i))';
    rmses = [lin_rmse; sog_rmse; mdp_rmse; med_rmse];

    %RMSE plot
    bar(rmses, 'BarLayout', 'grouped');
    legend(names);
    ylabel(titles(i-1));
    set(gca,'XTickLabel', group_labels);
    title(titles(i-1));
    filename = ['test', int2str(i), '.pdf']
    % TODO: move legend left
    % TODO: white bg
    % TODO: font
    % TODO: remove right and top edges
    % TODO: log scale for time and calls
    export_fig(filename);
end

%sdf('10_701'); %apply a style (if available... which is not on most machines other than Ben's)
