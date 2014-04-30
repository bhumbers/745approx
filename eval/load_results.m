


%SOG (Sum of Gaussians)
data = {};
data{1} = csvimport('sog_1_gaussians.csv');
% data{2} = csvimport('sog_2_gaussians.csv');
% data{3} = csvimport('sog_10_gaussians.csv');
names = data{1}(3:end,1)
sog_rmse = cell2mat(data{1}(3:end,2))';

%MDP (Markov Decision Process)
data = {};
data{1} = csvimport('mdp_1_rewards.csv');
% data{2} = csvimport('mdp_2_rewards.csv');
% data{3} = csvimport('mdp_10_rewards.csv')
mdp_rmse = cell2mat(data{1}(3:end,2))';

%MED (Median filter)

addpath('./sdf');
addpath('./export_fig');

rmses = [sog_rmse; mdp_rmse];
group_labels = {'SOG', 'MDP'};

%RMSE plot
bar(rmses, 'BarLayout', 'grouped');
legend(names);
ylabel('RMSE');
set(gca,'XTickLabel', group_labels);
title('Approximation Error');

sdf('10_701'); %apply a style (if available... which is not on most machines other than Ben's)
export_fig('test.pdf');