
%SOG (Sum of Gaussians)

%MDP (Markov Decision Process)
data = {};
data{1} = csvimport('mdp_1_rewards.csv');
data{2} = csvimport('mdp_2_rewards.csv');
data{3} = csvimport('mdp_10_rewards.csv');

%MED (Median filter)

addpath('./sdf')
addpath('./export_fig')


%RMSE plot
names = data{1}(2:end,1)
rmse = cell2mat(data{1}(2:end,2))'
bar(rmse)
set(gca,'XTickLabel', names)
title('RMSE')

sdf('10_701') %apply a style (if available)
export_fig('test.pdf')