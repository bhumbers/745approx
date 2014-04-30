
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
titles = {'Average Test Set Absolute Error', 'Average Test Set Gradient Error', 'Average Approximator Training/Compile Time', 'Execution Time Relative to Original', 'Instructions Executed Relative to Original'};
ylabels = {'RMS Error', 'RMS Error of Gradient', 'Normalized Training Time', 'Normalized Run Time', 'Normalized Instructions Executed'};
result_codes = {'rmse', 'grad_rmse', 'train_time', 'run_time', 'instructions'};

for result_idx = 1:5
    figure;
    
    i = result_idx + 1;
    lin_vals = cell2mat(data_lin(2:end,i))';
    sog_vals = cell2mat(data_sog(2:end,i))';
    mdp_vals = cell2mat(data_mdp(2:end,i))';
    med_vals = cell2mat(data_med(2:end,i))';
    vals = [lin_vals; sog_vals; mdp_vals; med_vals];
    
    %Normalize training time, runtime, and call counts relative to original
    if result_idx >= 3
        vals = bsxfun(@rdivide, vals, vals(:,1));
    end

    %Value plot
    handle = bar(vals, 'BarLayout', 'grouped');
    legend(names, 'location', 'NorthWest');
    ylabel(ylabels(result_idx));
    xlabel('Problem Type');
    set(gca,'XTickLabel', group_labels);
    title(titles(result_idx));
    filename = ['results_', result_codes{result_idx}, '.pdf']
    set(gcf,'color','w');
    grid off
    set(gcf, 'Position', [100, 100, 1000, 300]);
    set(findall(gcf,'type','text'),'fontSize',14,'fontWeight','bold')
    %Use log scale for train time, runtime, and call counts
    if result_idx >= 3
        set(gca,'YScale','log')
        %Use nicer y axis labels
        yticks = get(gca, 'YTick');
        ytickStr = []
        for ytick = yticks
            if (ytick < 0.1)
                ytickStr = [ytickStr, sprintf('%1.3f|',ytick)];
            elseif (ytick < 1)
                ytickStr = [ytickStr, sprintf('%1.2f|',ytick)];
            else
               ytickStr = [ytickStr, sprintf('%1.0f|',ytick)];
            end
        end
        set(gca,'YTickLabel',ytickStr)
    end
    
    box off
    
    export_fig(filename);
    
    result_idx = result_idx + 1;
end

%sdf('10_701'); %apply a style (if available... which is not on most machines other than Ben's)
