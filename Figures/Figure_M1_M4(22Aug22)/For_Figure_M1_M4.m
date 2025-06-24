%This file is to get figures on adult covtype gisette of Figure 1
close all;
clc;
%pwd;%%%  LR       LR        LR       SVM      SVM    SVM
data = {'Adult','Covtype','Gisette','Mnist','Ijcnn','W8a'};
darg = char(data(2));
%fprintf('\nData is %s:\n',darg);


pathh=strcat('SVRG_BB/Results_2022/',darg,'/');
find_par=1; %0 means accuracy and 1 means cost
reg = 1e-3;


if reg == 1e-2
    re = -2;
elseif reg == 1e-3
    re = -3;
else
    re = -4;
end
epsilon = 1*5e-15;
fnt=22; %Font
lgft=23; %
msize=12; %Marker size
lsize=2; % Legend size 
isbar=1;
cost1 = 'auto';
%time=[0,2000];
%time = 'auto';
var = 'auto';
acc = [0.75,1.05];

fname1={'M1','M2','M3'};
l = length(fname1);

method={'svrg_2bbs_eta_one','svrg_2bbs_eta_decay','svrg_2bbs_eta_decay_m1','LBFGS'};%'svrg_2bbs_eta_constant','svrg_2bbs_eta_constant_m1','svrg_2bbs_eta_one_m1'};
%method = {1 M1| M2 2| M3 5|  - 3 6 4} % From last 6 method of
%SVRG-2BBS, 1st is M1, 2nd is M2, and 5th is M3. Others have been discarded.

Act_cost=[]; % Actual cost
Act_dv=[]; %Actual daviation 
best = zeros(l,14);
bestL = zeros(1,4);
d = [];
fprintf(' %s-Best Parameters for %.1e  \n    |Method    | Step size | \n',darg,reg)
for i = 1:l
    best(i,:) = other_best(strcat(pathh,char(method(i))), find_par, reg);
     Act_cost = [Act_cost best(i,13)];
%     Act_dv = [Act_dv best(i,7)];
    d{i} = load(strcat(pathh,char(method(i)),sprintf('_%.1e_R_%.1e.mat',best(i,11),best(i,12))));
    ep = length(d{1}.S1.epoch);
    fprintf('    |%s  |   %.1e | \n', char(fname1(i)), best(i,11));
end

bestL(1,:) = other_best_LBFGS(strcat(pathh,char(method(end))), find_par, reg);
F = load(strcat(pathh,char(method(end)),sprintf('_%.1e_R_%.1e.mat',bestL(1,3),bestL(1,4))));


f_opt = F.LBFGS.cost(end);
cost=[0.1*f_opt,1];
fprintf('\n Optimal cost =  %.18e \n',f_opt);
f_opt1 = f_opt;
ind = find(Act_cost==f_opt);
% fprintf('\n Best Method = M%d-%s \n',ind,char(method(ind)));
g = [];
marker = ['^','>','o','d','<','_'];
col = [0.9290 0.6940 0.5250;
       0.45 0.64 0.28;
  %     0.90 0.1 0.650;
       0.494 0.184 0.556];
for j = 1:l
    % X axis properties
    g{j}.time = d{j}.S1.otime(1:end);
    g{j}.epoch = d{j}.S1.epoch(1:end);
    
    
    % Variance
    g{j}.variance = d{j}.S1.variance(1:end);
    
    
    % Train cost
    g{j}.cost_mean = mean(d{j}.S1.ocost(1:end,:),2);
    g{j}.cost_std = isbar*std(d{j}.S1.ocost(1:end,:),[],2);
    g{j}.cost = abs(g{j}.cost_mean)/f_opt;% - f_opt + epsilon)/(1+f_opt);
    
    
    % Optimality gap = cost - optimal cost
%     g{j}.opt_mean = abs(mean(d{j}.S1.opt_gap(1:end,:),2)); 
%     g{j}.opt_std = isbar*std(d{j}.S1.opt_gap(1:end,:),[],2);
%     g{j}.optgap = abs(g{j}.opt_mean);% - f_opt + epsilon)/(1+f_opt);
      g{j}.optgap = abs(g{j}.cost_mean - f_opt + epsilon)/(1+f_opt);

    
    % Validation cost
    g{j}.vcost_mean = mean(d{j}.S1.vcost(1:end,:),2);
    g{j}.vcost_std = (isbar*std(d{j}.S1.vcost(1:end,:),[],2));
    g{j}.vcost = abs(g{j}.vcost_mean);% - f_opt + epsilon);%/(1+f_opt);
    
    
    % Train accuracy
    g{j}.train_acc = mean(d{j}.S1.train_ac,2);
    
    
    % Validation accuracy 
    g{j}.val_acc = mean(d{j}.S1.val_ac,2);
end  



% f1 = figure; % cost vs time
% f2 = figure; % cost vs epoch
% f3 = figure; % vcost vs time
% f4 = figure; % vcost vs epoch
f5 = figure; % optimal cost vs time
f6 = figure; % optimal cost vs epoch
% f7 = figure; % variance vs time
 f8 = figure; % variance vs epoch
% f9 = figure; % train ac vs time
% f10 = figure; % train ac vs epoch
% f11 = figure; % val ac vs time
% f12 = figure; % val ac vs epoch

for j = 1:l


%     %g2 = errorbar(g{j}.epoch,g{j}.cost_mean,g{j}.cost_std,'Marker',marker(j),'MarkerSize',msize,'LineWidth',lsize); hold('on');
%     figure(f1)
%     plot(g{j}.time,g{j}.cost,'color',col(j,:),'Marker',marker(j),'MarkerSize',msize,'LineWidth',lsize); hold('on');
%     
%     
%     figure(f2)
%     plot(g{j}.epoch,g{j}.cost,'color',col(j,:),'Marker',marker(j),'MarkerSize',msize,'LineWidth',lsize); hold('on');
    
%     figure(f3)
%     plot(g{j}.time,g{j}.vcost,'color',col(j,:),'Marker',marker(j),'MarkerSize',msize,'LineWidth',lsize); hold('on');
    
%     figure(f4)
%     plot(g{j}.epoch,g{j}.vcost,'color',col(j,:),'Marker',marker(j),'MarkerSize',msize,'LineWidth',lsize); hold('on');
    
    figure(f5)
    plot(g{j}.time,g{j}.optgap,'color',col(j,:),'Marker',marker(j),'MarkerSize',msize,'LineWidth',lsize); hold('on');
    
    figure(f6)
    plot(g{j}.epoch,g{j}.optgap,'color',col(j,:),'Marker',marker(j),'MarkerSize',msize,'LineWidth',lsize); hold('on');
%     
%     figure(f7)
%     plot(g{j}.time,g{j}.variance,'color',col(j,:),'Marker',marker(j),'MarkerSize',msize,'LineWidth',lsize); hold('on');
%     
    figure(f8)
    plot(g{j}.epoch,g{j}.variance,'color',col(j,:),'Marker',marker(j),'MarkerSize',msize,'LineWidth',lsize); hold('on');
%     
%     
%     figure(f9)
%     plot(g{j}.time,g{j}.train_acc,'color',col(j,:),'Marker',marker(j),'MarkerSize',msize,'LineWidth',lsize); hold('on');
%     
%     
%     figure(f10)
%     plot(g{j}.epoch,g{j}.train_acc,'color',col(j,:),'Marker',marker(j),'MarkerSize',msize,'LineWidth',lsize); hold('on');
%     
%     
%     figure(f11)
%     plot(g{j}.time,g{j}.val_acc,'color',col(j,:),'Marker',marker(j),'MarkerSize',msize,'LineWidth',lsize); hold('on');
%     
%     
%     figure(f12)
%     plot(g{j}.epoch,g{j}.val_acc,'color',col(j,:),'Marker',marker(j),'MarkerSize',msize,'LineWidth',lsize); hold('on');
%     
    
    if j==l
    
%     figure(f1);
%     %title('Train cost');
%     xlabel('Time (sec)','Fontsize',fnt)
%     ylabel('Train cost','Fontsize',fnt)
%     Gr1 = gca;
%     set(gca,'Fontsize',fnt);
%     ylim(cost);
%     %Gr1.YScale = 'log';
% 
%     %legend(fname1,'Location','NorthOutside','Orientation','horizontal','Box','on');
%     hold off
%        
%       
%     figure(f2);
%     %title('Train cost');
%     xlabel(' Epoch','Fontsize',fnt)
%     ylabel('Train cost','Fontsize',fnt)
%     Gr2 = gca;
%     set(gca,'Fontsize',fnt);
%     ylim(cost);
%     Gr2.YScale = 'log';
%     %legend(fname1,'Location','NorthOutside','Orientation','horizontal','Box','on');
%     hold off
%      
%     
%     figure(f3);
%     %title('Val. cost');
%     xlabel('Time (sec)','Fontsize',fnt)
%     ylabel('Val. cost','Fontsize',fnt)
%     Gr1 = gca;
%     set(gca,'Fontsize',fnt);
%     ylim(cost);
%     Gr1.YScale = 'log';
%     %legend(fname1,'Location','NorthOutside','Orientation','horizontal','Box','on');
%     hold off
%        
%       
%     figure(f4);
%     title(strcat('\',sprintf('lambda = 10^{%d}',re)));
%     xlabel(' Epoch','Fontsize',fnt)
%     %ylim(cost);
%     ylabel('Val. cost','Fontsize',fnt)
%     Gr1 = gca;
%     set(gca,'Fontsize',fnt);
%     ylim(cost1);
%     Gr1.YScale = 'log';
%     %legend(fname1,'Location','NorthOutside','Orientation','horizontal','Box','on');
%     hold off
%     
%     
%     figure(f5);
%     %title('Train cost');
%     xlabel('Time (sec)','Fontsize',fnt)
%     ylabel('Opt. gap','Fontsize',fnt)
%     Gr1 = gca;
%     set(gca,'Fontsize',fnt);
%     ylim(cost1);
%     Gr1.YScale = 'log';
%     %legend(fname1,'Location','NorthOutside','Orientation','horizontal','Box','on');
%     saveas(gcf, sprintf('%s-%.1e-Opt_Time.eps',darg,reg) , 'epsc' )
%     hold off
%        
%       
    figure(f6);
    %title('\lambda  = 10^{-3}');
    xlabel(' Epoch','Fontsize',fnt)
    %ylim(cost);
    ylabel('Opt. gap','Fontsize',fnt)
    Gr1 = gca;
    ylim(cost1);
    set(gca,'Fontsize',fnt);
    Gr1.YScale = 'log';
    legend(fname1,'Location','NorthOutside','Orientation','horizontal','Box','on');
%     saveas(gcf, sprintf('%s-%.1e-Opt_Epoch.eps',darg,reg) , 'epsc' )
    hold off
    
    
%     figure(f7);
%     %title('Train cost');
%     xlabel('Time (sec)','Fontsize',fnt)
%     ylabel('Variance','Fontsize',fnt)
%     Gr1 = gca;
%     ylim(var);
%     set(gca,'Fontsize',fnt);
%     Gr1.YScale = 'log';
%     %legend(fname1,'Location','NorthOutside','Orientation','horizontal','Box','on');
%     hold off
%        
%       
%     figure(f8);
%     %title('Train cost');
%     xlabel(' Epoch','Fontsize',fnt)
%     ylim(var);
%     ylabel('Variance','Fontsize',fnt)
%     Gr1 = gca;
%     set(gca,'Fontsize',fnt);
%     Gr1.YScale = 'log';
%     %legend(fname1,'Location','NorthOutside','Orientation','horizontal','Box','on');
%     saveas(gcf, sprintf('%s-%.1e-Var_Epoch.eps',darg,reg) , 'epsc' )
%     hold off
%     
%     
%     
%     figure(f9);
%     %title('Train cost');
%     xlabel('Time (sec)','Fontsize',fnt)
%     ylabel('Train acc.','Fontsize',fnt)
%     Gr1 = gca;
%     ylim(acc);
%     set(gca,'Fontsize',fnt);
%     %Gr1.YScale = 'log';
%     %legend(fname1,'Location','NorthOutside','Orientation','horizontal','Box','on');
%     hold off
%        
%       
%     figure(f10);
%     %title('Train cost');
%     xlabel(' Epoch','Fontsize',fnt)
%     %ylim(cost);
%     ylabel('Train acc.','Fontsize',fnt)
%     Gr1 = gca;
%     ylim(acc);
%     set(gca,'Fontsize',fnt);
%     %Gr1.YScale = 'log';
%     %legend(fname1,'Location','NorthOutside','Orientation','horizontal','Box','on');
%     hold off
%     
%     
%     
%     figure(f11);
%     %title('Val acc');
%     xlabel('Time (sec)','Fontsize',fnt)
%     ylabel('Val. acc.','Fontsize',fnt)
%     Gr1 = gca;
%     ylim(acc);
%     set(gca,'Fontsize',fnt);
%     %Gr1.YScale = 'log';
%     %legend(fname1,'Location','NorthOutside','Orientation','horizontal','Box','on');
%     hold off
%        
%       
%     figure(f12);
%     %title('');
%     xlabel('Epoch','Fontsize',fnt)
%     ylim(acc);
%     ylabel('Val. acc.','Fontsize',fnt)
%     Gr1 = gca;
%     set(gca,'Fontsize',fnt);
%     %Gr1.YScale = 'log';
%     legend(fname1,'Location','NorthOutside','Orientation','horizontal','Box','on');
%     hold off
    end
    
    
end 
