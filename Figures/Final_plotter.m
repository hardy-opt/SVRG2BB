%function Final_plotter(darg)
close all;
clc;
%pwd;%%%  LR       LR        LR       SVM      SVM    SVM
data = {'Adult','Covtype','Gisette','Mnist','Ijcnn','W8a'};
darg = char(data(4));
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
%fname1={'SVRG','SVRG-BB','SVRG-2BB','SVRG-2D','SVRG-2BBS-ed','SVRG-2BBS-e1'};%'SVRG-2BBS-ec'
fname1={'SVRG    ','SVRG-BB ','SVRG-2BB','SVRG-2D ','SVRG-2BBS-M1 ','SVRG-2BBS-M3 '};
%fname1={'SVRG    ','SVRG-2BB','SVRG-2D '};
l = length(fname1);
method = {'svrg','svrg_bb','svrg_2bb','svrg_2d','svrg_2bbs_eta_one','svrg_2bbs_eta_decay_m1','LBFGS'};%'svrg_2bbs_eta_constant'
%method = {'svrg','svrg_2bb','svrg_2d','LBFGS'};%'svrg_2bbs_eta_constant'

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
marker = ['p','+','s','*','^','o','>','<','_'];
col = [0 0 0;
       0 0 1;
       1 0 0;
       %0.466 0.674 0.188;
       0.290 0.694 0.025;
       0.9290 0.6940 0.5250;
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
    
    figure(f5);
    %title('Train cost');
    xlabel('Time (sec)','Fontsize',fnt)
    ylabel('Opt. gap','Fontsize',fnt)
    Gr1 = gca;
    set(gca,'Fontsize',fnt);
    ylim(cost1);
    Gr1.YScale = 'log';
    legend(fname1,'Location','NorthOutside','Orientation','horizontal','Box','on');
    %saveas(gcf, sprintf('%s-%.1e-Opt_Time.eps',darg,reg) , 'epsc' )
    hold off
       
      
    figure(f6);
    %title('\lambda  = 10^{-3}');
    xlabel(' Epoch','Fontsize',fnt)
    %ylim(cost);
    ylabel('Opt. gap','Fontsize',fnt)
    Gr1 = gca;
    ylim(cost1);
    set(gca,'Fontsize',fnt);
    Gr1.YScale = 'log';
    %legend(fname1,'Location','NorthOutside','Orientation','horizontal','Box','on');
   % saveas(gcf, sprintf('%s-%.1e-Opt_Epoch.eps',darg,reg) , 'epsc' )
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
    figure(f8);
    %title('Train cost');
    xlabel(' Epoch','Fontsize',fnt)
    ylim(var);
    ylabel('Variance','Fontsize',fnt)
    Gr1 = gca;
    set(gca,'Fontsize',fnt);
    Gr1.YScale = 'log';
    %legend(fname1,'Location','NorthOutside','Orientation','horizontal','Box','on');
   % saveas(gcf, sprintf('%s-%.1e-Var_Epoch.eps',darg,reg) , 'epsc' )
    hold off
    
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

% figure;
% for j = 1:l
%     %mean and standart daviation of optimal cost
%     
%     g3 = errorbar(g{j}.epoch,g{j}.opt_mean,g{j}.opt_std,'Marker',marker(j),'MarkerSize',msize,'LineWidth',lsize); hold('on');
%     
%     if j==l
%     %ylim(cost);
%     title('Optcost/Epoch');
%     xlabel('Epoch','Fontsize',fnt)
%     ylabel('Optimal cost','Fontsize',fnt)
%     Gr1 = gca;
%     set(gca,'Fontsize',fnt);
%     Gr1.YScale = 'log';
%     legend(fname1,'Location','NorthOutside','Orientation','horizontal','Box','on');
%     hold off
%     end
%     
%     
%     %mean and standart daviation of validation cost
% %     g{j}.vcost_mean = mean(d{j}.S1.vcost(1:end,:),2);
% %     g{j}.vcost_std = std(d{j}.S1.vcost(1:end,:),[],2);
%         
%     
%     %mean and standart daviation of Train ACC
% %     g{j}.train_ac_mean = mean(d{j}.S1.train_ac(1:end,:),2);
% %     g{j}.train_ac_std = std(d{j}.S1.train_ac(1:end,:),[],2);
%     
%     %mean and standart daviation of Train ACC
% %     g{j}.val_ac_mean = mean(d{j}.S1.val_ac(1:end,:),2);
% %     g{j}.val_ac_std = std(d{j}.S1.val_ac(1:end,:),[],2);
%     
%     %Lacotte style
%     %g{j}.Lacotte_style = (g{j}.cost_mean - f_opt + epsilon)/(1+f_opt);
%     
% 
% end



% for
% 
% e=1;
%  c = 1e-0;
% c1 = (mean(d.S1.ocost(1:end,:),2) - f_opt + epsilon)/(1+f_opt1)*c;
% c2 = (mean(d1.S1.ocost(1:end,:),2) - f_opt + epsilon)/(1+f_opt1)*c;
% c3 = (mean(d2.S1.ocost(1:end,:),2) - f_opt + epsilon)/(1+f_opt1)*c;
% c4 = (mean(d4.S1.ocost(1:end,:),2) - f_opt + epsilon)/(1+f_opt1)*c;
% c5 = (mean(d5.S1.ocost(1:end,:),2) - f_opt + epsilon)/(1+f_opt1)*c;
% c6 = (mean(d6.S1.ocost(1:end,:),2) - f_opt + epsilon)/(1+f_opt1)*c;
% 
% 
% c1 = (mean(d.S1.ocost(1:end,:),2) - f_opt + epsilon)/(1+f_opt1)*c;
% c2 = (mean(d1.S1.ocost(1:end,:),2) - f_opt + epsilon)/(1+f_opt1)*c;
% c3 = (mean(d2.S1.ocost(1:end,:),2) - f_opt + epsilon)/(1+f_opt1)*c;
% c4 = (mean(d4.S1.ocost(1:end,:),2) - f_opt + epsilon)/(1+f_opt1)*c;
% c5 = (mean(d5.S1.ocost(1:end,:),2) - f_opt + epsilon)/(1+f_opt1)*c;
% c6 = (mean(d6.S1.ocost(1:end,:),2) - f_opt + epsilon)/(1+f_opt1)*c;
% 
% % s1 = isbar*std(d.S1.ocost(1:end,:),[],2)/(1+f_opt1);
% % s2 = isbar*std(d1.S1.ocost(1:end,:),[],2)/(1+f_opt1);
% % s3 = isbar*std(d2.S1.ocost(1:end,:),[],2)/(1+f_opt1);
% % s4 = isbar*std(d4.S1.ocost(1:end,:),[],2)/(1+f_opt1);
% % % s5 = isbar*std(d5.S1.ocost(1:end,:),[],2)/(1+f_opt1);
% % disp('svrg');
% % [d.S1.ocost c1]
% % disp('svrg-bb');
% % [d1.S1.ocost c2]
% % disp('svrg-2bb');
% % [d2.S1.ocost c3]
% % disp('svrg-dh');
% % [d4.S1.ocost c4]
% % disp('svrg-new step');
% %[d5.S1.ocost c5]
%  figure; hold on;
% plot(d.S1.epoch(1:endd,:),(c1),'r--*','MarkerSize',msize,'LineWidth',lsize);hold on;
% plot(d1.S1.epoch(1:endd,:),(c2),'b--o','MarkerSize',msize,'LineWidth',lsize);hold on;
% plot(d2.S1.epoch(1:endd,:),(c3),'g-s','MarkerSize',msize,'LineWidth',lsize);hold on;
% plot(d3.S1.epoch(2:endd,:),(d3.S1.variance(2:endd,:)),'k:>','MarkerSize',msize,'LineWidth',lsize);hold on;
% plot(d4.S1.epoch(1:endd,:),(c4),'m--o','MarkerSize',msize,'LineWidth',lsize);hold on;
% plot(d5.S1.epoch(1:endd,:),(c5(1:endd)),'k:>','MarkerSize',msize,'LineWidth',lsize);hold on;
% plot(d6.S1.epoch(1:endd,:),(c6(1:endd)),'k--o','MarkerSize',msize,'LineWidth',lsize);hold on;
% xlabel('Epoch','Fontsize',lgft)
% ylabel('Original cost','Fontsize',lgft)
% ylim(cost);
% xlim(epoch);
% G = gca;
% set(gca,'Fontsize',fnt);
% G.YScale = 'log';
% lgh=legend(fname1,'location','northeast','NumColumns',5);
% % %saveas (gcf, 'Variance_Epoch' , 'epsc' )
% 
% 
% e=1;
% 
% figure;hold on;
% %title("Optimal gap Vs. Epoch");
% errorbar(d.S1.epoch(e:endd,:),((mean(abs(d.S1.ocost(e:endd,:)),2))-f_opt+epsilon)/(1+f_opt),isbar*(std(d.S1.ocost(e:endd,:),[],2))/(1+f_opt1),'r--*','MarkerSize',msize,'LineWidth',lsize);hold on;
% errorbar(d1.S1.epoch(e:endd,:),((mean(abs(d1.S1.ocost(e:endd,:)),2))-f_opt+epsilon)/(1+f_opt),isbar*(std(d1.S1.ocost(e:endd,:),[],2))/(1+f_opt1),'b--o','MarkerSize',msize,'LineWidth',lsize);hold on;
% errorbar(d2.S1.epoch(e:endd,:),((mean(abs(d2.S1.ocost(e:endd,:)),2))-f_opt+epsilon)/(1+f_opt),isbar*(std(d2.S1.ocost(e:endd,:),[],2))/(1+f_opt1),'g-s','MarkerSize',msize,'LineWidth',lsize);hold on;
% %errorbar(d3.S1.epoch(e:endd,:),abs((mean(abs(d3.S1.ocost(e:endd,:)),2))-f_opt+epsilon)/(1+f_opt),isbar*(std(d3.S1.ocost(e:endd,:),[],2)),'k:>','MarkerSize',msize,'LineWidth',lsize);hold on;
% errorbar(d4.S1.epoch(e:endd,:),((mean(abs(d4.S1.ocost(e:endd,:)),2))-f_opt+epsilon)/(1+f_opt),isbar*(std(d4.S1.ocost(e:endd,:),[],2))/(1+f_opt1),'m--o','MarkerSize',msize,'LineWidth',lsize);hold on;
% errorbar(d5.S1.epoch(e:endd,:),((mean(abs(d5.S1.ocost(e:endd,:)),2))-f_opt+epsilon)/(1+f_opt),isbar*(std(d5.S1.ocost(e:endd,:),[],2))/(1+f_opt1),'k:>','MarkerSize',msize,'LineWidth',lsize);hold on;
% xlabel('Epoch','Fontsize',lgft)
% ylabel('relative error','Fontsize',lgft)
% ylim(cost);
% xlim(epoch);
% legend(fname1,'location','northeast','NumColumns',1);
% G = gca;
% G.YScale = 'log';
% set(gca,'Fontsize',fnt);
% %legendmarkeradjust(8,12)
% saveas (gcf, 'opt_gap_Epoch' , 'epsc' )
% 
% 
% figure;hold on;
% %title("Optimal gap Vs. time");
% e=1;
% errorbar(d.S1.otime(e:endd,:),((mean(abs(d.S1.ocost(e:endd,:)),2))-f_opt),isbar*(std(d.S1.ocost(e:endd,:),[],2))/(1+f_opt),'r--*','MarkerSize',msize,'LineWidth',lsize);hold on;
% errorbar(d1.S1.otime(e:endd,:),((mean(abs(d1.S1.ocost(e:endd,:)),2))-f_opt),isbar*(std(d1.S1.ocost(e:endd,:),[],2))/(1+f_opt),'b--o','MarkerSize',msize,'LineWidth',lsize);hold on;
% errorbar(d2.S1.otime(e:endd,:),((mean(abs(d2.S1.ocost(e:endd,:)),2))-f_opt),isbar*(std(d2.S1.ocost(e:endd,:),[],2))/(1+f_opt),'g-s','MarkerSize',msize,'LineWidth',lsize);hold on;
% %errorbar(d3.S1.otime(e:endd,:),abs((mean(abs(d3.S1.ocost(e:endd,:)),2))-f_opt),isbar*(std(d3.S1.ocost(e:endd,:),[],2))/(1+f_opt),'k:>','MarkerSize',msize,'LineWidth',lsize);hold on;
% errorbar(d4.S1.otime(e:endd,:),((mean(abs(d4.S1.ocost(e:endd,:)),2))-f_opt),isbar*(std(d4.S1.ocost(e:endd,:),[],2))/(1+f_opt),'m--o','MarkerSize',msize,'LineWidth',lsize);hold on;
% errorbar(mean(d5.S1.otime(e:endd,:),2),((mean(abs(d5.S1.ocost(e:endd,:)),2))-f_opt),(isbar*std(d5.S1.ocost(e:endd,:),[],2))/(1+f_opt),'k:>','MarkerSize',msize,'LineWidth',lsize);hold on;
% xlabel('time','Fontsize',lgft)
% ylabel('relative error','Fontsize',lgft)
% ylim(cost);
% xlim(time);
% % xticks(20:40:340);
% G = gca;
% G.YScale = 'log';
% set(gca,'Fontsize',fnt);
% legend(fname1,'location','northeast');
% %legendmarkeradjust(8,12)
% saveas (gcf, 'opt_gap_time' , 'epsc' )
% 
% %endd=20;
% fig=figure;hold on;
% %title("Variance Vs. Epoch");
% plot(d.S1.epoch(2:endd,:),(d.S1.variance(2:endd,:)),'r--*','MarkerSize',msize,'LineWidth',lsize);hold on;
% plot(d1.S1.epoch(2:endd,:),(d1.S1.variance(2:endd,:)),'b--o','MarkerSize',msize,'LineWidth',lsize);hold on;
% plot(d2.S1.epoch(2:endd,:),(d2.S1.variance(2:endd,:)),'g-s','MarkerSize',msize,'LineWidth',lsize);hold on;
% %plot(d3.S1.epoch(2:endd,:),(d3.S1.variance(2:endd,:)),'k:>','MarkerSize',msize,'LineWidth',lsize);hold on;
% plot(d4.S1.epoch(2:endd,:),(d4.S1.variance(2:endd,:)),'m--o','MarkerSize',msize,'LineWidth',lsize);hold on;
% plot(d5.S1.epoch(2:endd,:),mean(d5.S1.variance(2:endd,:),2),'k:>','MarkerSize',msize,'LineWidth',lsize);hold on;
% xlabel('Epoch','Fontsize',lgft)
% ylabel('Variance','Fontsize',lgft)
% ylim(cost);
% xlim(epoch);
% G = gca;
% set(gca,'Fontsize',fnt);
% G.YScale = 'log';
% lgh=legend(fname1,'location','northeast','NumColumns',5);
% saveas (gcf, 'Variance_Epoch' , 'epsc' )
% %saveLegendToImage(fig, lgh, 'legend3', 'epsc');
% 
% 
% figure;hold on;
% %title("Variance Vs. Time");
% plot(d.S1.otime(2:endd,:),(d.S1.variance(2:endd,:)),'r--*','MarkerSize',msize,'LineWidth',lsize);hold on;
% plot(d1.S1.otime(2:endd,:),(d1.S1.variance(2:endd,:)),'b--o','MarkerSize',msize,'LineWidth',lsize);hold on;
% plot(d2.S1.otime(2:endd,:),(d2.S1.variance(2:endd,:)),'g-s','MarkerSize',msize,'LineWidth',lsize);hold on;
% %plot(d3.S1.otime(2:endd,:),(d3.S1.variance(2:endd,:)),'k:>','MarkerSize',msize,'LineWidth',lsize);hold on;
% plot(d4.S1.otime(2:endd,:),(d4.S1.variance(2:endd,:)),'m--o','MarkerSize',msize,'LineWidth',lsize);hold on;
% plot(mean(d5.S1.otime(2:endd,:),2),mean(d5.S1.variance(2:endd,:),2),'k:>','MarkerSize',msize,'LineWidth',lsize);hold on;
% xlabel('Time','Fontsize',lgft)
% ylabel('Variance','Fontsize',lgft)
% ylim(cost);
% xlim(time);
% % xticks(20:40:340);
% legend(fname1,'location','northeast');
% G = gca;
% G.YScale = 'log';
% set(gca,'Fontsize',fnt);
% saveas (gcf, 'Variance_Time' , 'epsc' )


%  figure;hold on;
% % title("Train Acc Vs. Epoch");
%  errorbar(d.S1.epoch(e:endd,:),mean(d.S1.train_ac(e:endd,:),2),isbar*std(d.S1.train_ac(e:endd,:),[],2),'r-*','MarkerSize',msize,'LineWidth',lsize);hold on;
%  errorbar(d.S1.epoch(2:endd,:),mean(d1.S1.train_ac(2:endd,:),2),isbar*std(d1.S1.train_ac(2:endd,:),[],2),'b--*','MarkerSize',msize,'LineWidth',lsize);hold on;
%  errorbar(d.S1.epoch(e:endd,:),mean(d2.S1.train_ac(e:endd,:),2),isbar*std(d2.S1.train_ac(e:endd,:),[],2),'g-s','MarkerSize',msize,'LineWidth',lsize);hold on;
%  errorbar(d.S1.epoch(2:endd,:),mean(d3.S1.train_ac(2:endd,:),2),isbar*std(d3.S1.train_ac(2:endd,:),[],2),'k--*','MarkerSize',msize,'LineWidth',lsize);hold on;
%  errorbar(d.S1.epoch(e:endd,:),mean(d4.S1.train_ac(e:endd,:),2),isbar*std(d4.S1.train_ac(e:endd,:),[],2),'m--o','MarkerSize',msize,'LineWidth',lsize);hold on;
%  errorbar(d.S1.epoch(e:endd,:),mean(d5.S1.train_ac(e:endd,:),2),isbar*std(d5.S1.train_ac(e:endd,:),[],2),'y--o','MarkerSize',msize,'LineWidth',lsize);hold on;
% 
%  xlabel('Epoch','FontSize',lgft)
%  ylabel('Train acc','FontSize',lgft)
%  ylim(acc);
%  xlim(epoch);
%  %legend(fname,'location','southeast');
% % G = gca;
% % G.YScale = 'log';
%  set(gca,'Fontsize',fnt);
% % saveas (gcf, 'Train_Acc_Epoch' , 'epsc' )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  figure;hold on;
% % title("Val Acc Vs. Epoch");
%  errorbar(d.S1.epoch(e:endd,:),mean(d.S1.val_ac(e:endd,:),2),isbar*std(d.S1.val_ac(e:endd,:),[],2),'r-*','MarkerSize',msize,'LineWidth',lsize); hold on;
%  errorbar(d.S1.epoch(2:endd,:),mean(d1.S1.val_ac(2:endd,:),2),isbar*std(d1.S1.val_ac(2:endd,:),[],2),'b-*','MarkerSize',msize,'LineWidth',lsize);
%  errorbar(d.S1.epoch(e:endd,:),mean(d2.S1.val_ac(e:endd,:),2),isbar*std(d2.S1.val_ac(e:endd,:),[],2),'g-s','MarkerSize',msize,'LineWidth',lsize); hold on;
%  errorbar(d.S1.epoch(2:endd,:),mean(d3.S1.val_ac(2:endd,:),2),isbar*std(d3.S1.val_ac(2:endd,:),[],2),'k-*','MarkerSize',msize,'LineWidth',lsize);
%  errorbar(d.S1.epoch(e:endd,:),mean(d4.S1.val_ac(e:endd,:),2),isbar*std(d4.S1.val_ac(e:endd,:),[],2),'m--o','MarkerSize',msize,'LineWidth',lsize); hold on;
%  errorbar(d.S1.epoch(e:endd,:),mean(d5.S1.val_ac(e:endd,:),2),isbar*std(d5.S1.val_ac(e:endd,:),[],2),'k:>','MarkerSize',msize,'LineWidth',lsize); hold on;
%  xlabel('Epoch','FontSize',lgft)
%  ylabel('Test acc','FontSize',lgft)
%   ylim(accc);
%  xlim(epoch);
%  %legend(fname,'location','southeast');
% % %G = gca;
% % %G.YScale = 'log';
%  set(gca,'Fontsize',fnt);
% % saveas (gcf, 'Val_Acc_Epoch' , 'epsc' )

%end