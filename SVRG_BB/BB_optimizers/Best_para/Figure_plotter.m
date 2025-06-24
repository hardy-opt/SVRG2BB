function Figure_plotter(darg)
close all;
%clear;
clc;
pwd

if strcmp(darg, 'GISETTE') %l2-LR
    endd = 11;
elseif strcmp(darg, 'COVTYPE') %l2-LR
    endd = 13;
elseif strcmp(darg, 'W8A') % l2-LR
    endd = 21;%26;
elseif strcmp(darg, 'IJCNN') %l2-SVM
    endd = 16;
elseif strcmp(darg, 'ADULT')     % l2-svm
    endd = 31;%26;
elseif strcmp(darg, 'MNIST')  %l2-LR
    
end
pathh=strcat('SVRG_BB/Results_2022/',darg,'/');
fname1={'SVRG','SVRG-BB','SVRG-2BB','SVRG-2D','SVRG-2BBSNew'};
find_par=1; %0 means accuracy and 1 means cost

reg = 1e-3;
%endd=16;
fnt=20; %Font
lgft=23; %
msize=10; %Marker size
lsize=2; % Legend size 
isbar=1;
acc='auto';
accc='auto';
%cost=[1e-2,0.4];
cost = ['auto'];
%time=[0,2000];
time = 'auto';
%epoch=[0,20];
epoch='auto';
%%% Group 1 : SVRG , SVRG DH, SVRG-2BB --> best, best4, best2
%%% Group 2 : SVRG, SVRG-BB, SVRG-2BBS --> best, best1, best3
%fname1={'SVRG','SVRG-BB','SVRG-2BB','SVRG-2BBS','SVRG-2D'};%
Ac=[]; % Actual cost
Ad=[]; %Actual daviation 
best = other_best(strcat(pathh,'svrg'), find_par,reg); %SVRG
Ac = [Ac best(6)];
Ad = [Ad best(7)];
best1 = other_best(strcat(pathh,'svrg_bb'), find_par,reg); %SVRG-BB
Ac = [Ac best1(6)];
Ad = [Ad best1(7)];
best2 = other_best(strcat(pathh,'svrg_2bb'), find_par,reg); %SVRG-2BB
Ac = [Ac best2(6)];
Ad = [Ad best2(7)];
best3 = other_best(strcat(pathh,'svrg_2bbs_eta_decay'), find_par,reg); %SVRG-2BBS
% Ac = [Ac best3(6)];
% Ad = [Ad best3(7)];
best4 = other_best(strcat(pathh,'svrg_2d'), find_par,reg); %SVRG-2D
Ac = [Ac best4(6)];
Ad = [Ad best4(7)];
%New
best5 = other_best(strcat(pathh,'svrg_2bbs_eta_constant'), find_par,reg); %SVRG-2BBNewS
Ac = [Ac best5(13)];
Ad = [Ad best5(7)];
best6 = other_best(strcat(pathh,'svrg_2bbs_eta_one'), find_par,reg); %SVRG-2BBNewS
Ac = [Ac best5(13)];
Ad = [Ad best5(7)];

f_opt = 0*min(Ac);
f_opt1 = f_opt;
epsilon = 0;
indices = find(Ac==f_opt);
fprintf('Method = %d',indices);

Name = strcat(pathh,'svrg',sprintf('_%.1e_R_%.1e.mat',best(11),best(12)));
Name1 = strcat(pathh,'svrg_bb',sprintf('_%.1e_R_%.1e.mat',best1(11),best1(12)));
fprintf('\n \n Now our BB\n \n');
Name2 = strcat(pathh,'svrg_2bb',sprintf('_%.1e_R_%.1e.mat',best2(11),best2(12)));
Name3 = strcat(pathh,'svrg_2bbs_eta_decay',sprintf('_%.1e_R_%.1e.mat',best3(11),best3(12)));
Name4 = strcat(pathh,'svrg_2d',sprintf('_%.1e_R_%.1e.mat',best4(11),best4(12)));
Name5 = strcat(pathh,'svrg_2bbs_eta_constant',sprintf('_%.1e_R_%.1e.mat',best5(11),best5(12)));
Name6 = strcat(pathh,'svrg_2bbs_eta_one',sprintf('_%.1e_R_%.1e.mat',best5(11),best5(12)));
Name
d=load(Name);   % 1 SVRG
d1=load(Name1); % 2 SVRG-BB
d2=load(Name2); % 3 SVRG-2BB
d3=load(Name3); % 4 SVRG-2BBS-eta-decay
d4=load(Name4); % 5 SVRG-2D
d5=load(Name5); % 6 SVRG-2BBS-eta-constant
d6=load(Name6); % 7 SVRG-2BB-eta-one

e=1;
 c = 1e-0;
c1 = (mean(d.S1.ocost(1:end,:),2) - f_opt + epsilon)/(1+f_opt1)*c;
c2 = (mean(d1.S1.ocost(1:end,:),2) - f_opt + epsilon)/(1+f_opt1)*c;
c3 = (mean(d2.S1.ocost(1:end,:),2) - f_opt + epsilon)/(1+f_opt1)*c;
c4 = (mean(d4.S1.ocost(1:end,:),2) - f_opt + epsilon)/(1+f_opt1)*c;
c5 = (mean(d5.S1.ocost(1:end,:),2) - f_opt + epsilon)/(1+f_opt1)*c;
c6 = (mean(d6.S1.ocost(1:end,:),2) - f_opt + epsilon)/(1+f_opt1)*c;


c1 = (mean(d.S1.ocost(1:end,:),2) - f_opt + epsilon)/(1+f_opt1)*c;
c2 = (mean(d1.S1.ocost(1:end,:),2) - f_opt + epsilon)/(1+f_opt1)*c;
c3 = (mean(d2.S1.ocost(1:end,:),2) - f_opt + epsilon)/(1+f_opt1)*c;
c4 = (mean(d4.S1.ocost(1:end,:),2) - f_opt + epsilon)/(1+f_opt1)*c;
c5 = (mean(d5.S1.ocost(1:end,:),2) - f_opt + epsilon)/(1+f_opt1)*c;
c6 = (mean(d6.S1.ocost(1:end,:),2) - f_opt + epsilon)/(1+f_opt1)*c;

% s1 = isbar*std(d.S1.ocost(1:end,:),[],2)/(1+f_opt1);
% s2 = isbar*std(d1.S1.ocost(1:end,:),[],2)/(1+f_opt1);
% s3 = isbar*std(d2.S1.ocost(1:end,:),[],2)/(1+f_opt1);
% s4 = isbar*std(d4.S1.ocost(1:end,:),[],2)/(1+f_opt1);
% % s5 = isbar*std(d5.S1.ocost(1:end,:),[],2)/(1+f_opt1);
% disp('svrg');
% [d.S1.ocost c1]
% disp('svrg-bb');
% [d1.S1.ocost c2]
% disp('svrg-2bb');
% [d2.S1.ocost c3]
% disp('svrg-dh');
% [d4.S1.ocost c4]
% disp('svrg-new step');
%[d5.S1.ocost c5]
 figure; hold on;
plot(d.S1.epoch(1:endd,:),(c1),'r--*','MarkerSize',msize,'LineWidth',lsize);hold on;
plot(d1.S1.epoch(1:endd,:),(c2),'b--o','MarkerSize',msize,'LineWidth',lsize);hold on;
plot(d2.S1.epoch(1:endd,:),(c3),'g-s','MarkerSize',msize,'LineWidth',lsize);hold on;
plot(d3.S1.epoch(2:endd,:),(d3.S1.variance(2:endd,:)),'k:>','MarkerSize',msize,'LineWidth',lsize);hold on;
plot(d4.S1.epoch(1:endd,:),(c4),'m--o','MarkerSize',msize,'LineWidth',lsize);hold on;
plot(d5.S1.epoch(1:endd,:),(c5(1:endd)),'k:>','MarkerSize',msize,'LineWidth',lsize);hold on;
plot(d6.S1.epoch(1:endd,:),(c6(1:endd)),'k--o','MarkerSize',msize,'LineWidth',lsize);hold on;
xlabel('Epoch','Fontsize',lgft)
ylabel('Original cost','Fontsize',lgft)
ylim(cost);
xlim(epoch);
G = gca;
set(gca,'Fontsize',fnt);
G.YScale = 'log';
lgh=legend(fname1,'location','northeast','NumColumns',5);
% %saveas (gcf, 'Variance_Epoch' , 'epsc' )


e=1;

figure;hold on;
%title("Optimal gap Vs. Epoch");
errorbar(d.S1.epoch(e:endd,:),((mean(abs(d.S1.ocost(e:endd,:)),2))-f_opt+epsilon)/(1+f_opt),isbar*(std(d.S1.ocost(e:endd,:),[],2))/(1+f_opt1),'r--*','MarkerSize',msize,'LineWidth',lsize);hold on;
errorbar(d1.S1.epoch(e:endd,:),((mean(abs(d1.S1.ocost(e:endd,:)),2))-f_opt+epsilon)/(1+f_opt),isbar*(std(d1.S1.ocost(e:endd,:),[],2))/(1+f_opt1),'b--o','MarkerSize',msize,'LineWidth',lsize);hold on;
errorbar(d2.S1.epoch(e:endd,:),((mean(abs(d2.S1.ocost(e:endd,:)),2))-f_opt+epsilon)/(1+f_opt),isbar*(std(d2.S1.ocost(e:endd,:),[],2))/(1+f_opt1),'g-s','MarkerSize',msize,'LineWidth',lsize);hold on;
%errorbar(d3.S1.epoch(e:endd,:),abs((mean(abs(d3.S1.ocost(e:endd,:)),2))-f_opt+epsilon)/(1+f_opt),isbar*(std(d3.S1.ocost(e:endd,:),[],2)),'k:>','MarkerSize',msize,'LineWidth',lsize);hold on;
errorbar(d4.S1.epoch(e:endd,:),((mean(abs(d4.S1.ocost(e:endd,:)),2))-f_opt+epsilon)/(1+f_opt),isbar*(std(d4.S1.ocost(e:endd,:),[],2))/(1+f_opt1),'m--o','MarkerSize',msize,'LineWidth',lsize);hold on;
errorbar(d5.S1.epoch(e:endd,:),((mean(abs(d5.S1.ocost(e:endd,:)),2))-f_opt+epsilon)/(1+f_opt),isbar*(std(d5.S1.ocost(e:endd,:),[],2))/(1+f_opt1),'k:>','MarkerSize',msize,'LineWidth',lsize);hold on;
xlabel('Epoch','Fontsize',lgft)
ylabel('relative error','Fontsize',lgft)
ylim(cost);
xlim(epoch);
legend(fname1,'location','northeast','NumColumns',1);
G = gca;
G.YScale = 'log';
set(gca,'Fontsize',fnt);
%legendmarkeradjust(8,12)
saveas (gcf, 'opt_gap_Epoch' , 'epsc' )


figure;hold on;
%title("Optimal gap Vs. time");
e=1;
errorbar(d.S1.otime(e:endd,:),((mean(abs(d.S1.ocost(e:endd,:)),2))-f_opt),isbar*(std(d.S1.ocost(e:endd,:),[],2))/(1+f_opt),'r--*','MarkerSize',msize,'LineWidth',lsize);hold on;
errorbar(d1.S1.otime(e:endd,:),((mean(abs(d1.S1.ocost(e:endd,:)),2))-f_opt),isbar*(std(d1.S1.ocost(e:endd,:),[],2))/(1+f_opt),'b--o','MarkerSize',msize,'LineWidth',lsize);hold on;
errorbar(d2.S1.otime(e:endd,:),((mean(abs(d2.S1.ocost(e:endd,:)),2))-f_opt),isbar*(std(d2.S1.ocost(e:endd,:),[],2))/(1+f_opt),'g-s','MarkerSize',msize,'LineWidth',lsize);hold on;
%errorbar(d3.S1.otime(e:endd,:),abs((mean(abs(d3.S1.ocost(e:endd,:)),2))-f_opt),isbar*(std(d3.S1.ocost(e:endd,:),[],2))/(1+f_opt),'k:>','MarkerSize',msize,'LineWidth',lsize);hold on;
errorbar(d4.S1.otime(e:endd,:),((mean(abs(d4.S1.ocost(e:endd,:)),2))-f_opt),isbar*(std(d4.S1.ocost(e:endd,:),[],2))/(1+f_opt),'m--o','MarkerSize',msize,'LineWidth',lsize);hold on;
errorbar(mean(d5.S1.otime(e:endd,:),2),((mean(abs(d5.S1.ocost(e:endd,:)),2))-f_opt),(isbar*std(d5.S1.ocost(e:endd,:),[],2))/(1+f_opt),'k:>','MarkerSize',msize,'LineWidth',lsize);hold on;
xlabel('time','Fontsize',lgft)
ylabel('relative error','Fontsize',lgft)
ylim(cost);
xlim(time);
% xticks(20:40:340);
G = gca;
G.YScale = 'log';
set(gca,'Fontsize',fnt);
legend(fname1,'location','northeast');
%legendmarkeradjust(8,12)
saveas (gcf, 'opt_gap_time' , 'epsc' )

%endd=20;
fig=figure;hold on;
%title("Variance Vs. Epoch");
plot(d.S1.epoch(2:endd,:),(d.S1.variance(2:endd,:)),'r--*','MarkerSize',msize,'LineWidth',lsize);hold on;
plot(d1.S1.epoch(2:endd,:),(d1.S1.variance(2:endd,:)),'b--o','MarkerSize',msize,'LineWidth',lsize);hold on;
plot(d2.S1.epoch(2:endd,:),(d2.S1.variance(2:endd,:)),'g-s','MarkerSize',msize,'LineWidth',lsize);hold on;
%plot(d3.S1.epoch(2:endd,:),(d3.S1.variance(2:endd,:)),'k:>','MarkerSize',msize,'LineWidth',lsize);hold on;
plot(d4.S1.epoch(2:endd,:),(d4.S1.variance(2:endd,:)),'m--o','MarkerSize',msize,'LineWidth',lsize);hold on;
plot(d5.S1.epoch(2:endd,:),mean(d5.S1.variance(2:endd,:),2),'k:>','MarkerSize',msize,'LineWidth',lsize);hold on;
xlabel('Epoch','Fontsize',lgft)
ylabel('Variance','Fontsize',lgft)
ylim(cost);
xlim(epoch);
G = gca;
set(gca,'Fontsize',fnt);
G.YScale = 'log';
lgh=legend(fname1,'location','northeast','NumColumns',5);
saveas (gcf, 'Variance_Epoch' , 'epsc' )
%saveLegendToImage(fig, lgh, 'legend3', 'epsc');


figure;hold on;
%title("Variance Vs. Time");
plot(d.S1.otime(2:endd,:),(d.S1.variance(2:endd,:)),'r--*','MarkerSize',msize,'LineWidth',lsize);hold on;
plot(d1.S1.otime(2:endd,:),(d1.S1.variance(2:endd,:)),'b--o','MarkerSize',msize,'LineWidth',lsize);hold on;
plot(d2.S1.otime(2:endd,:),(d2.S1.variance(2:endd,:)),'g-s','MarkerSize',msize,'LineWidth',lsize);hold on;
%plot(d3.S1.otime(2:endd,:),(d3.S1.variance(2:endd,:)),'k:>','MarkerSize',msize,'LineWidth',lsize);hold on;
plot(d4.S1.otime(2:endd,:),(d4.S1.variance(2:endd,:)),'m--o','MarkerSize',msize,'LineWidth',lsize);hold on;
plot(mean(d5.S1.otime(2:endd,:),2),mean(d5.S1.variance(2:endd,:),2),'k:>','MarkerSize',msize,'LineWidth',lsize);hold on;
xlabel('Time','Fontsize',lgft)
ylabel('Variance','Fontsize',lgft)
ylim(cost);
xlim(time);
% xticks(20:40:340);
legend(fname1,'location','northeast');
G = gca;
G.YScale = 'log';
set(gca,'Fontsize',fnt);
saveas (gcf, 'Variance_Time' , 'epsc' )


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

end