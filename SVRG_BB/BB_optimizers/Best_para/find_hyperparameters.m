close all;
clear;
clc;

path='RCV/';
%path='Ijcnn/';
fname={'svrg-dh','svrg-bb','svrg-2bb','svrg-bbb','svrg'};
%fname1={'SVRG-2D','SVRG-BB','SVRG-2BB','SVRG-2BBS','SVRG'};
%fname1={'SVRG','SVRG-BB','SVRG-2BB','SVRG-2BBS'};
fname1={'SVRG','SVRG-BB','SVRG-2BB','SVRG-2BBS','SVRG-DH'};

find_par=1;
%find_par=1;
reg = 1e-0;
%best = nys_best(path, find_par);

msize=9;
lsize=1.5;
isbar=1;
acc=[0.5,1];
%cost=[1e-18,1];
cost = ['auto'];
%time=[1,250];
time = 'auto';
epoch=[0,8];

best = other_best(strcat(path,'svrg'), find_par,reg);
best1 = other_best(strcat(path,'svrg_bb'), find_par,reg);
best2 = other_best(strcat(path,'svrgbb'), find_par,reg);
best3 = other_best(strcat(path,'svrg_bbb'), find_par,reg);
best4 = other_best(strcat(path,'svrgdh'), find_par,reg);

Name = strcat(path,'svrg',sprintf('_%.1e_R_%.1e.mat',best(11),best(12)));
d=load(Name);
Name1 = strcat(path,'svrg_bb',sprintf('_%.1e_R_%.1e.mat',best1(11),best1(12)));
d1=load(Name1);
fprintf('\n \n Now our BB\n \n');
Name2 = strcat(path,'svrgbb',sprintf('_%.1e_R_%.1e.mat',best2(11),best2(12)));
d2=load(Name2);
Name3 = strcat(path,'svrg_bbb',sprintf('_%.1e_R_%.1e.mat',best3(11),best3(12)));
d3=load(Name3);
Name4 = strcat(path,'svrgdh',sprintf('_%.1e_R_%.1e.mat',best4(11),best4(12)));
d4=load(Name4);

% figure;hold on;
% title("Train Acc Vs. Epoch");
% errorbar(d.S1.epoch(2:end,:),mean(d.S1.train_ac(2:end,:),2),isbar*std(d.S1.train_ac(2:end,:),[],2),'r--*','MarkerSize',msize,'LineWidth',lsize);hold on;
% errorbar(d.S1.epoch(2:end,:),mean(d1.S1.train_ac(2:end,:),2),isbar*std(d1.S1.train_ac(2:end,:),[],2),'b--*','MarkerSize',msize,'LineWidth',lsize);hold on;
% errorbar(d.S1.epoch(2:end,:),mean(d2.S1.train_ac(2:end,:),2),isbar*std(d2.S1.train_ac(2:end,:),[],2),'g--*','MarkerSize',msize,'LineWidth',lsize);hold on;
% errorbar(d.S1.epoch(2:end,:),mean(d3.S1.train_ac(2:end,:),2),isbar*std(d3.S1.train_ac(2:end,:),[],2),'k--*','MarkerSize',msize,'LineWidth',lsize);hold on;
% errorbar(d.S1.epoch(2:end,:),mean(d4.S1.train_ac(2:end,:),2),isbar*std(d4.S1.train_ac(2:end,:),[],2),'m--*','MarkerSize',msize,'LineWidth',lsize);hold on;
% xlabel('Epoch','FontSize',15)
% ylabel('Train Acc','FontSize',15)
% ylim(acc);
% xlim(epoch);
% legend(fname,'location','southeast');
% G = gca;
% G.YScale = 'log';
% set(gca,'Fontsize',14);
% saveas (gcf, 'Train_Acc_Epoch' , 'epsc' )
e=1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure;hold on;
% %title("Optimal gap Vs. Epoch");
% errorbar(d.S1.epoch(e:end,:),mean(d.S1.train_ac(e:end,:),2),isbar*std(d.S1.train_ac(e:end,:),[],2),'r--*','MarkerSize',msize,'LineWidth',lsize);hold on;
% errorbar(d1.S1.epoch(e:end,:),mean(d1.S1.train_ac(e:end,:),2),isbar*std(d1.S1.train_ac(e:end,:),[],2),'b--o','MarkerSize',msize,'LineWidth',lsize);hold on;
% errorbar(d2.S1.epoch(e:end,:),mean(d2.S1.train_ac(e:end,:),2),isbar*std(d2.S1.train_ac(e:end,:),[],2),'g-s','MarkerSize',msize,'LineWidth',lsize);hold on;
% errorbar(d3.S1.epoch(e:end,:),mean(d3.S1.train_ac(e:end,:),2),isbar*std(d3.S1.train_ac(e:end,:),[],2),'k:>','MarkerSize',msize,'LineWidth',lsize);hold on;
% errorbar(d4.S1.epoch(e:end,:),mean(d4.S1.train_ac(e:end,:),2),isbar*std(d4.S1.train_ac(e:end,:),[],2),'m--*','MarkerSize',msize,'LineWidth',lsize);hold on;
% 
% xlabel('Epoch','Fontsize',15)
% ylabel('Opt gap','Fontsize',15)
% ylim([0.75,1]);
% xlim(epoch);
% legend(fname1,'location','northeast');
% G = gca;
% %G.YScale = 'log';
% set(gca,'Fontsize',14);
% saveas (gcf, 'Train_acc' , 'epsc' )
% 
% 
% figure;hold on;
% %title("Optimal gap Vs. Epoch");
% errorbar(d.S1.epoch(e:end,:),mean(d.S1.val_ac(e:end,:),2),isbar*std(d.S1.val_ac(e:end,:),[],2),'r--*','MarkerSize',msize,'LineWidth',lsize);hold on;
% errorbar(d1.S1.epoch(e:end,:),mean(d1.S1.val_ac(e:end,:),2),isbar*std(d1.S1.val_ac(e:end,:),[],2),'b--o','MarkerSize',msize,'LineWidth',lsize);hold on;
% errorbar(d2.S1.epoch(e:end,:),mean(d2.S1.val_ac(e:end,:),2),isbar*std(d2.S1.val_ac(e:end,:),[],2),'g-s','MarkerSize',msize,'LineWidth',lsize);hold on;
% errorbar(d3.S1.epoch(e:end,:),mean(d3.S1.val_ac(e:end,:),2),isbar*std(d3.S1.val_ac(e:end,:),[],2),'k:>','MarkerSize',msize,'LineWidth',lsize);hold on;
% errorbar(d4.S1.epoch(e:end,:),mean(d4.S1.val_ac(e:end,:),2),isbar*std(d4.S1.val_ac(e:end,:),[],2),'m--*','MarkerSize',msize,'LineWidth',lsize);hold on;
% 
% xlabel('Epoch','Fontsize',15)
% ylabel('Opt gap','Fontsize',15)
% ylim([0.85,1]);
% xlim(epoch);
% legend(fname1,'location','northeast');
% G = gca;
% %G.YScale = 'log';
% set(gca,'Fontsize',14);
% saveas (gcf, 'Valc' , 'epsc' )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure;hold on;
% title("Val Acc Vs. Epoch");
% errorbar(d.S1.epoch(2:end,:),mean(d.S1.val_ac(2:end,:),2),isbar*std(d.S1.val_ac(2:end,:),[],2),'r-*','MarkerSize',msize,'LineWidth',lsize);
% errorbar(d.S1.epoch(2:end,:),mean(d1.S1.val_ac(2:end,:),2),isbar*std(d1.S1.val_ac(2:end,:),[],2),'b-*','MarkerSize',msize,'LineWidth',lsize);
% errorbar(d.S1.epoch(2:end,:),mean(d2.S1.val_ac(2:end,:),2),isbar*std(d2.S1.val_ac(2:end,:),[],2),'g-*','MarkerSize',msize,'LineWidth',lsize);
% errorbar(d.S1.epoch(2:end,:),mean(d3.S1.val_ac(2:end,:),2),isbar*std(d3.S1.val_ac(2:end,:),[],2),'k-*','MarkerSize',msize,'LineWidth',lsize);
% %errorbar(d.S1.epoch(2:end,:),mean(d4.S1.val_ac(2:end,:),2),isbar*std(d4.S1.val_ac(2:end,:),[],2),'m-*','MarkerSize',msize,'LineWidth',lsize);
% xlabel('Epoch','FontSize',15)
% ylabel('Val Acc','FontSize',15)
% ylim(acc);
% xlim(epoch);
% legend(fname,'location','southeast');
% %G = gca;
% %G.YScale = 'log';
% set(gca,'Fontsize',14);
% saveas (gcf, 'Val_Acc_Epoch' , 'epsc' )

% figure;hold on;
% title("Train Cost Vs. Epoch");
% errorbar(d.S1.epoch(1:end,:),mean(d1.S1.ocost(1:end,:),2),isbar*std(d1.S1.ocost(1:end,:),[],2),'b--*','MarkerSize',msize,'LineWidth',lsize);hold on;
% errorbar(d.S1.epoch(1:end,:),mean(d2.S1.ocost(1:end,:),2),isbar*std(d2.S1.ocost(1:end,:),[],2),'g--*','MarkerSize',msize,'LineWidth',lsize);hold on;
% errorbar(d.S1.epoch(1:end,:),mean(d3.S1.ocost(1:end,:),2),isbar*std(d3.S1.ocost(1:end,:),[],2),'k--*','MarkerSize',msize,'LineWidth',lsize);hold on;
% errorbar(d.S1.epoch(1:end,:),mean(d4.S1.ocost(1:end,:),2),isbar*std(d4.S1.ocost(1:end,:),[],2),'m--*','MarkerSize',msize,'LineWidth',lsize);hold on;
% errorbar(d.S1.epoch(1:end,:),mean(d.S1.ocost(1:end,:),2),isbar*std(d.S1.ocost(1:end,:),[],2),'r--*','MarkerSize',msize,'LineWidth',lsize);hold on;
% xlabel('Epoch')
% ylabel('Train Cost')
% ylim(cost);
% xlim(epoch);
% legend(fname1,'location','northeast');
%G = gca;
%G.YScale = 'log';
% saveas (gcf, 'Train_Cost_Epoch' , 'epsc' )
% 
% figure;hold on;
% title("Val Cost Vs. Epoch");
% errorbar(d1.S1.epoch(2:end,:),mean(d1.S1.vcost(2:end,:),2),isbar*std(d1.S1.vcost(2:end,:),[],2),'b-*','MarkerSize',msize,'LineWidth',lsize);
% errorbar(d2.S1.epoch(2:end,:),mean(d2.S1.vcost(2:end,:),2),isbar*std(d2.S1.vcost(2:end,:),[],2),'g-*','MarkerSize',msize,'LineWidth',lsize);
% errorbar(d3.S1.epoch(2:end,:),mean(d3.S1.vcost(2:end,:),2),isbar*std(d3.S1.vcost(2:end,:),[],2),'k-*','MarkerSize',msize,'LineWidth',lsize);
% errorbar(d4.S1.epoch(2:end,:),mean(d4.S1.vcost(2:end,:),2),isbar*std(d4.S1.vcost(2:end,:),[],2),'m-*','MarkerSize',msize,'LineWidth',lsize);
% errorbar(d.S1.epoch(2:end,:),mean(d.S1.vcost(2:end,:),2),isbar*std(d.S1.vcost(2:end,:),[],2),'r-*','MarkerSize',msize,'LineWidth',lsize);
% xlabel('Epoch')
% ylabel('Val Cost')
% ylim(cost);
% xlim(epoch);
% legend(fname1,'location','northeast');
% saveas (gcf, 'Val_Cost_Epoch' , 'epsc' )

% % % 
% figure;hold on;
% %title("Train Acc Vs. Time");
% errorbar(d.S1.otime(2:7),mean(d.S1.train_ac(2:7,:),2),isbar*std(d.S1.train_ac(2:7,:),[],2),'r--*','MarkerSize',msize,'LineWidth',lsize);hold on;
% errorbar(d1.S1.otime(2:7),mean(d1.S1.train_ac(2:7,:),2),isbar*std(d1.S1.train_ac(2:7,:),[],2),'b--*','MarkerSize',msize,'LineWidth',lsize);hold on;
% errorbar(d2.S1.otime(2:7),mean(d2.S1.train_ac(2:7,:),2),isbar*std(d2.S1.train_ac(2:7,:),[],2),'g--*','MarkerSize',msize,'LineWidth',lsize);hold on;
% errorbar(d3.S1.otime(2:7),mean(d3.S1.train_ac(2:7,:),2),isbar*std(d3.S1.train_ac(2:7,:),[],2),'k--*','MarkerSize',msize,'LineWidth',lsize);hold on;
% %errorbar(d4.S1.otime(2:end),mean(d4.S1.train_ac(2:end,:),2),isbar*std(d4.S1.train_ac(2:end,:),[],2),'m--*','MarkerSize',msize,'LineWidth',lsize);hold on;
% xlabel('Time','Fontsize',15)
% ylabel('Train Acc','Fontsize',15)
% ylim(acc);
% xlim([20,80]);
% legend(fname1,'location','southeast');
% G = gca;
% G.YScale = 'log';
% set(gca,'Fontsize',14);
% saveas (gcf, 'Train_Acc_Time' , 'epsc' )
% 
% figure;hold on;
% %title("Val Acc Vs. Time");
% errorbar(d.S1.otime(2:7),mean(d.S1.val_ac(2:7,:),2),isbar*std(d.S1.val_ac(2:7,:),[],2),'r-*','MarkerSize',msize,'LineWidth',lsize);
% errorbar(d1.S1.otime(2:7),mean(d1.S1.val_ac(2:7,:),2),isbar*std(d1.S1.val_ac(2:7,:),[],2),'b-*','MarkerSize',msize,'LineWidth',lsize);
% errorbar(d2.S1.otime(2:7),mean(d2.S1.val_ac(2:7,:),2),isbar*std(d2.S1.val_ac(2:7,:),[],2),'g-*','MarkerSize',msize,'LineWidth',lsize);
% errorbar(d3.S1.otime(2:7),mean(d3.S1.val_ac(2:7,:),2),isbar*std(d3.S1.val_ac(2:7,:),[],2),'k-*','MarkerSize',msize,'LineWidth',lsize);
% %errorbar(d4.S1.otime(2:end),mean(d4.S1.val_ac(2:end,:),2),isbar*std(d4.S1.val_ac(2:end,:),[],2),'m-*','MarkerSize',msize,'LineWidth',lsize);
% xlabel('Time (seconds)','Fontsize',15)
% ylabel('Val Acc','Fontsize',15)
% ylim([0.6,1]);
% xlim([20,70]);
% legend(fname1,'location','southeast');
% %G = gca;
% %G.YScale = 'log';
% set(gca,'Fontsize',14);
% saveas (gcf, 'Val_Acc_Time' , 'epsc' )
% 
% figure;hold on;
% title("Train Cost Vs. Time");
% errorbar(d.S1.otime(2:end),mean(d.S1.ocost(2:end,:),2),isbar*std(d.S1.ocost(2:end,:),[],2),'r--*','MarkerSize',msize,'LineWidth',lsize);hold on;
% errorbar(d1.S1.otime(2:end),mean(d1.S1.ocost(2:end,:),2),isbar*std(d1.S1.ocost(2:end,:),[],2),'b--*','MarkerSize',msize,'LineWidth',lsize);hold on;
% errorbar(d2.S1.otime(2:end),mean(d2.S1.ocost(2:end,:),2),isbar*std(d2.S1.ocost(2:end,:),[],2),'g--*','MarkerSize',msize,'LineWidth',lsize);hold on;
% errorbar(d3.S1.otime(2:end),mean(d3.S1.ocost(2:end,:),2),isbar*std(d3.S1.ocost(2:end,:),[],2),'k--*','MarkerSize',msize,'LineWidth',lsize);hold on;
% %errorbar(d4.S1.otime(2:end),mean(d4.S1.ocost(2:end,:),2),isbar*std(d4.S1.ocost(2:end,:),[],2),'m--*','MarkerSize',msize,'LineWidth',lsize);hold on;
% 
% xlabel('Time (seconds)')
% ylabel('Train Cost')
% ylim(cost);
% xlim(time);
% legend(fname1,'location','northeast');
% G = gca;
% G.YScale = 'log';
% saveas (gcf, 'Train_Cost_Time' , 'epsc' )

% figure;hold on;
% title("Val Cost Vs. Time");
% errorbar(d1.S1.otime(2:end),mean(d1.S1.vcost(2:end,:),2),isbar*std(d1.S1.vcost(2:end,:),[],2),'b-*','MarkerSize',msize,'LineWidth',lsize);
% errorbar(d2.S1.otime(2:end),mean(d2.S1.vcost(2:end,:),2),isbar*std(d2.S1.vcost(2:end,:),[],2),'g-*','MarkerSize',msize,'LineWidth',lsize);
% errorbar(d3.S1.otime(2:end),mean(d3.S1.vcost(2:end,:),2),isbar*std(d3.S1.vcost(2:end,:),[],2),'k-*','MarkerSize',msize,'LineWidth',lsize);
% errorbar(d4.S1.otime(2:end),mean(d4.S1.vcost(2:end,:),2),isbar*std(d4.S1.vcost(2:end,:),[],2),'m-*','MarkerSize',msize,'LineWidth',lsize);
% errorbar(d.S1.otime(2:end),mean(d.S1.vcost(2:end,:),2),isbar*std(d.S1.vcost(2:end,:),[],2),'r-*','MarkerSize',msize,'LineWidth',lsize);
% xlabel('Time (seconds)')
% ylabel('Val Cost')
% ylim(cost);
% xlim(time);
% legend(fname1,'location','northeast');
% G = gca;
% G.YScale = 'log';
% saveas (gcf, 'Val_Cost_Time' , 'epsc' )
% 

e=1;

figure;hold on;
%title("Optimal gap Vs. Epoch");
errorbar(d.S1.epoch(e:end,:),mean(d.S1.opt_gap(e:end,:),2),isbar*std(d.S1.opt_gap(e:end,:),[],2),'r--*','MarkerSize',msize,'LineWidth',lsize);hold on;
errorbar(d1.S1.epoch(e:end,:),mean(d1.S1.opt_gap(e:end,:),2),isbar*std(d1.S1.opt_gap(e:end,:),[],2),'b--o','MarkerSize',msize,'LineWidth',lsize);hold on;
errorbar(d2.S1.epoch(e:end,:),mean(d2.S1.opt_gap(e:end,:),2),isbar*std(d2.S1.opt_gap(e:end,:),[],2),'g-s','MarkerSize',msize,'LineWidth',lsize);hold on;
errorbar(d3.S1.epoch(e:end,:),mean(d3.S1.opt_gap(e:end,:),2),isbar*std(d3.S1.opt_gap(e:end,:),[],2),'k:>','MarkerSize',msize,'LineWidth',lsize);hold on;
errorbar(d4.S1.epoch(e:end,:),mean(d4.S1.opt_gap(e:end,:),2),isbar*std(d4.S1.opt_gap(e:end,:),[],2),'m--*','MarkerSize',msize,'LineWidth',lsize);hold on;

xlabel('Epoch','Fontsize',15)
ylabel('Opt gap','Fontsize',15)
ylim(cost);
xlim(epoch);
legend(fname1,'location','northeast','NumColumns',1);
G = gca;
G.YScale = 'log';
set(gca,'Fontsize',15);
%legendmarkeradjust(8,12)
saveas (gcf, 'opt_gap_Epoch' , 'epsc' )


figure;hold on;
%title("Optimal gap Vs. time");
e=1;
errorbar(d.S1.otime(e:end,:),mean(d.S1.opt_gap(e:end,:),2),isbar*std(d.S1.opt_gap(e:end,:),[],2),'r--*','MarkerSize',msize,'LineWidth',lsize);hold on;
errorbar(d1.S1.otime(e:end,:),mean(d1.S1.opt_gap(e:end,:),2),isbar*std(d1.S1.opt_gap(e:end,:),[],2),'b--o','MarkerSize',msize,'LineWidth',lsize);hold on;
errorbar(d2.S1.otime(e:end,:),mean(d2.S1.opt_gap(e:end,:),2),isbar*std(d2.S1.opt_gap(e:end,:),[],2),'g-s','MarkerSize',msize,'LineWidth',lsize);hold on;
errorbar(d3.S1.otime(e:end,:),mean(d3.S1.opt_gap(e:end,:),2),isbar*std(d3.S1.opt_gap(e:end,:),[],2),'k:>','MarkerSize',msize,'LineWidth',lsize);hold on;
errorbar(d4.S1.otime(e:end,:),mean(d4.S1.opt_gap(e:end,:),2),isbar*std(d4.S1.opt_gap(e:end,:),[],2),'m--*','MarkerSize',msize,'LineWidth',lsize);hold on;

% %%%
% errorbar(d.S1.otime(2:end),mean(d.S1.ocost(2:end,:),2),isbar*std(d.S1.ocost(2:end,:),[],2),'b-*','MarkerSize',msize,'LineWidth',lsize);
% errorbar(d1.S1.otime(2:end),mean(d1.S1.ocost(2:end,:),2),isbar*std(d1.S1.ocost(2:end,:),[],2),'g-*','MarkerSize',msize,'LineWidth',lsize);
% errorbar(d2.S1.otime(2:end),mean(d2.S1.ocost(2:end,:),2),isbar*std(d2.S1.ocost(2:end,:),[],2),'k-*','MarkerSize',msize,'LineWidth',lsize);
% errorbar(d3.S1.otime(2:end),mean(d3.S1.ocost(2:end,:),2),isbar*std(d3.S1.ocost(2:end,:),[],2),'m-*','MarkerSize',msize,'LineWidth',lsize);
% errorbar(d4.S1.otime(2:end),mean(d4.S1.ocost(2:end,:),2),isbar*std(d4.S1.ocost(2:end,:),[],2),'r-*','MarkerSize',msize,'LineWidth',lsize);

%%%
xlabel('time','Fontsize',15)
ylabel('Opt gap','Fontsize',15)
ylim(cost);
xlim(time);
% xticks(20:40:340);
G = gca;
G.YScale = 'log';
set(gca,'Fontsize',15);
legend(fname1,'location','northeast');
legendmarkeradjust(8,12)
saveas (gcf, 'opt_gap_time' , 'epsc' )

%%%%

fig=figure;hold on;
%title("Variance Vs. Epoch");
plot(d.S1.epoch(2:end,:),(d.S1.variance(2:end,:)),'r--*','MarkerSize',msize,'LineWidth',lsize);hold on;
plot(d1.S1.epoch(2:end,:),(d1.S1.variance(2:end,:)),'b--o','MarkerSize',msize,'LineWidth',lsize);hold on;
plot(d2.S1.epoch(2:end,:),(d2.S1.variance(2:end,:)),'g-s','MarkerSize',msize,'LineWidth',lsize);hold on;
plot(d3.S1.epoch(2:end,:),(d3.S1.variance(2:end,:)),'k:>','MarkerSize',msize,'LineWidth',lsize);hold on;
plot(d4.S1.epoch(2:end,:),(d4.S1.variance(2:end,:)),'m--*','MarkerSize',msize,'LineWidth',lsize);hold on;
xlabel('Epoch','Fontsize',12)
ylabel('Variance','Fontsize',12)
ylim(cost);
xlim(epoch);
G = gca;
set(gca,'Fontsize',13);
G.YScale = 'log';
lgh=legend(fname1,'location','northeast','NumColumns',1);
saveas (gcf, 'Variance_Epoch' , 'epsc' )
%saveLegendToImage(fig, lgh, 'legend3', 'epsc');




 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% 
figure;hold on;
%title("Variance Vs. Time");
plot(d.S1.otime(2:end,:),(d.S1.variance(2:end,:)),'r--*','MarkerSize',msize,'LineWidth',lsize);hold on;
plot(d1.S1.otime(2:end,:),(d1.S1.variance(2:end,:)),'b--o','MarkerSize',msize,'LineWidth',lsize);hold on;
plot(d2.S1.otime(2:end,:),(d2.S1.variance(2:end,:)),'g-s','MarkerSize',msize,'LineWidth',lsize);hold on;
plot(d3.S1.otime(2:end,:),(d3.S1.variance(2:end,:)),'k:>','MarkerSize',msize,'LineWidth',lsize);hold on;
plot(d4.S1.otime(2:end,:),(d4.S1.variance(2:end,:)),'m--*','MarkerSize',msize,'LineWidth',lsize);hold on;

xlabel('Time','Fontsize',15)
ylabel('Variance','Fontsize',15)
ylim(cost);
xlim(time);
% xticks(20:40:340);
legend(fname1,'location','northeast');
G = gca;
G.YScale = 'log';
set(gca,'Fontsize',14);
saveas (gcf, 'Variance_Time' , 'epsc' )
