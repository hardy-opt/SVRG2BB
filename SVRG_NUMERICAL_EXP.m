function  SVRG_NUMERICAL_EXP()

    clc;
    clear;
    close all;
    datast = {'Adult','Ijcnn','Gisette','Mnist','W8a','Covtype'};
for d = 2:2 % 1:6%2%dataset        
    dat = char(datast(d));
    
    for m=9:9%10%methods
    
        for reg = [1e-3] 
   
                for step =  [0.1 0.01 0.001 0.0001 0.00001 1 10 100 1000]

                  Tr_s1 = []; %Tr_s2 = []; Tr_s3 = [];
                  C_s1 = []; %C_s2 = []; C_s3 = [];
                  Vc_s1 = [];% Vc_s2 = []; Vc_s3 = [];
                  Vl_s1 = []; %Vl_s2 = []; Vl_s3 = [];
                  otime_s1 =[];% otime_s2 =[]; otime_s3 =[];
                  opt_g = [];
                  var = [];
                  lr_rate = [];
    
                        for s=1:2
                            
                            fprintf('\n ======      Loop number: S=%d, Step=%f, reg=%f,  m = %d, data = %d ======== \n',s,step,reg,m,d);
                            if d==1
                                data = ADULT(s); 
                                problem = logistic_regression1(data.x_train, data.y_train, data.x_test, data.y_test,reg); 
                                options.max_epoch=60; %30    
                                [w_opt,infos_LBFGS] = problem.calc_solution(100);
                            elseif d==2
                                data = IJCNN1(s);
                                problem = linear_svm1(data.x_train, data.y_train, data.x_test, data.y_test,reg);
                                options.max_epoch=20; %30;    
                                [w_opt,infos_LBFGS] = problem.calc_solution(80);%80
                                size(infos_LBFGS.cost)
                            elseif d==3
                                data = GISETTE(s);
                                problem = logistic_regression1(data.x_train, data.y_train, data.x_test, data.y_test,reg); 
                                options.max_epoch=35; %20    
                                [w_opt,infos_LBFGS] = problem.calc_solution(45);
                            elseif d==4
                                data = MNIST38(s);
                                problem = linear_svm1(data.x_train, data.y_train, data.x_test, data.y_test,reg); 
                                options.max_epoch=75; %25    
                                [w_opt,infos_LBFGS] = problem.calc_solution(200);
                            elseif d==5
                                data = W8A(s);
                                %problem = linear_svm1(data.x_train, data.y_train, data.x_test, data.y_test,reg);
                                problem = logistic_regression1(data.x_train, data.y_train, data.x_test, data.y_test,reg);
                                options.max_epoch=60;    
                                [w_opt,infos_LBFGS] = problem.calc_solution(100);
                            elseif d==6
                                data = COVTYPE(s);
                                problem = logistic_regression1(data.x_train, data.y_train, data.x_test, data.y_test,reg); 
                                options.max_epoch=20;    
                                [w_opt,infos_LBFGS] = problem.calc_solution(45);
                                
                            end


                            f_opt = problem.cost(w_opt); 
                            fprintf('f_opt: %.24e\n', f_opt);  
                            options.f_opt = f_opt;
                            options.w_init = data.w_init;   
                            options.step_alg = 'fix';
                            options.step_init = step; 
                            options.verbose = 2;
                            options.regu = reg;
                            options.batch_size = 1;
                            options.tol_optgap = 10^-27;
    
                            if m==1
                                tic
                                [w_s1, info_s1] = svrg(problem, options);
                                toc
                            elseif m==2
                                tic
                                [w_s1, info_s1] = svrgdh(problem, options);
                                toc
                                fprintf('this is diagonal \n /n')
                                
                            
                            elseif m==3
                                tic
                                [w_s1, info_s1] = svrg_bb(problem, options);
                                toc
                            elseif m==4
                                tic
                                [w_s1, info_s1] = svrgbb(problem, options);
                                toc
                            elseif m==5
                                tic
                                [w_s1, info_s1] = svrgbbb(problem, options,1);  % (eta/1) * bb --> SVRG-2BBS --> BB step size with SVRG-2BB eta=fix
                                toc
                            elseif m==6
                                tic
                                 options.step_alg = 'decay';
                                [w_s1, info_s1] = svrgbbb(problem, options,2);  % (eta/1) * bb --> BB step size with (eta/m=1)*BB --> eta decay
                                toc
                            elseif m==7
                                tic
                                options.step_alg = 'fix';
                                [w_s1, info_s1] = svrgbbb(problem, options,3);  % (eta/m) * bb 
                            %    [w_s1, info_s1] = svrg2nd(problem, options);elseif m==5
                                toc
                            elseif m==8
                                tic
                                options.step_alg = 'fix';
                                [w_s1, info_s1] = svrgbbb(problem, options,4);  % (eta/1) * bb --> SVRG-2BBS --> BB step size with SVRG-2BB eta=fix
                                toc
                            elseif m==9
                                tic
                                options.step_init
                                options.step_alg = 'decay';
                                [w_s1, info_s1] = svrgbbb(problem, options,5);  % (eta/1) * bb --> BB step size with (eta/m=1)*BB --> eta decay
                                toc
                            elseif m==10
                                tic
                                options.step_alg = 'fix';
                                [w_s1, info_s1] = svrgbbb(problem, options,6);  % (eta/m) * bb 
                                toc
                            end
                            pathh = 'SVRG_BB/Results_July2022';
                            LBFGS = infos_LBFGS;
                            Name = sprintf('%s/%s/LBFGS_%.1e_R_%.1e.mat',pathh,dat,options.step_init,reg);
                            save(Name,'LBFGS');% 
                            if isinf(w_s1)
                                %(info_s1.iter(end) < options.max_epoch) && (info_s1.optgap(end) < options.tol_optgap)
                                break;
                            end
                            Tr_s1 = [Tr_s1 info_s1.acc_tr];
                            Vl_s1 = [Vl_s1 info_s1.acc_val];
                            C_s1 = [C_s1 info_s1.cost'];
                            Vc_s1 = [Vc_s1 info_s1.val_cost];
                            otime_s1 = [otime_s1 info_s1.time'];
                            opt_g = [opt_g info_s1.optgap'];
                            var = [var info_s1.var];
                            if m > 4
                            lr_rate = [lr_rate info_s1.lr_rate];
                            end
                            optm = f_opt;
                        end
                            if info_s1.iter(end) == options.max_epoch
                            info.epoch = info_s1.iter';    
                            info.train_ac = (Tr_s1);  %info_s1.std = mean(Tr_s1);
                            info.val_ac = (Vl_s1);  %info_s1.std = mean(Tr_s1);
                            info.ocost = C_s1;
                            info.vcost = Vc_s1;
                            info.opt_gap = opt_g;
                            info.otime=mean(otime_s1,s);
                            info.variance = mean(var,s);
                            if m>4
                            info.lr_rate = mean(lr_rate,s);
                            end
                            info.grad_count = (info_s1.grad_calc_count)';

                            S1=info;
                            
                            if m==1
                            Name = sprintf('%s/%s/svrg_%.1e_R_%.1e.mat',pathh,dat,options.step_init,reg);
                            save(Name,'S1');%
                            elseif m==2
                            Name = sprintf('%s/%s/svrg_2d_%.1e_R_%.1e.mat',pathh,dat,options.step_init,reg);
                            save(Name,'S1');%
                            elseif m==3
                            Name = sprintf('%s/%s/svrg_bb_%.1e_R_%.1e.mat',pathh,dat,options.step_init,reg);
                            save(Name,'S1');% SVRG_BB with BB step size
                            elseif m==4
                            Name = sprintf('%s/%s/svrg_2bb_%.1e_R_%.1e.mat',pathh,dat,options.step_init,reg);
                            save(Name,'S1');% SVRG_BB with second order info
                            elseif m==5
                            %Name = sprintf('/home/optima/Desktop/SVRG_library/Results_30_Dec2021/%s/svrg_bbb_%.1e_R_%.1e.mat',dat,options.step_init,reg);
                            Name = sprintf('%s/%s/svrg_2bbs_eta_one_%.1e_R_%.1e.mat',pathh,dat,options.step_init,reg);
                            save(Name,'S1');% SVRG_BB with BB step size and BB in 2nd order info.
                            elseif m==6
                            Name = sprintf('%s/%s/svrg_2bbs_eta_decay_%.1e_R_%.1e.mat',pathh,dat,options.step_init,reg);
                            save(Name,'S1');% SVRG_BB with BB step size and BB in 2nd order info. with eta and m = 1
                            elseif m==7
                            Name = sprintf('%s/%s/svrg_2bbs_eta_constant_%.1e_R_%.1e.mat',pathh,dat,options.step_init,reg);
                            save(Name,'S1'); % SVRG_BB with BB step size and BB in 2nd order info. with eta and m = 2n
                            elseif m==8
                            %Name = sprintf('/home/optima/Desktop/SVRG_library/Results_30_Dec2021/%s/svrg_bbb_%.1e_R_%.1e.mat',dat,options.step_init,reg);
                            Name = sprintf('%s/%s/svrg_2bbs_eta_one_m1_%.1e_R_%.1e.mat',pathh,dat,options.step_init,reg);
                            save(Name,'S1');% SVRG_BB with BB step size and BB in 2nd order info.
                            elseif m==9
                            Name = sprintf('%s/%s/svrg_2bbs_eta_decay_m1_%.1e_R_%.1e.mat',pathh,dat,options.step_init,reg);
                            save(Name,'S1');% SVRG_BB with BB step size and BB in 2nd order info. with eta and m = 1
                            elseif m==10
                            Name = sprintf('%s/%s/svrg_2bbs_eta_constant_m1_%.1e_R_%.1e.mat',pathh,dat,options.step_init,reg);
                            save(Name,'S1'); % SVRG_BB with BB step size and BB in 2nd order info. with eta and m = 2n
                            end
                            lr_rate
                            clear info_s1;
                            clear lr_rate
                            clear S1;
                            clear info;
                            end
                            
                        %end
                end
        end
    end
end
end


