function [w, infos] = svrgbbb(problem, in_options,mval)


    % set dimensions and samples
    d = problem.dim();
    n = problem.samples();

    % set local options 
    local_options = [];
    
    % merge options
    options = mergeOptions(get_default_options(d), local_options);   
    options = mergeOptions(options, in_options);         
    
    %%%%%%%%%%%%%%%
    acc_tr = [0];
    acc_val = [0];
    
    vr=[0];
    val_cost=[0];
    lr_rate = [0];
   % step=1;
    %%%%%%%%%%%%%%%%


    % initialize
    total_iter = 0;
    epoch = 0;
    grad_calc_count = 0;
    w = options.w_init;
    vg = norm(problem.full_grad(w))^2;
    var = [vg];
    num_of_bachces = floor(n / options.batch_size)*2;  
    
     
    if ~isfield(options, 'max_inner_iter')
        options.max_inner_iter = num_of_bachces;
    end
    
    step = options.step_init;
    stepn = step;
    sto = step;    
    regul = options.regu;

    
    %mval, eta val 
 
    
   
    % store first infos
    clear infos;    
    [infos, f_val, optgap] = store_infos(problem, w, options, [], epoch, grad_calc_count, 0);  
    
    
    % display infos
    if options.verbose > 0
        fprintf('SVRG-2BBS: Epoch = %03d, cost = %.16e, optgap = %.4e\n', epoch, f_val, optgap);
    end
    
    % set start time
    start_time = tic();
    
    % main loop
    %while (optgap > options.tol_optgap) && (epoch < options.max_epoch)
    while  (epoch < options.max_epoch)
     % permute samples
        %if options.permute_on
          %  perm_idx = randperm(n);
       % else
            perm_idx = [1:n 1:n];
      %  end
           
        if epoch > 0 
            % store full gradient
            full_grad_old = full_grad;

            % compute full gradient
            full_grad = problem.full_grad(w);
  
            % automatic step size selection based on Barzilai-Borwein (BB)
            w_diff = w - w0;
            g_diff = full_grad - full_grad_old;
            step =  ((w_diff' * w_diff) / (w_diff' * g_diff));  
           
            w_m = w0;
            s2 = w_diff'*w_diff;  
            A = ((w_diff' * g_diff)/s2); %% B^m
           % fprintf('step:%f\n', step);
        else
            % compute full gradient
            full_grad = problem.full_grad(w);
        end
        
        % store w
        w0 = w;
        grad_calc_count = grad_calc_count + n;        
        
            
       % if epoch>0 
        
       %end
        
        for j = 1 : num_of_bachces
            
                    if mval==1  % m = 2n and eta_0 = 1,  m=5
                        mv = num_of_bachces;
                        stp = 1;
                        stepn = (stp/mv)*step;

                    elseif mval==2  % m = 2n and constant c = eta_0 (decay), m=6
                        mv = num_of_bachces/2; 
                        %stp = options.stepsizefun(total_iter, options);
                        stp = sto / (1 + sto * regul * total_iter);
                        stepn = (stp/mv)*step;
                        
                    elseif mval==3  % m = 2n and constant c = eta_0, m=7
                        mv = num_of_bachces;
                        stp = step;
                        stepn = (stp/mv)*step;
                        
                    elseif mval==4  % m = 1 and constant c = 1,  m=8
                        mv = 1;
                        stp = 1;
                        stepn = (stp/mv)*step;

                    elseif mval==5  % m = 1 and constant c = eta_0 (decay), m=9
                        mv = 1/2; 
                        stp = sto / (1 + sto * regul * total_iter);
                        %options.stepsizefun(total_iter, options);
                        stepn = (stp/mv)*step;
                        
                    elseif mval==6  % m = 1 and constant c = eta_0, m=10
                        mv = 1;
                        stp = step;
                        stepn = (stp/mv)*step;
                    end

           
            
            % calculate variance reduced gradient
            start_index = (j-1) * options.batch_size + 1;
            indice_j = perm_idx(start_index:start_index+options.batch_size-1);
            grad = problem.grad(w, indice_j);
            grad_0 = problem.grad(w0, indice_j);
            

                if epoch > 0
                grad_m = problem.grad(w_m,indice_j); 
                
                yt_diff = grad_0 - grad_m;% + mu*s_diff;  %x^m - x^m-1
                
                vec = w - w0; % x^k - x^m
                
                Atk = (w_diff' * yt_diff)/s2; %%% a= s'*s/s'*y
                
                %g = problem.grad(w,ind) - problem.grad(w0,ind) + full_grad + problem.hess_vec() + problem.hess_vec();
                bb = A*vec -Atk*vec;
                else
                bb=0;
                end

                    % update w
                    w_var = w; % previouse w to calculate variance
                    v =  (full_grad + grad - grad_0 + bb);
                    w = w - stepn * v;
            
            
            if any(isnan(v)) || any(isinf(v)) || any(isnan(w)) || any(isinf(w)) || optgap>10^5
                   w=inf;
                   fprintf('\n ======== INF/NAN ======== \n')
                 return;   
            end
            
%             % proximal operator
%             if ismethod(problem, 'prox')
%                 w = problem.prox(w, step);
%             end  
%             
            total_iter = total_iter + 1;
        end
        % measure elapsed time
        elapsed_time = toc(start_time);
        
        vr = norm(v-problem.grad(w_var,1:n))^2;
        
        

        
        % count gradient evaluations
        grad_calc_count = grad_calc_count + 2*j * options.batch_size;        
        epoch = epoch + 1;
        
          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        test_cost = problem.test_cost(w);
        val_cost = [val_cost test_cost];
        
        var = [var vr];
        
        p_tr = problem.prediction(w,'Tr');
        acctr = problem.accuracy(p_tr,'Tr');    
        acc_tr = [acc_tr acctr]; 
        
        
        p_vl = problem.prediction(w,'Vl');
        accvl = problem.accuracy(p_vl,'Vl');
        acc_val = [acc_val accvl]; 
        
        lr_rate = [lr_rate stepn];
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


        
        % store infos
        [infos, f_val, optgap] = store_infos(problem, w, options, infos, epoch, grad_calc_count, elapsed_time);          

        % display infos
        if options.verbose > 0
            fprintf('SVRG-2BBNS: Epoch = %03d, cost = %.24e, optgap = %.4e\n', epoch, f_val, optgap);
        end
    end
    
    if optgap < options.tol_optgap
        fprintf('Optimality gap tolerance reached: tol_optgap = %g\n', options.tol_optgap);
    elseif epoch == options.max_epoch
        fprintf('Max epoch reached: max_epochr = %g\n', options.max_epoch);
    end
    
    infos.acc_tr = acc_tr';
    infos.acc_val = acc_val';
    infos.var = var';
    infos.val_cost = val_cost';
    infos.lr_rate = lr_rate';
    

end