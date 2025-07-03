% Originally Created by H. Tankaria
% Last Modified by H. Tankaria on 2023-10-01
% This function finds the best parameters from the saved files in a given path.
% It returns the best parameters based on either accuracy or optimality gap.
function [best] = other_best(path,find_par,regul)
    reg=regul;% 0.1 0.01 0.001 0.0001 0.00001];
    step = [10000 1000 100 10 1 0.1 0.01 0.001 0.0001 0.00001];
   % fprintf('\n \n %s \n \n',path);
%for reg=[1 0.1 0.01 0.001 0.0001 0.00001]   
%for 
%step = [10 5 1 0.1 0.01 0.001 0.0001];
    
    sl = length(step);
    if find_par == 0
        m_acc_best = zeros(sl*1,14);
    else
        m_acc_best = 10^10*ones(sl*1,14);
    end

   % fprintf('size of m_acc_best = %d x %d \n',size(m_acc_best));
    l=1;
        for j=1:1
            for k=1:sl
                Name = strcat(path,sprintf('_%.1e_R_%.1e.mat',step(k),reg(j)));
                if exist(Name,'file')
                  S = load(Name);
                 
                    
                    if find_par == 0 %%% 0 means accuracy
                        [v,ix]=max(mean(S.S1.val_ac(2:end,:),2));
                    else %%%%
                        %[v,ix]=min(mean(S.S1.opt_gap(2:end,:),2));

                        [v,ix]=min(mean(S.S1.ocost(2:end,:),2));
                    end
                    ix=ix+1;
                    m_acc_best(l,:)=[ix,...
                        mean(S.S1.train_ac(ix,:)),std(S.S1.train_ac(ix,:)),...
                        mean(S.S1.val_ac(ix,:)),std(S.S1.val_ac(ix,:)),...
                        mean(S.S1.ocost(ix,:)),std(S.S1.ocost(ix,:)),...
                        mean(S.S1.vcost(ix,:)),std(S.S1.vcost(ix,:)),...
                        S.S1.otime(ix),step(k),reg(j),...
                        mean(S.S1.opt_gap(ix,:)),std(S.S1.opt_gap(ix,:))];
                    l=l+1;
                else
                   % disp(Name);
                end
            end
        end

    if find_par == 0
        [~,idx]=sort(m_acc_best(:,4),'desc');
    else
       % [~,idx]=sort(m_acc_best(:,13));  %%% 13 for optimality gap
        
       [~,idx]=sort(m_acc_best(:,6)); % cost
    end
    
%     if find_par ==0
%     disp([round(m_acc_best(idx(1:10),1)), m_acc_best(idx(1:10), 2:14)]);
%     else
%         
%     disp([round(m_acc_best(idx(1:10),1)), m_acc_best(idx(1:10), 2:14)]);
%     end
    
    best=m_acc_best(idx(1),:);
   % disp(best);
    
%     figure;
%     l=1;
%     for j=1:6
%         for k=1:6
%             a(j,k)=m_acc_best(l,8);
%             l=l+1;
%         end
%     end
%     imagesc(1./a);ylabel("Reg");xlabel("Learning");zlabel("value");
end