
%reg=[ 1 0.1 0.01 0.001 0.0001];
%step = [ 1 0.1 0.01 0.001 0.0001 0.00001];
path='W8A/';
i=1;
R = [];
% R = char(36,1);
    for reg = [ 1 0.1 0.01 0.001 0.0001 0.00001]
        for step = [ 1 0.1 0.01 0.001 0.0001 0.00001]
            Name = strcat(path,sprintf('svrgbb_%.1e_R_%.1e.mat',step,reg));
            if exist(Name,'file')
            R = [R reg];
            d=load(Name);
            plot(d.S1.opt_gap);
 %           legend(sprintf('R%.1eS%.1e',reg,step));
            hold on;
            i=i+1;
            else
                disp(Name);
            end
        end
    end
            disp(R)
            legend(string(R));
            lgd = legend;
            lgd.FontSize=14;
            lgd.NumColumns = 4;
