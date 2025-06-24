function [] = DPP_rand_comp()
D = RCV1(1);
X = D.x_train';
y = D.y_train';
P = 10; % Partitions
K = 50; % CLusters
n=size(X,1);
[dppX,dppy,dpp_idxp,dpp_idxn] = DPP(X,y,P,K,1);
H=(X'*X)/n;
res=zeros(P,2);

X1=[];
for i=1:P
    idx=randperm(n,floor(n*i/P));
    
    H1 = X(idx,:)'*X(idx,:);
    H1 = H1/size(idx,2);
    res(i,1)=norm(H-H1,'fro');
    X1=[X1;dppX{i}];
    H2 = X1'*X1;
    H2 = H2/size(X1,1);
    res(i,2)=norm(H-H2,'fro');
end
figure;
plot(res);
xlabel('Amount of Data (%)')
ylabel('||H-H1||_F')
legend('Random','DPP')
end