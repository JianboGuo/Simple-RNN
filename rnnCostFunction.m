function [J, grad]=rnnCostFunction(params,input_layer_size, hidden_layer_size, num_labels,...
                                  dataTrain, lambda)
%dataTrain store all the features and labels. rnnCostFunction could compute the
%cost and gradient, which would be processed by GD algorithm


%-------------------turn parameter vector to matrix------------
U_size=hidden_layer_size*input_layer_size;
V_size=num_labels*hidden_layer_size;
W_size=hidden_layer_size*hidden_layer_size;
U=reshape(params(1:U_size), hidden_layer_size,input_layer_size);
V=reshape(params(U_size+1:U_size+V_size), num_labels, hidden_layer_size);
W=reshape(params(U_size+V_size+1:U_size+V_size+W_size), hidden_layer_size, hidden_layer_size);
b=reshape(params(U_size+V_size+W_size+1:U_size+V_size+W_size+hidden_layer_size), hidden_layer_size,1);
c=reshape(params(U_size+V_size+W_size+hidden_layer_size+1:end), num_labels,1);

%------------------train rnn----------------------------------
[m, my]=size(dataTrain);
J=0;
nabla_U_L=zeros(hidden_layer_size,input_layer_size);
nabla_V_L=zeros(num_labels,hidden_layer_size);
nabla_W_L=zeros(hidden_layer_size,hidden_layer_size);
nabla_b_L=zeros(hidden_layer_size,1);
nabla_c_L=zeros(num_labels,1);

for i=1:m
    T=dataTrain(i,1);
    %-----obtain 1 sample from dataTrain-----
    X_this_sample=zeros(T, input_layer_size);
    Y_this_sample=zeros(T,1);
    for t=1:T
        X_this_sample(t,:)=dataTrain(i,2+(input_layer_size+1)*(t-1):(input_layer_size+1)*t);
        Y_this_sample(t,1)=dataTrain(i, 1+(input_layer_size+1)*t);
    end
    Y_this_sample=~(Y_this_sample==0);%1 means bug(s)
    s=zeros(hidden_layer_size,T);
    for t=1:T;%Forward
        if t==1
            s(:,1)=tanh(U*(X_this_sample(1,:)')+b);%hidden layer states are stored by column
        else
            s(:,t)=tanh(U*(X_this_sample(t,:)')+W*s(:,t-1)+b);
        end
    end
    o=V*s(:,T)+c;
    p=sigmoid(o);
    L=-Y_this_sample(T)*log(p)-(1-Y_this_sample(T))*log(1-p);
    J=J+L;%compute J
    nabla_s_L=zeros(hidden_layer_size,T);
    
    nabla_o_L=Y_this_sample(T)*(p-1)+(1-Y_this_sample(T))*p;
    nabla_s_L(:,T)=V'*nabla_o_L;
    for t=T-1:-1:1%backpropogation
        nabla_s_L(:,t)=W'*diag(1-s(:,t+1).^2)*nabla_s_L(:,t+1);
    end
    nabla_c_L=nabla_c_L+nabla_o_L;%compute the gradient
    nabla_V_L=nabla_V_L+nabla_o_L*(s(:,T)');
    for t=1:T
        nabla_b_L=nabla_b_L+diag(1-s(:,t).^2)*nabla_s_L(:,t);
        nabla_U_L=nabla_U_L+diag(1-s(:,t).^2)*nabla_s_L(:,t)*X_this_sample(t,:);
    end
    for t=2:T
        nabla_W_L=nabla_W_L+diag(1-s(:,t).^2)*nabla_s_L(:,t)*(s(:,t-1)');
    end
end
J=1/m*J+lambda/(2*m)*(sum(sum(U.^2))+sum(sum(V.^2))+sum(sum(W.^2)));
nabla_b_L=1/m*nabla_b_L;
nabla_c_L=1/m*nabla_c_L;
nabla_U_L=1/m*nabla_U_L+lambda/m*U;
nabla_V_L=1/m*nabla_V_L+lambda/m*V;
nabla_W_L=1/m*nabla_W_L+lambda/m*W;
grad=[nabla_U_L(:);nabla_V_L(:);nabla_W_L(:);nabla_b_L(:);nabla_c_L(:)];

end
