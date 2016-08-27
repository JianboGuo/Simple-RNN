%Copyright (C) 2016 by Jianbo Guo, Yibin Liu and Yanhui Li. Date 2016-08-27
%
% 
% Permission is granted for anyone to copy, use, or modify these
% programs and accompanying documents for purposes of research or
% education, provided this copyright notice is retained, and note is
% made of any changes that have been made.
% 
% These programs and documents are distributed without any warranty,
% express or implied.  As the programs were written for research
% purposes only, they have not been tested to the degree that would be
% advisable in any important application.  All use of these programs is
% entirely at the user's own risk.
%



clear; clc; close all;
input_layer_size=20;  % There are 20 metrics for promise set
% hidden_layer_size=?; % see below
num_labels=1;%2-classification problem


nameSeq={'ant','camel', 'ivy','jedit','log4j','lucene', 'poi','velocity', 'xalan', 'xerces'};

for name_i=1:10
    name=nameSeq{name_i};
    fprintf('Data: %s\n',name);
    path=['D:\RNN_globalrand_same_hyperpara+nodes_i7-win10\CODE_Gitub\',name,'\'];
    mkdir(path);
    
    %% -------------------------data------------------------------
    dataTest=importdata(['D:\RNN_globalrand_same_hyperpara+nodes_i7-win10\0809_final_results\DATA_SET\cross_release\',name,'_promise_1.csv']);
    dataTrain=importdata(['D:\RNN_globalrand_same_hyperpara+nodes_i7-win10\0809_final_results\DATA_SET\cross_release\',name,'_promise_0.csv']);
    
    %% ------------------------hyperparameter--------------------------
    hidden_layer_size=2; %hyperpara(1) selected from {2,3,5,7,9};
    lambda=0.1; %hyperpara(2)selected from {0.1,1,2,4,8,14,16};
    iter=100; %hyperpara(3)selected from {100,400,800,1200,1600,2000};
    csvwrite([path,num2str(hidden_layer_size),'-',num2str(lambda),'-',...
        num2str(iter),'.csv'],[]);
    fprintf('hidden_layer_size: %d, lambda: %f, iter: %d\n', hidden_layer_size, lambda, iter);
    
    %% ----------------------random seed-----------------------------
    rng('shuffle'); %seed the random generator
    s=rng;
    save([path,'s.mat'],'-struct','s');

    aucVec=zeros(10,1);%see i=1:10

    %% -------------------10 random initializations------------------------------------------
    for i=1:10 %# of rand
        U=randInitializeWeights(input_layer_size-1,hidden_layer_size);
        b=zeros(hidden_layer_size,1);
        W=randInitializeWeights(hidden_layer_size-1,hidden_layer_size);
        V=randInitializeWeights(hidden_layer_size-1,num_labels);
        c=zeros(num_labels,1);
        params=[U(:);V(:);W(:); b(:); c(:)];

        % fprintf('\n Training reccurent neural network...\n');
        options = optimset('MaxIter', iter); % # of interations, 300-1000

        costFunc=@(p) rnnCostFunction(p, input_layer_size, hidden_layer_size, num_labels,...
                                          dataTrain, lambda);
        % fprintf('Rand %d\n',i);                              
        [rnn_params, cost] = fmincg(costFunc, params, options);

        U_size=hidden_layer_size*input_layer_size;
        V_size=num_labels*hidden_layer_size;
        W_size=hidden_layer_size*hidden_layer_size;
        U=reshape(rnn_params(1:U_size), hidden_layer_size,input_layer_size);
        V=reshape(rnn_params(U_size+1:U_size+V_size), num_labels, hidden_layer_size);
        W=reshape(rnn_params(U_size+V_size+1:U_size+V_size+W_size), hidden_layer_size, hidden_layer_size);
        b=reshape(rnn_params(U_size+V_size+W_size+1:U_size+V_size+W_size+hidden_layer_size), hidden_layer_size,1);
        c=reshape(rnn_params(U_size+V_size+W_size+hidden_layer_size+1:end), num_labels,1);

        [pred, yTest]=predict(U,V,W,b,c,dataTest);
        [tpr, fpr]=roc(yTest', pred');
        [X,Y,T,AUC] = perfcurve(yTest',pred','true');
        aucVec(i)=AUC;

        csvwrite([path,name,'_',num2str(i),'U.csv'],U);
        csvwrite([path,name,'_',num2str(i),'V.csv'],V);
        csvwrite([path,name,'_',num2str(i),'W.csv'],W);
        csvwrite([path,name,'_',num2str(i),'b.csv'],b);
        csvwrite([path,name,'_',num2str(i),'c.csv'],c);
        csvwrite([path,name,'_',num2str(i),'roc.csv'],[tpr;fpr]);
        csvwrite([path,name,'_',num2str(i),'testVScondition.csv'],[pred,yTest]);
    end

    auc=mean(aucVec);
    csvwrite([path,name,'_aucVecAndMean.csv'],[aucVec;auc]);
    
end