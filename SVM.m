clear
clc
close all;

%% loading data
load data_training.csv
TrainingSample=data_training(:,2:end);
TrainingLabel=data_training(:,1);

load data_testing.csv
TestingSample=data_testing(:,2:end);
TestingLabel=data_testing(:,1);

%% svm
clear DM; clear numClass; 

numClass = max(TestingLabel);
DM = zeros((numClass+1), size(TestingSample,1));

Sigma = 0.5;
C = 0.5;

for i = 0:numClass
    for j = i+1:numClass
        
        if(i == 9)
            break;
        end
            
        intij = (TrainingLabel == i) | (TrainingLabel == j);
        TrainingSampleij = TrainingSample(intij, :);
        TrainingLabelij = TrainingLabel(intij, :);
                
        %training 訓練
        svmStruct = svmtrain(TrainingSampleij, TrainingLabelij, 'showplot', 0, 'kernel_function', 'rbf', 'rbf_sigma', Sigma, 'boxConstraint', C);

        OutLabel = svmclassify(svmStruct, TestingSample, 'showplot', 0); % 計算結果
        
        rowIn = OutLabel;
        colIn = (1:1:size(TestingSample,1))';
           
        IinearInd = sub2ind([(numClass+1), size(TestingSample,1)], rowIn, colIn);
        
        DM(IinearInd) = DM(IinearInd) + 1;   
    end
end

%% Performance
[~, OutLabel] = max(DM);
acc = mean(grp2idx(OutLabel) == grp2idx(TestingLabel)) %把類別用數字表示 , sum(把裡面加起來)

