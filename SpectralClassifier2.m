% Define the path to your CSV file
filePath = 'Ziyue-testing-record-[2024.04.18-17.14.08] (1).csv';  % Update this with the correct path

% Load the data
data = readtable(filePath);

% Convert table to array 
eegData = table2array(data);
% eegData = eegData(3:18);


% Replace NaN values with zeros
eegData(isnan(eegData)) = -1;


% Extract EEG channel data and Event Id
eegChannels = filteredData(eegData(:, 3:18), 512);  % Adjust the indices based on specific columns
eventIds = eegData(:, 19);


% Find indices for error and correct events
errorIndices = find(eventIds == 0);
correctIndices = find(eventIds == 1);


%% Plot discriminant power matrix

Fs = 512;  % Sampling frequency
nElectrodes = 16;  % Total number of electrodes
freqRange = 1:15;  % Frequencies from 1 to 15 Hz

% Initialize the discriminant power matrix
discriminantPower = zeros(nElectrodes, length(freqRange));

% Loop through each electrode
for elec = 1:nElectrodes
    % Compute PSD for error and correct events separately
    errorPSD = computePSDForElectrode(errorIndices, eegChannels(:, elec), Fs, freqRange);
    correctPSD = computePSDForElectrode(correctIndices, eegChannels(:, elec), Fs, freqRange);
    
    % Calculate discriminative power, here using simple difference of means
    discriminantPower(elec, :) = mean(errorPSD, 1) - mean(correctPSD, 1);
end

% Plot the discriminant power matrix
figure;
imagesc(freqRange, 1:nElectrodes, discriminantPower);
colorbar;
xlabel('Frequency (Hz)');
ylabel('Electrode Number');
title('Discriminant Power Matrix');
set(gca, 'YDir', 'normal');  % Ensure the y-axis starts from the bottom
%% 


% Parameters for spectral analysis
Fs = 512; % Sampling frequency, change as per your data
frequencyRange = [4 8]; % Frequency range of interest



% Extract features for error and correct events
errorFeatures = extractPSDFeatures(errorIndices, eegChannels, Fs, frequencyRange);
correctFeatures = extractPSDFeatures(correctIndices, eegChannels, Fs, frequencyRange);


% Prepare dataset for classification
features = [errorFeatures; correctFeatures];
labels = [zeros(length(errorFeatures), 1); ones(length(correctFeatures), 1)];

%% LDA

% Load and prepare your data here as previously described
% Assuming 'features' and 'labels' are already defined

rng(1); % Sets the seed of the MATLAB random number generator for reproducibility

% Create a stratified 5-fold cross-validation partition
% cv = cvpartition(labels, 'KFold', 5, 'Stratify', true);
% 
% % Initialize arrays to store performance metrics for each fold
% acc = zeros(cv.NumTestSets, 1);
% auc = zeros(cv.NumTestSets, 1);
% mcc = zeros(cv.NumTestSets, 1);
% 
% % Loop over each fold
% for i = 1:cv.NumTestSets
%     % Indices for training and test set
%     trainIdx = cv.training(i);
%     testIdx = cv.test(i);
% 
%     % Training and test data for this fold
%     XTrain = features(trainIdx, :);
%     YTrain = labels(trainIdx);
%     XTest = features(testIdx, :);
%     YTest = labels(testIdx);
% 
%     % Train the LDA classifier
%     lda = fitcdiscr(XTrain, YTrain);
% 
%     % Predict the test set
%     YPred = predict(lda, XTest);
% 
%     % Predict scores (probabilities) for AUC calculation
%     [~, score] = predict(lda, XTest);
% 
%     % Check if current test set contains less than two classes
%     if length(unique(YTest)) < 2
%         fprintf('Warning: Only one class present in fold %d. Skipping AUC calculation for this fold.\n', i);
%         auc(i) = NaN; % Not enough data to calculate AUC
%     else
%         % Calculate AUC
%         [~,~,~,auc(i)] = perfcurve(YTest, score(:,2), 1);
%     end
% 
%     % Calculate accuracy
%     acc(i) = sum(YPred == YTest) / length(YTest);
% 
%     % Compute confusion matrix elements
%     confMat = confusionmat(YTest, YPred);
%     TP = confMat(2,2); % True Positives
%     TN = confMat(1,1); % True Negatives
%     FP = confMat(1,2); % False Positives
%     FN = confMat(2,1); % False Negatives
% 
%     % Calculate MCC
%     numerator = (TP * TN) - (FP * FN);
%     denominator = sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN));
%     mcc(i) = numerator / denominator;
% 
%     % Check for NaN values in MCC due to zero division
%     if isnan(mcc(i))
%         mcc(i) = 0; % If denominator is zero, MCC is not defined
%     end
% end
% 
% % Display average of metrics across all folds
% fprintf('Average Accuracy: %.2f%%\n', mean(acc, 'omitnan') * 100);
% fprintf('Average AUC: %.2f\n', mean(auc, 'omitnan'));
% fprintf('Average MCC: %.2f\n', mean(mcc, 'omitnan'));






%% 

% % Train the SVM classifier
% SVMModel = fitcsvm(features, labels, 'KernelFunction', 'linear', 'Standardize', true, 'ClassNames', [0, 1]);
% 
% % Optionally, you can choose a different kernel function like 'rbf' (Radial Basis Function) based on your data characteristics
% % SVMModel = fitcsvm(features, labels, 'KernelFunction', 'rbf', 'Standardize', true, 'ClassNames', [0, 1]);
% 
% 
% rng(1); % Sets the seed of the MATLAB random number generator for reproducibility
% 
% % Set up cross-validated SVM model
% CVSVMModel = crossval(SVMModel, 'KFold', 5); % 5-fold cross-validation
% 
% % Calculate the classification loss, which is the fraction of misclassifications
% classificationError = kfoldLoss(CVSVMModel);
% disp(['Classification Error: ', num2str(classificationError)]);
% 
% accuracy = 1 - classificationError; % classifier accuracy
% disp(['Classification Accuracy: ', num2str(accuracy)]);
%% % Balance features in cross validation 
% Parameters
numFolds = 5;
cv = cvpartition(labels, 'KFold', numFolds);

% Initialize arrays to store the performance metrics
accuracy = zeros(numFolds, 1);
errorRate = zeros(numFolds, 1);

% Cross-validation loop
for i = 1:numFolds
    % Training indices for this fold
    trainIdx = cv.training(i);
    % Testing indices for this fold
    testIdx = cv.test(i);
    
    % Balance the training data
    trainErrorIdx = find(labels(trainIdx) == 0);
    trainCorrectIdx = find(labels(trainIdx) == 1);
    
    % Determine the minimum number of samples between the two classes
    minTrainSamples = min(length(trainErrorIdx), length(trainCorrectIdx));
    
    % Randomly select the same number of samples from each class
    balancedTrainIdx = [trainErrorIdx(randperm(length(trainErrorIdx), minTrainSamples)); ...
                        trainCorrectIdx(randperm(length(trainCorrectIdx), minTrainSamples))];
    
    % Train the LDA classifier on the balanced training data
    % model = fitcdiscr(features(balancedTrainIdx, :), labels(balancedTrainIdx));
    
    % Train the SVM classifier on the balanced training data
    model = fitcsvm(features(balancedTrainIdx, :), labels(balancedTrainIdx), 'KernelFunction', 'linear', 'Standardize', true, 'ClassNames', [0, 1]);

    % Test the classifier on the testing data
    predictions = predict(model, features(testIdx, :));
    
    % Calculate accuracy
    correctPredictions = sum(predictions == labels(testIdx));
    accuracy(i) = correctPredictions / length(predictions);
    
    % Calculate error rate
    errorRate(i) = 1 - accuracy(i);
end

% Average performance metrics across all folds
meanAccuracy = mean(accuracy);
meanErrorRate = mean(errorRate);

% Display results
fprintf('Average Accuracy: %.2f%%\n', meanAccuracy * 100);
fprintf('Average Error Rate: %.2f%%\n', meanErrorRate * 100);
%% 

%% Obtain MCC for SVM classifier

TP = 0; FP = 0; TN = 0; FN = 0;  % True Positives, False Positives, True Negatives, False Negatives

% Access the number of test sets correctly
numTestSets = cv.NumTestSets;

% Loop through each fold to collect predictions and scores
for i = 1:numTestSets
    % Indices for the test set in this fold
    idxTest = cv.test(i);

    % True labels for this fold
    trueLabelsFold = labels(idxTest);

    % Predicted labels for this fold
    predictedLabelsFold = predict(model, features(idxTest,:));

    % Compute the confusion matrix for this fold
    C = confusionmat(trueLabelsFold, predictedLabelsFold);

    % Update the counts for TP, FP, TN, FN based on the confusion matrix
    TN = TN + C(1,1);
    FP = FP + C(1,2);
    FN = FN + C(2,1);
    TP = TP + C(2,2);
end

% Calculate MCC
numerator = (TP * TN) - (FP * FN);
denominator = sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN));
MCC = 0;  % Default value in case of division by zero

if denominator ~= 0
    MCC = numerator / denominator;
end

% Display the MCC
disp(['Matthews Correlation Coefficient (MCC): ', num2str(MCC)]);

%% Obtain ROC AUC for SVM classifier

% Initialize arrays to store scores and corresponding labels
scores = [];
testLabels = [];

% Loop through each fold to collect predictions and scores
for i = 1:numTestSets
    idxTest = cv.test(i);
    [label, score] = predict(model, features(idxTest,:));

    scores = [scores; score(:,2)]; % Append the scores for the positive class
    testLabels = [testLabels; labels(idxTest)];
end

% Calculate ROC curve and AUC
[X, Y, T, AUC] = perfcurve(testLabels, scores, 1);

% Display AUC
disp(['Cross-Validated AUC: ', num2str(AUC)]);

% Optionally, plot the ROC curve
figure;
plot(X, Y);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curve for k-Fold Cross-Validation');


% % Obtain MCC for SVM classifier
% 
% TP = 0; FP = 0; TN = 0; FN = 0;  % True Positives, False Positives, True Negatives, False Negatives
% 
% % Access the number of test sets correctly
% numTestSets = CVSVMModel.Partition.NumTestSets;
% 
% % Loop through each fold to collect predictions and scores
% for i = 1:numTestSets
%     % Indices for the test set in this fold
%     idxTest = CVSVMModel.Partition.test(i);
% 
%     % True labels for this fold
%     trueLabelsFold = labels(idxTest);
% 
%     % Predicted labels for this fold
%     predictedLabelsFold = predict(CVSVMModel.Trained{i}, features(idxTest,:));
% 
%     % Compute the confusion matrix for this fold
%     C = confusionmat(trueLabelsFold, predictedLabelsFold);
% 
%     % Update the counts for TP, FP, TN, FN based on the confusion matrix
%     TN = TN + C(1,1);
%     FP = FP + C(1,2);
%     FN = FN + C(2,1);
%     TP = TP + C(2,2);
% end
% 
% % Calculate MCC
% numerator = (TP * TN) - (FP * FN);
% denominator = sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN));
% MCC = 0;  % Default value in case of division by zero
% 
% if denominator ~= 0
%     MCC = numerator / denominator;
% end
% 
% % Display the MCC
% disp(['Matthews Correlation Coefficient (MCC): ', num2str(MCC)]);
% %% 
% 
% 
% % Obtain ROC AUC for SVM classifier
% % Initialize arrays to store scores and corresponding labels
% scores = [];
% testLabels = [];
% 
% 
% % Access the number of test sets correctly
% numTestSets = CVSVMModel.Partition.NumTestSets;
% 
% % Loop through each fold to collect predictions and scores
% for i = 1:numTestSets
%     idxTest = find(CVSVMModel.Partition.test(i));
%     [label, score] = predict(CVSVMModel.Trained{i}, features(idxTest,:));
% 
%     scores = [scores; score(:,2)]; % Append the scores for the positive class
%     testLabels = [testLabels; labels(idxTest)];
% end
% 
% 
% % Calculate ROC curve and AUC
% [X, Y, T, AUC] = perfcurve(testLabels, scores, 0); % Here, 0 denotes the positive class (error), 1 denotes the negative class (correct)
% 
% % Display AUC
% disp(['Cross-Validated AUC: ', num2str(AUC)]);
% 
% % Optionally, plot the ROC curve
% figure;
% plot(X, Y);
% xlabel('False Positive Rate');
% ylabel('True Positive Rate');
% title('ROC Curve for k-Fold Cross-Validation');







