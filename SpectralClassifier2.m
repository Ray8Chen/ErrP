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
frequencyRange = [1 3]; % Frequency range of interest



% Extract features for error and correct events
errorFeatures = extractPSDFeatures(errorIndices, eegChannels, Fs, frequencyRange);
correctFeatures = extractPSDFeatures(correctIndices, eegChannels, Fs, frequencyRange);


% Prepare dataset for classification
features = [errorFeatures; correctFeatures];
labels = [zeros(length(errorFeatures), 1); ones(length(correctFeatures), 1)];


% Train the SVM classifier
SVMModel = fitcsvm(features, labels, 'KernelFunction', 'linear', 'Standardize', true, 'ClassNames', [0, 1]);

% Optionally, you can choose a different kernel function like 'rbf' (Radial Basis Function) based on your data characteristics
% SVMModel = fitcsvm(features, labels, 'KernelFunction', 'rbf', 'Standardize', true, 'ClassNames', [0, 1]);


rng(1); % Sets the seed of the MATLAB random number generator for reproducibility

% Set up cross-validated SVM model
CVSVMModel = crossval(SVMModel, 'KFold', 5); % 5-fold cross-validation

% Calculate the classification loss, which is the fraction of misclassifications
classificationError = kfoldLoss(CVSVMModel);
disp(['Classification Error: ', num2str(classificationError)]);

accuracy = 1 - classificationError; % classifier accuracy
disp(['Classification Accuracy: ', num2str(accuracy)]);


% Obtain MCC for SVM classifier

TP = 0; FP = 0; TN = 0; FN = 0;  % True Positives, False Positives, True Negatives, False Negatives

% Access the number of test sets correctly
numTestSets = CVSVMModel.Partition.NumTestSets;

% Loop through each fold to collect predictions and scores
for i = 1:numTestSets
    % Indices for the test set in this fold
    idxTest = CVSVMModel.Partition.test(i);
    
    % True labels for this fold
    trueLabelsFold = labels(idxTest);
    
    % Predicted labels for this fold
    predictedLabelsFold = predict(CVSVMModel.Trained{i}, features(idxTest,:));
    
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
%% 


% Obtain ROC AUC for SVM classifier
% Initialize arrays to store scores and corresponding labels
scores = [];
testLabels = [];


% Access the number of test sets correctly
numTestSets = CVSVMModel.Partition.NumTestSets;

% Loop through each fold to collect predictions and scores
for i = 1:numTestSets
    idxTest = find(CVSVMModel.Partition.test(i));
    [label, score] = predict(CVSVMModel.Trained{i}, features(idxTest,:));
    
    scores = [scores; score(:,2)]; % Append the scores for the positive class
    testLabels = [testLabels; labels(idxTest)];
end


% Calculate ROC curve and AUC
[X, Y, T, AUC] = perfcurve(testLabels, scores, 0); % Here, 0 denotes the positive class (error), 1 denotes the negative class (correct)

% Display AUC
disp(['Cross-Validated AUC: ', num2str(AUC)]);

% Optionally, plot the ROC curve
figure;
plot(X, Y);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curve for k-Fold Cross-Validation');







