% Define the path to your directory containing CSV files
directoryPath = '/Users/raychen/Desktop/BCI project/psychopy/data/subject-13';  % Update this with the correct path

% Get a list of all CSV file names in the directory
csvFiles = {dir(fullfile(directoryPath, '*.csv')).name};

% Initialize variables to hold concatenated data
allEEGData = [];
allEventIds = [];

% Loop over each CSV file
for i = 1:length(csvFiles)
    % Get the full path to the current file
    filePath = fullfile(directoryPath, csvFiles{i});
    
    % Load the data
    data = readtable(filePath);
    
    % Convert table to array 
    eegData = table2array(data);
    
    % Replace NaN values with zeros (or a placeholder value like -1)
    eegData(isnan(eegData)) = -1;
    
    % Concatenate data from all files
    allEEGData = [allEEGData; eegData(:, 3:18)];  % Assuming columns 3 to 18 are EEG channels
    allEventIds = [allEventIds; eegData(:, 19)];  % Assuming column 19 is Event Ids
end

% Filter the concatenated data
eegChannels = filteredData(allEEGData, 512);  % Adjust the sampling rate if necessary
eventIds = allEventIds;

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

% % Plot the discriminant power matrix

% Define the electrode labels
electrodeLabels = {'FC3', 'C1', 'FC1', 'Cz', 'CPz', 'Fz', 'CP3', 'CP4', 'C2', 'FC4', 'C4', 'C3', 'CP1', 'CP2', 'FCz', 'FC2'};

% Plot the discriminant power matrix
figure;
imagesc(freqRange, 1:nElectrodes, discriminantPower);
h = colorbar;  % Create colorbar and store its handle
xlabel('Frequency (Hz)');
ylabel('Electrode');
title('Discriminant Power Matrix');
set(gca, 'YDir', 'normal');  % Ensure the y-axis starts from the bottom

% Set the y-axis tick labels
set(gca, 'YTick', 1:nElectrodes, 'YTickLabel', electrodeLabels);

% Add unit to the colorbar
ylabel(h, 'uV^2');



%% Parameters for spectral analysis
frequencyRange = [4 8]; % Frequency range of interest

% Extract features for error and correct events
errorFeatures = extractPSDFeatures(errorIndices, eegChannels, Fs, frequencyRange);
correctFeatures = extractPSDFeatures(correctIndices, eegChannels, Fs, frequencyRange);

% Prepare dataset for classification
features = [errorFeatures; correctFeatures];
labels = [zeros(length(errorFeatures), 1); ones(length(correctFeatures), 1)];

%% 

rng(1); % Sets the seed of the MATLAB random number generator for reproducibility

% Balance features in cross-validation 
numFolds = 8;
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
    model = fitcsvm(features(balancedTrainIdx, :), labels(balancedTrainIdx), 'KernelFunction', 'rbf', 'Standardize', true, 'ClassNames', [0, 1]);

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
