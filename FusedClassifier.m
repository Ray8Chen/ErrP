% Define the path to directory containing CSV files
directoryPath = '/Users/raychen/Desktop/BCI project/psychopy/data/subject-4'; 

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
    allEEGData = [allEEGData; eegData(:, 3:18)];  
    allEventIds = [allEventIds; eegData(:, 19)];  
end

% Filter the concatenated data
eegChannels = filteredData(allEEGData, 512);  
eventIds = allEventIds;

% Find indices for error and correct events
errorIndices = find(eventIds == 0);
correctIndices = find(eventIds == 1);

%% Spectral Feature Extraction
Fs = 512;  % Sampling frequency
frequencyRange = [4 8]; % Frequency range of interest

% Extract features for error and correct events
spectralFeaturesError = extractPSDFeatures(errorIndices, eegChannels, Fs, frequencyRange);
spectralFeaturesCorrect = extractPSDFeatures(correctIndices, eegChannels, Fs, frequencyRange);


%% Temporal Feature Extraction
% Extract segments for error and correct events in electrodes
% Parameters
fs = 512; % Sample rate (Hz) 
time_window = [0, 0.8]; % 0 to 0.8 seconds
window_samples = round(time_window * fs);

error_segments = [];
correct_segments = [];

for i = 1:length(errorIndices)
    index = errorIndices(i);
    if index + window_samples(2) <= size(allEEGData, 1)
        segment = allEEGData(index + window_samples(1):index + window_samples(2), [2,3,4,5,6,9,15,16]);
        error_segments = [error_segments; segment'];
    end
end

for i = 1:length(correctIndices)
    index = correctIndices(i);
    if index + window_samples(2) <= size(allEEGData, 1)
        segment = allEEGData(index + window_samples(1):index + window_samples(2), [2,3,4,5,6,9,15,16]);
        correct_segments = [correct_segments; segment'];
    end
end

% Compute the average segment for error and correct events
mean_error_segment = mean(error_segments, 1);
mean_correct_segment = mean(correct_segments, 1);

% Find the time of maximum positive difference
diff_segment = mean_error_segment - mean_correct_segment;
[~, max_diff_index] = max(diff_segment);

% Convert index to time
max_diff_time = max_diff_index / fs;

% Update window_start and window_end
window_start = max(0, max_diff_time - 0.005) * fs;
window_end = min(max_diff_time + 0.005, time_window(2)) * fs;

% Initialize feature arrays
features_error = zeros(length(errorIndices), 2); 
features_correct = zeros(length(correctIndices), 2);

% Extract features using electrodes

% Extract features for error events
for i = 1:length(errorIndices)
    index = errorIndices(i);
    % Ensure the index does not go beyond data length
    if index + window_end <= size(allEEGData, 1)
        % Extract only the 6th electrode
        segment = allEEGData(index + window_start:index + window_end, [2,3,4,5,6,9,15,16]);
        features_error(i, 1) = mean(segment, 'all'); % Mean of all values in the segment
        features_error(i, 2) = var(segment, 0, 'all'); % Variance of all values
    end
end

% Extract features for correct events
for i = 1:length(correctIndices)
    index = correctIndices(i);
    if index + window_end <= size(allEEGData, 1)
        segment = allEEGData(index + window_start:index + window_end, [2,3,4,5,6,9,15,16]);
        features_correct(i, 1) = mean(segment, 'all');
        features_correct(i, 2) = var(segment, 0, 'all');
    end
end

% Reshape the temporal features to a single vector
temporalFeaturesError = features_error;
temporalFeaturesCorrect = features_correct;

% Apply min-max scaling to temporalFeaturesError
if ~isempty(temporalFeaturesError)
    minValError = min(temporalFeaturesError);
    maxValError = max(temporalFeaturesError);
    temporalFeaturesError = (temporalFeaturesError - minValError) ./ (maxValError - minValError);
end

% Apply min-max scaling to temporalFeaturesCorrect
if ~isempty(temporalFeaturesCorrect)
    minValCorrect = min(temporalFeaturesCorrect);
    maxValCorrect = max(temporalFeaturesCorrect);
    temporalFeaturesCorrect = (temporalFeaturesCorrect - minValCorrect) ./ (maxValCorrect - minValCorrect);
end

%% Ensure Consistent Dimensions Before Concatenation
% Check dimensions
if size(spectralFeaturesError, 1) ~= size(temporalFeaturesError, 1)
    minSize = min(size(spectralFeaturesError, 1), size(temporalFeaturesError, 1));
    spectralFeaturesError = spectralFeaturesError(1:minSize, :);
    temporalFeaturesError = temporalFeaturesError(1:minSize, :);
end

if size(spectralFeaturesCorrect, 1) ~= size(temporalFeaturesCorrect, 1)
    minSize = min(size(spectralFeaturesCorrect, 1), size(temporalFeaturesCorrect, 1));
    spectralFeaturesCorrect = spectralFeaturesCorrect(1:minSize, :);
    temporalFeaturesCorrect = temporalFeaturesCorrect(1:minSize, :);
end

%% Combine Spectral and Temporal Features
combinedFeaturesError = [spectralFeaturesError, temporalFeaturesError];
combinedFeaturesCorrect = [spectralFeaturesCorrect, temporalFeaturesCorrect];

combinedFeatures = [combinedFeaturesError; combinedFeaturesCorrect];
% Define labels (0 for error and 1 for correct)
labels = [zeros(length(combinedFeaturesError), 1); ones(length(combinedFeaturesCorrect), 1)];

%% %% Classification
% Train a classifier

rng(1); % Sets the seed of the MATLAB random number generator for reproducibility
% Balance features in cross-validation 
% Parameters
numFolds = 8;
cv = cvpartition(labels, 'KFold', numFolds);

% Initialize arrays to store the performance metrics
accuracy = zeros(numFolds, 1);
errorRate = zeros(numFolds, 1);

% Cross-validation loop
for i = 1:numFolds
    % Training indices for this fold
    trainIdx = cv.training(i);
    
    % Balance the training data
    trainErrorIdx = find(labels(trainIdx) == 0);
    trainCorrectIdx = find(labels(trainIdx) == 1);
    
    % Determine the minimum number of samples between the two classes
    minTrainSamples = min(length(trainErrorIdx), length(trainCorrectIdx));
    
    % Randomly select the same number of samples from each class
    balancedTrainIdx = [trainErrorIdx(randperm(length(trainErrorIdx), minTrainSamples)); ...
                        trainCorrectIdx(randperm(length(trainCorrectIdx), minTrainSamples))];
    
    % Ensure labels for balancedTrainIdx are properly defined
    trainLabels = labels(balancedTrainIdx);

    % Train the SVM classifier on the balanced training data
    model = fitcsvm(combinedFeatures(balancedTrainIdx, :), trainLabels, 'KernelFunction', 'rbf', 'Standardize', true, 'ClassNames', [0, 1]);

    % Test the classifier on the entire dataset
    predictions = predict(model, combinedFeatures);
    
    % Calculate accuracy
    correctPredictions = sum(predictions == labels);
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

% ROC AUC
% Initialize arrays to store the predicted scores and actual labels
scores = [];
trueLabels = [];

% Cross-validation loop
for i = 1:numFolds
    % Training indices for this fold
    trainIdx = cv.training(i);
    
    % Balance the training data
    trainErrorIdx = find(labels(trainIdx) == 0);
    trainCorrectIdx = find(labels(trainIdx) == 1);
    
    % Determine the minimum number of samples between the two classes
    minTrainSamples = min(length(trainErrorIdx), length(trainCorrectIdx));
    
    % Randomly select the same number of samples from each class
    balancedTrainIdx = [trainErrorIdx(randperm(length(trainErrorIdx), minTrainSamples)); ...
                        trainCorrectIdx(randperm(length(trainCorrectIdx), minTrainSamples))];
    
    % Ensure labels for balancedTrainIdx are properly defined
    trainLabels = labels(balancedTrainIdx);

    % Train the SVM classifier on the balanced training data
    model = fitcsvm(combinedFeatures(balancedTrainIdx, :), trainLabels, 'KernelFunction', 'rbf', 'Standardize', true, 'ClassNames', [0, 1]);

    % Test the classifier on the entire dataset and obtain scores
    [~, score] = predict(model, combinedFeatures);
    
    % Store scores and corresponding labels
    scores = [scores; score(:,2)];  % Assuming the second column contains the positive class scores
    trueLabels = [trueLabels; labels];
end

% Calculate ROC curve and AUC
[X, Y, T, AUC] = perfcurve(trueLabels, scores, 1);

% Plot ROC curve
figure;
plot(X, Y);
hold on;
plot([0 1], [0 1], 'r--'); % Add a reference line
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(sprintf('ROC Curve (AUC = %.2f)', AUC));
hold off;

% Display AUC
fprintf('Area Under Curve (AUC): %.2f\n', AUC);

% Calculate MCC
% Initialize arrays to store MCC for each fold
mcc = zeros(numFolds, 1);

% Cross-validation loop
for i = 1:numFolds
    % Training indices for this fold
    trainIdx = cv.training(i);
    
    % Balance the training data
    trainErrorIdx = find(labels(trainIdx) == 0);
    trainCorrectIdx = find(labels(trainIdx) == 1);
    
    % Determine the minimum number of samples between the two classes
    minTrainSamples = min(length(trainErrorIdx), length(trainCorrectIdx));
    
    % Randomly select the same number of samples from each class
    balancedTrainIdx = [trainErrorIdx(randperm(length(trainErrorIdx), minTrainSamples)); ...
                        trainCorrectIdx(randperm(length(trainCorrectIdx), minTrainSamples))];
    
    % Ensure labels for balancedTrainIdx are properly defined
    trainLabels = labels(balancedTrainIdx);

    % Train the SVM classifier on the balanced training data
    model = fitcsvm(combinedFeatures(balancedTrainIdx, :), trainLabels, 'KernelFunction', 'rbf', 'Standardize', true, 'ClassNames', [0, 1]);

    % Test the classifier on the entire dataset
    predictions = predict(model, combinedFeatures);
    
    % Calculate confusion matrix
    [C,~] = confusionmat(labels, predictions);
    
    % Extract true positives, false positives, true negatives, and false negatives
    TP = C(2,2);
    TN = C(1,1);
    FP = C(1,2);
    FN = C(2,1);
    
    % Calculate MCC
    numerator = TP * TN - FP * FN;
    denominator = sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN));

    if denominator == 0
        mcc(i) = 0;  % Handle division by zero by setting MCC to 0, which is a conservative neutral value in this context
    else
        mcc(i) = numerator / denominator;
    end
end

% Average MCC across all folds
meanMCC = mean(mcc);

% Display average MCC
fprintf('Average MCC: %.3f\n', meanMCC);



