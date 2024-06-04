% Define the path to your directory containing CSV files
directoryPath = '/Users/raychen/Desktop/BCI project/psychopy/data/subject-4';  % Update this with the correct path

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

% Parameters
fs = 512; % Sample rate (Hz) - adjust this as per your data
time_window = [0, 0.8]; % 0 to 0.8 seconds
window_samples = round(time_window * fs);

% band-pass filter EEG data
allEEGData = filteredData(allEEGData, 512);

% Event indices
error_indices = find(allEventIds == 0);
correct_indices = find(allEventIds == 1);

% Extract segments for error and correct events in channel 6 and 15
error_segments = [];
correct_segments = [];

for i = 1:length(error_indices)
    index = error_indices(i);
    if index + window_samples(2) <= size(allEEGData, 1)
        segment = allEEGData(index + window_samples(1):index + window_samples(2), [6, 15]);
        error_segments = [error_segments; segment'];
    end
end

for i = 1:length(correct_indices)
    index = correct_indices(i);
    if index + window_samples(2) <= size(allEEGData, 1)
        segment = allEEGData(index + window_samples(1):index + window_samples(2), [6, 15]);
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

% Display the updated window
fprintf('Updated window_start: %.3f seconds\n', window_start / fs);
fprintf('Updated window_end: %.3f seconds\n', window_end / fs);

% Initialize feature arrays
features_error = zeros(length(error_indices), 2); % Assuming 2 features: mean and variance
features_correct = zeros(length(correct_indices), 2);

% Extract features using electrodes Fz(CH6) (and FCz(CH15))

% Extract features for error events
for i = 1:length(error_indices)
    index = error_indices(i);
    % Ensure the index does not go beyond data length
    if index + window_end <= size(allEEGData, 1)
        % Extract only the 6th electrode
        segment = allEEGData(index + window_start:index + window_end, [6, 15]);
        features_error(i, 1) = mean(segment, 'all'); % Mean of all values in the segment
        features_error(i, 2) = var(segment, 0, 'all'); % Variance of all values
    end
end

% Extract features for correct events
for i = 1:length(correct_indices)
    index = correct_indices(i);
    if index + window_end <= size(allEEGData, 1)
        segment = allEEGData(index + window_start:index + window_end, [6, 15]);
        features_correct(i, 1) = mean(segment, 'all');
        features_correct(i, 2) = var(segment, 0, 'all');
    end
end

% Combine features and labels for classifier
features = [features_error; features_correct];
labels = [zeros(size(features_error, 1), 1); ones(size(features_correct, 1), 1)];

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
    model = fitcdiscr(features(balancedTrainIdx, :), labels(balancedTrainIdx));

    % Train the SVM classifier on the balanced training data
    % model = fitcsvm(features(balancedTrainIdx, :), labels(balancedTrainIdx), 'KernelFunction', 'rbf', 'Standardize', true, 'ClassNames', [0, 1]);

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

% ROC AUC
% Initialize arrays to store the predicted scores and actual labels
scores = [];
trueLabels = [];

% Cross-validation loop
for i = 1:numFolds
    % Training indices for this fold
    trainIdx = cv.training(i);
    % Testing indices for this fold
    testIdx = cv.test(i);
    
    % Train the LDA classifier on the training data
    model = fitcdiscr(features(trainIdx,:), labels(trainIdx));
    
    % Train the SVM classifier on the training data
    % model = fitcsvm(features(balancedTrainIdx, :), labels(balancedTrainIdx), 'KernelFunction', 'rbf', 'Standardize', true, 'ClassNames', [0, 1]);

    % Test the classifier on the testing data and obtain scores
    [~, score] = predict(model, features(testIdx,:));
    
    % Store scores and corresponding labels
    scores = [scores; score(:,2)];  % Assuming the second column contains the positive class scores
    trueLabels = [trueLabels; labels(testIdx)];
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
    % Testing indices for this fold
    testIdx = cv.test(i);
    
    % Train the LDA classifier on the training data
    model = fitcdiscr(features(trainIdx,:), labels(trainIdx));
    
    % Train the SVM classifier on the training data
    % model = fitcsvm(features(balancedTrainIdx, :), labels(balancedTrainIdx), 'KernelFunction', 'rbf', 'Standardize', true, 'ClassNames', [0, 1]);

    % Test the classifier on the testing data
    predictions = predict(model, features(testIdx,:));
    
    % Calculate confusion matrix
    [C,~] = confusionmat(labels(testIdx), predictions);
    
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
