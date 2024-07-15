% Define the path to your base directory containing subdirectories
baseDirectoryPath = '/Users/raychen/Desktop/BCI project/psychopy/data';  % Update this with the correct path

% Get a list of all folders in the base directory
folderInfo = dir(baseDirectoryPath);
folderInfo = folderInfo([folderInfo.isdir] & ~ismember({folderInfo.name}, {'.', '..'}));
folderNames = {folderInfo.name};

% Initialize variables for performance metrics
accuracies = zeros(nchoosek(length(folderNames), 4), 1);

rng(1); % Sets the seed of the MATLAB random number generator for reproducibility

% Loop over all combinations of 4 folders for training and 1 folder for testing
combinations = nchoosek(1:length(folderNames), 4);
for k = 1:size(combinations, 1)
    % Select folders for training
    trainFolders = combinations(k, :);
    testFolders = setdiff(1:length(folderNames), trainFolders);
    
    % Initialize variables to hold concatenated data for training
    trainEEGData = [];
    trainEventIds = [];
    
    % Load and process training data
    for i = trainFolders
        folderPath = fullfile(baseDirectoryPath, folderNames{i});
        csvFiles = dir(fullfile(folderPath, '*.csv'));
        for j = 1:length(csvFiles)
            filePath = fullfile(folderPath, csvFiles(j).name);
            data = readtable(filePath);
            eegData = table2array(data);
            eegData(isnan(eegData)) = -1;
            trainEEGData = [trainEEGData; eegData(:, 3:18)];
            trainEventIds = [trainEventIds; eegData(:, 19)];
        end
    end
    
    % Filter the training data
    trainEEGData = filteredData(trainEEGData, 512);

    % Event indices
    error_indices = find(trainEventIds == 0);
    correct_indices = find(trainEventIds == 1);

    % Parameters
    fs = 512; % Sample rate (Hz)
    time_window = [0, 0.8]; % 0 to 0.8 seconds
    window_samples = round(time_window * fs);

    % Extract segments for error and correct events in channel 6 and 15
    error_segments = [];
    correct_segments = [];

    for i = 1:length(error_indices)
        index = error_indices(i);
        if index + window_samples(2) <= size(trainEEGData, 1)
            segment = trainEEGData(index + window_samples(1):index + window_samples(2), [2,3,4,5,6,9,15,16]);
            error_segments = [error_segments; segment'];
        end
    end

    for i = 1:length(correct_indices)
        index = correct_indices(i);
        if index + window_samples(2) <= size(trainEEGData, 1)
            segment = trainEEGData(index + window_samples(1):index + window_samples(2), [2,3,4,5,6,9,15,16]);
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
    features_error = zeros(length(error_indices), 2); % Assuming 2 features: mean and variance
    features_correct = zeros(length(correct_indices), 2);

    % Extract features for error events
    for i = 1:length(error_indices)
        index = error_indices(i);
        if index + window_end <= size(trainEEGData, 1)
            segment = trainEEGData(index + window_start:index + window_end, [2,3,4,5,6,9,15,16]);
            features_error(i, 1) = mean(segment, 'all');
            features_error(i, 2) = var(segment, 0, 'all');
        end
    end

    % Extract features for correct events
    for i = 1:length(correct_indices)
        index = correct_indices(i);
        if index + window_end <= size(trainEEGData, 1)
            segment = trainEEGData(index + window_start:index + window_end, [2,3,4,5,6,9,15,16]);
            features_correct(i, 1) = mean(segment, 'all');
            features_correct(i, 2) = var(segment, 0, 'all');
        end
    end

    % Apply min-max scaling to features_error
    if ~isempty(features_error)
        minValError = min(features_error);
        maxValError = max(features_error);
        features_error = (features_error - minValError) ./ (maxValError - minValError);
    end

    % Apply min-max scaling to features_correct
    if ~isempty(features_correct)
        minValCorrect = min(features_correct);
        maxValCorrect = max(features_correct);
        features_correct = (features_correct - minValCorrect) ./ (maxValCorrect - minValCorrect);
    end

    % Combine features and labels for classifier
    features = [features_error; features_correct];
    labels = [zeros(size(features_error, 1), 1); ones(size(features_correct, 1), 1)];

    % Calculate class frequencies
    numErrorSamples = size(features_error, 1);
    numCorrectSamples = size(features_correct, 1);

    % Compute weights (inverse of class frequencies)
    weightError = 1 / numErrorSamples;
    weightCorrect = 1 / numCorrectSamples;

    % Normalize weights (optional)
    totalSamples = numErrorSamples + numCorrectSamples;
    weightError = weightError * totalSamples / 2;
    weightCorrect = weightCorrect * totalSamples / 2;

    % Assign weights to the samples
    weights = zeros(size(labels));
    weights(labels == 0) = weightError;
    weights(labels == 1) = weightCorrect;

    % Fit SVM model with RBF kernel and class weights
    SVMModel = fitcsvm(features, labels, 'KernelFunction', 'rbf', 'Weights', weights);

    % Initialize variables to hold concatenated data for testing
    testEEGData = [];
    testEventIds = [];
    
    % Load and process testing data
    for i = testFolders
        folderPath = fullfile(baseDirectoryPath, folderNames{i});
        csvFiles = dir(fullfile(folderPath, '*.csv'));
        for j = 1:length(csvFiles)
            filePath = fullfile(folderPath, csvFiles(j).name);
            data = readtable(filePath);
            eegData = table2array(data);
            eegData(isnan(eegData)) = -1;
            testEEGData = [testEEGData; eegData(:, 3:18)];
            testEventIds = [testEventIds; eegData(:, 19)];
        end
    end
    
    % Filter the testing data
    testEEGData = filteredData(testEEGData, 512);

    % Event indices for testing data
    testErrorIndices = find(testEventIds == 0);
    testCorrectIndices = find(testEventIds == 1);

    % Extract segments for error and correct events in testing data
    testErrorSegments = [];
    testCorrectSegments = [];

    for i = 1:length(testErrorIndices)
        index = testErrorIndices(i);
        if index + window_samples(2) <= size(testEEGData, 1)
            segment = testEEGData(index + window_samples(1):index + window_samples(2), [2,3,4,5,6,9,15,16]);
            testErrorSegments = [testErrorSegments; segment'];
        end
    end

    for i = 1:length(testCorrectIndices)
        index = testCorrectIndices(i);
        if index + window_samples(2) <= size(testEEGData, 1)
            segment = testEEGData(index + window_samples(1):index + window_samples(2), [2,3,4,5,6,9,15,16]);
            testCorrectSegments = [testCorrectSegments; segment'];
        end
    end

    % Initialize feature arrays for testing data
    testFeaturesError = zeros(length(testErrorIndices), 2); 
    testFeaturesCorrect = zeros(length(testCorrectIndices), 2);

    % Extract features for error events in testing data
    for i = 1:length(testErrorIndices)
        index = testErrorIndices(i);
        if index + window_end <= size(testEEGData, 1)
            segment = testEEGData(index + window_start:index + window_end, [2,3,4,5,6,9,15,16]);
            testFeaturesError(i, 1) = mean(segment, 'all');
            testFeaturesError(i, 2) = var(segment, 0, 'all');
        end
    end

    % Extract features for correct events in testing data
    for i = 1:length(testCorrectIndices)
        index = testCorrectIndices(i);
        if index + window_end <= size(testEEGData, 1)
            segment = testEEGData(index + window_start:index + window_end, [2,3,4,5,6,9,15,16]);
            testFeaturesCorrect(i, 1) = mean(segment, 'all');
            testFeaturesCorrect(i, 2) = var(segment, 0, 'all');
        end
    end

    % Apply min-max scaling to testFeaturesError
    if ~isempty(testFeaturesError)
        testMinValError = min(testFeaturesError);
        testMaxValError = max(testFeaturesError);
        testFeaturesError = (testFeaturesError - testMinValError) ./ (testMaxValError - testMinValError);
    end

    % Apply min-max scaling to testFeaturesCorrect
    if ~isempty(testFeaturesCorrect)
        testMinValCorrect = min(testFeaturesCorrect);
        testMaxValCorrect = max(testFeaturesCorrect);
        testFeaturesCorrect = (testFeaturesCorrect - testMinValCorrect) ./ (testMaxValCorrect - testMinValCorrect);
    end

    % Combine features and labels for testing data
    testFeatures = [testFeaturesError; testFeaturesCorrect];
    testLabels = [zeros(size(testFeaturesError, 1), 1); ones(size(testFeaturesCorrect, 1), 1)];

    % Predict and calculate accuracy
    predictions = predict(SVMModel, testFeatures);
    accuracy = sum(predictions == testLabels) / length(testLabels);
    accuracies(k) = accuracy;
end

% Display average accuracy
averageAccuracy = mean(accuracies);
disp(['Average Accuracy: ', num2str(averageAccuracy)]);


% % Define the path to your base directory containing subdirectories
% baseDirectoryPath = '/Users/raychen/Desktop/BCI project/psychopy/data';  % Update this with the correct path
% 
% % Get a list of all folders in the base directory
% folderInfo = dir(baseDirectoryPath);
% folderInfo = folderInfo([folderInfo.isdir] & ~ismember({folderInfo.name}, {'.', '..'}));
% folderNames = {folderInfo.name};
% 
% % Initialize variables for performance metrics
% accuracies = zeros(nchoosek(length(folderNames), 3), 1);
% 
% rng(1); % Sets the seed of the MATLAB random number generator for reproducibility
% 
% % Loop over all combinations of 3 folders for training and 2 folders for testing
% combinations = nchoosek(1:length(folderNames), 3);
% for k = 1:size(combinations, 1)
%     % Select folders for training
%     trainFolders = combinations(k, :);
%     testFolders = setdiff(1:length(folderNames), trainFolders);
% 
%     % Initialize variables to hold concatenated data for training
%     trainEEGData = [];
%     trainEventIds = [];
% 
%     % Load and process training data
%     for i = trainFolders
%         folderPath = fullfile(baseDirectoryPath, folderNames{i});
%         csvFiles = dir(fullfile(folderPath, '*.csv'));
%         for j = 1:length(csvFiles)
%             filePath = fullfile(folderPath, csvFiles(j).name);
%             data = readtable(filePath);
%             eegData = table2array(data);
%             eegData(isnan(eegData)) = -1;
%             trainEEGData = [trainEEGData; eegData(:, 3:18)];
%             trainEventIds = [trainEventIds; eegData(:, 19)];
%         end
%     end
% 
%     % Filter the training data
%     trainEEGData = filteredData(trainEEGData, 512);
% 
%     % Event indices
%     error_indices = find(trainEventIds == 0);
%     correct_indices = find(trainEventIds == 1);
% 
%     % Parameters
%     fs = 512; % Sample rate (Hz)
%     time_window = [0, 0.8]; % 0 to 0.8 seconds
%     window_samples = round(time_window * fs);
% 
%     % Extract segments for error and correct events in channel 6 and 15
%     error_segments = [];
%     correct_segments = [];
% 
%     for i = 1:length(error_indices)
%         index = error_indices(i);
%         if index + window_samples(2) <= size(trainEEGData, 1)
%             segment = trainEEGData(index + window_samples(1):index + window_samples(2), [2,3,4,5,6,9,15,16]);
%             error_segments = [error_segments; segment'];
%         end
%     end
% 
%     for i = 1:length(correct_indices)
%         index = correct_indices(i);
%         if index + window_samples(2) <= size(trainEEGData, 1)
%             segment = trainEEGData(index + window_samples(1):index + window_samples(2), [2,3,4,5,6,9,15,16]);
%             correct_segments = [correct_segments; segment'];
%         end
%     end
% 
%     % Compute the average segment for error and correct events
%     mean_error_segment = mean(error_segments, 1);
%     mean_correct_segment = mean(correct_segments, 1);
% 
%     % Find the time of maximum positive difference
%     diff_segment = mean_error_segment - mean_correct_segment;
%     [~, max_diff_index] = max(diff_segment);
% 
%     % Convert index to time
%     max_diff_time = max_diff_index / fs;
% 
%     % Update window_start and window_end
%     window_start = max(0, max_diff_time - 0.005) * fs;
%     window_end = min(max_diff_time + 0.005, time_window(2)) * fs;
% 
%     % Initialize feature arrays
%     features_error = zeros(length(error_indices), 2); % Assuming 2 features: mean and variance
%     features_correct = zeros(length(correct_indices), 2);
% 
%     % Extract features for error events
%     for i = 1:length(error_indices)
%         index = error_indices(i);
%         if index + window_end <= size(trainEEGData, 1)
%             segment = trainEEGData(index + window_start:index + window_end, [2,3,4,5,6,9,15,16]);
%             features_error(i, 1) = mean(segment, 'all');
%             features_error(i, 2) = var(segment, 0, 'all');
%         end
%     end
% 
%     % Extract features for correct events
%     for i = 1:length(correct_indices)
%         index = correct_indices(i);
%         if index + window_end <= size(trainEEGData, 1)
%             segment = trainEEGData(index + window_start:index + window_end, [2,3,4,5,6,9,15,16]);
%             features_correct(i, 1) = mean(segment, 'all');
%             features_correct(i, 2) = var(segment, 0, 'all');
%         end
%     end
% 
%     % Apply min-max scaling to features_error
%     if ~isempty(features_error)
%         minValError = min(features_error);
%         maxValError = max(features_error);
%         features_error = (features_error - minValError) ./ (maxValError - minValError);
%     end
% 
%     % Apply min-max scaling to features_correct
%     if ~isempty(features_correct)
%         minValCorrect = min(features_correct);
%         maxValCorrect = max(features_correct);
%         features_correct = (features_correct - minValCorrect) ./ (maxValCorrect - minValCorrect);
%     end
% 
%     % Combine features and labels for classifier
%     features = [features_error; features_correct];
%     labels = [zeros(size(features_error, 1), 1); ones(size(features_correct, 1), 1)];
% 
%     % Calculate class frequencies
%     numErrorSamples = size(features_error, 1);
%     numCorrectSamples = size(features_correct, 1);
% 
%     % Compute weights (inverse of class frequencies)
%     weightError = 1 / numErrorSamples;
%     weightCorrect = 1 / numCorrectSamples;
% 
%     % Normalize weights (optional)
%     totalSamples = numErrorSamples + numCorrectSamples;
%     weightError = weightError * totalSamples / 2;
%     weightCorrect = weightCorrect * totalSamples / 2;
% 
%     % Assign weights to the samples
%     weights = zeros(size(labels));
%     weights(labels == 0) = weightError;
%     weights(labels == 1) = weightCorrect;
% 
%     % Fit SVM model with RBF kernel and class weights
%     SVMModel = fitcsvm(features, labels, 'KernelFunction', 'rbf', 'Weights', weights);
% 
%     % Initialize variables to hold concatenated data for testing
%     testEEGData = [];
%     testEventIds = [];
% 
%     % Load and process testing data
%     for i = testFolders
%         folderPath = fullfile(baseDirectoryPath, folderNames{i});
%         csvFiles = dir(fullfile(folderPath, '*.csv'));
%         for j = 1:length(csvFiles)
%             filePath = fullfile(folderPath, csvFiles(j).name);
%             data = readtable(filePath);
%             eegData = table2array(data);
%             eegData(isnan(eegData)) = -1;
%             testEEGData = [testEEGData; eegData(:, 3:18)];
%             testEventIds = [testEventIds; eegData(:, 19)];
%         end
%     end
% 
%     % Filter the testing data
%     testEEGData = filteredData(testEEGData, 512);
% 
%     % Event indices for testing data
%     testErrorIndices = find(testEventIds == 0);
%     testCorrectIndices = find(testEventIds == 1);
% 
%     % Extract segments for error and correct events in testing data
%     testErrorSegments = [];
%     testCorrectSegments = [];
% 
%     for i = 1:length(testErrorIndices)
%         index = testErrorIndices(i);
%         if index + window_samples(2) <= size(testEEGData, 1)
%             segment = testEEGData(index + window_samples(1):index + window_samples(2), [2,3,4,5,6,9,15,16]);
%             testErrorSegments = [testErrorSegments; segment'];
%         end
%     end
% 
%     for i = 1:length(testCorrectIndices)
%         index = testCorrectIndices(i);
%         if index + window_samples(2) <= size(testEEGData, 1)
%             segment = testEEGData(index + window_samples(1):index + window_samples(2), [2,3,4,5,6,9,15,16]);
%             testCorrectSegments = [testCorrectSegments; segment'];
%         end
%     end
% 
%     % Initialize feature arrays for testing data
%     testFeaturesError = zeros(length(testErrorIndices), 2); 
%     testFeaturesCorrect = zeros(length(testCorrectIndices), 2);
% 
%     % Extract features for error events in testing data
%     for i = 1:length(testErrorIndices)
%         index = testErrorIndices(i);
%         if index + window_end <= size(testEEGData, 1)
%             segment = testEEGData(index + window_start:index + window_end, [2,3,4,5,6,9,15,16]);
%             testFeaturesError(i, 1) = mean(segment, 'all');
%             testFeaturesError(i, 2) = var(segment, 0, 'all');
%         end
%     end
% 
%     % Extract features for correct events in testing data
%     for i = 1:length(testCorrectIndices)
%         index = testCorrectIndices(i);
%         if index + window_end <= size(testEEGData, 1)
%             segment = testEEGData(index + window_start:index + window_end, [2,3,4,5,6,9,15,16]);
%             testFeaturesCorrect(i, 1) = mean(segment, 'all');
%             testFeaturesCorrect(i, 2) = var(segment, 0, 'all');
%         end
%     end
% 
%     % Apply min-max scaling to testFeaturesError
%     if ~isempty(testFeaturesError)
%         testMinValError = min(testFeaturesError);
%         testMaxValError = max(testFeaturesError);
%         testFeaturesError = (testFeaturesError - testMinValError) ./ (testMaxValError - testMinValError);
%     end
% 
%     % Apply min-max scaling to testFeaturesCorrect
%     if ~isempty(testFeaturesCorrect)
%         testMinValCorrect = min(testFeaturesCorrect);
%         testMaxValCorrect = max(testFeaturesCorrect);
%         testFeaturesCorrect = (testFeaturesCorrect - testMinValCorrect) ./ (testMaxValCorrect - testMinValCorrect);
%     end
% 
%     % Combine features and labels for testing data
%     testFeatures = [testFeaturesError; testFeaturesCorrect];
%     testLabels = [zeros(size(testFeaturesError, 1), 1); ones(size(testFeaturesCorrect, 1), 1)];
% 
%     % Predict and calculate accuracy
%     predictions = predict(SVMModel, testFeatures);
%     accuracy = sum(predictions == testLabels) / length(testLabels);
%     accuracies(k) = accuracy;
% end
% 
% % Display average accuracy
% averageAccuracy = mean(accuracies);
% disp(['Average Accuracy: ', num2str(averageAccuracy)]);
