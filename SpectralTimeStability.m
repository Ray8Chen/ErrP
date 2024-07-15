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
    eegChannels = filteredData(trainEEGData, 512);  % Adjust the sampling rate if necessary
    eventIds = trainEventIds;

    % Find indices for error and correct events
    errorIndices = find(eventIds == 0);
    correctIndices = find(eventIds == 1);

    % Parameters for spectral analysis
    Fs = 512; % Sampling frequency
    frequencyRange = [4 8]; % Frequency range of interest

    % Extract features for error and correct events
    errorFeatures = extractPSDFeatures(errorIndices, eegChannels, Fs, frequencyRange);
    correctFeatures = extractPSDFeatures(correctIndices, eegChannels, Fs, frequencyRange);

    % Prepare dataset for classification
    features = [errorFeatures; correctFeatures];
    labels = [zeros(length(errorFeatures), 1); ones(length(correctFeatures), 1)];

    % Calculate class frequencies
    numErrorSamples = length(errorFeatures);
    numCorrectSamples = length(correctFeatures);

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
    testEegChannels = filteredData(testEEGData, 512);  % Adjust the sampling rate if necessary
    testEventIds = testEventIds;

    % Find indices for error and correct events in testing data
    testErrorIndices = find(testEventIds == 0);
    testCorrectIndices = find(testEventIds == 1);

    % Extract features for error and correct events in testing data
    testErrorFeatures = extractPSDFeatures(testErrorIndices, testEegChannels, Fs, frequencyRange);
    testCorrectFeatures = extractPSDFeatures(testCorrectIndices, testEegChannels, Fs, frequencyRange);

    % Prepare testing dataset for classification
    testFeatures = [testErrorFeatures; testCorrectFeatures];
    testLabels = [zeros(length(testErrorFeatures), 1); ones(length(testCorrectFeatures), 1)];

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
%     eegChannels = filteredData(trainEEGData, 512);  % Adjust the sampling rate if necessary
%     eventIds = trainEventIds;
% 
%     % Find indices for error and correct events
%     errorIndices = find(eventIds == 0);
%     correctIndices = find(eventIds == 1);
% 
%     % Parameters for spectral analysis
%     Fs = 512; % Sampling frequency
%     frequencyRange = [4 8]; % Frequency range of interest
% 
%     % Extract features for error and correct events
%     errorFeatures = extractPSDFeatures(errorIndices, eegChannels, Fs, frequencyRange);
%     correctFeatures = extractPSDFeatures(correctIndices, eegChannels, Fs, frequencyRange);
% 
%     % Prepare dataset for classification
%     features = [errorFeatures; correctFeatures];
%     labels = [zeros(length(errorFeatures), 1); ones(length(correctFeatures), 1)];
% 
%     % Calculate class frequencies
%     numErrorSamples = length(errorFeatures);
%     numCorrectSamples = length(correctFeatures);
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
%     testEegChannels = filteredData(testEEGData, 512);  % Adjust the sampling rate if necessary
%     testEventIds = testEventIds;
% 
%     % Find indices for error and correct events in testing data
%     testErrorIndices = find(testEventIds == 0);
%     testCorrectIndices = find(testEventIds == 1);
% 
%     % Extract features for error and correct events in testing data
%     testErrorFeatures = extractPSDFeatures(testErrorIndices, testEegChannels, Fs, frequencyRange);
%     testCorrectFeatures = extractPSDFeatures(testCorrectIndices, testEegChannels, Fs, frequencyRange);
% 
%     % Prepare testing dataset for classification
%     testFeatures = [testErrorFeatures; testCorrectFeatures];
%     testLabels = [zeros(length(testErrorFeatures), 1); ones(length(testCorrectFeatures), 1)];
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
