% Define the path to your directory containing CSV files
baseDirectoryPath = '/Users/raychen/Desktop/BCI project/psychopy/data';  % Update this with the correct path

% Get a list of all folders
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
    
    % Spectral Feature Extraction for training data
    Fs = 512;  % Sampling frequency
    frequencyRange = [4 8]; % Frequency range of interest
    spectralFeaturesError = extractPSDFeatures(errorIndices, eegChannels, Fs, frequencyRange);
    spectralFeaturesCorrect = extractPSDFeatures(correctIndices, eegChannels, Fs, frequencyRange);

    % Temporal Feature Extraction for training data
    fs = 512; % Sample rate (Hz) - adjust this as per your data
    time_window = [0, 0.8]; % 0 to 0.8 seconds
    window_samples = round(time_window * fs);

    error_segments = [];
    correct_segments = [];

    for i = 1:length(errorIndices)
        index = errorIndices(i);
        if index + window_samples(2) <= size(trainEEGData, 1)
            segment = trainEEGData(index + window_samples(1):index + window_samples(2), [2,3,4,5,6,9,15,16]);
            error_segments = [error_segments; segment'];
        end
    end

    for i = 1:length(correctIndices)
        index = correctIndices(i);
        if index + window_samples(2) <= size(trainEEGData, 1)
            segment = trainEEGData(index + window_samples(1):index + window_samples(2), [2,3,4,5,6,9,15,16]);
            correct_segments = [correct_segments; segment'];
        end
    end

    mean_error_segment = mean(error_segments, 1);
    mean_correct_segment = mean(correct_segments, 1);

    diff_segment = mean_error_segment - mean_correct_segment;
    [~, max_diff_index] = max(diff_segment);

    max_diff_time = max_diff_index / fs;

    window_start = max(0, max_diff_time - 0.005) * fs;
    window_end = min(max_diff_time + 0.005, time_window(2)) * fs;

    features_error = zeros(length(errorIndices), 2);
    features_correct = zeros(length(correctIndices), 2);

    for i = 1:length(errorIndices)
        index = errorIndices(i);
        if index + window_end <= size(trainEEGData, 1)
            segment = trainEEGData(index + window_start:index + window_end, [2,3,4,5,6,9,15,16]);
            features_error(i, 1) = mean(segment, 'all');
            features_error(i, 2) = var(segment, 0, 'all');
        end
    end

    for i = 1:length(correctIndices)
        index = correctIndices(i);
        if index + window_end <= size(trainEEGData, 1)
            segment = trainEEGData(index + window_start:index + window_end, [2,3,4,5,6,9,15,16]);
            features_correct(i, 1) = mean(segment, 'all');
            features_correct(i, 2) = var(segment, 0, 'all');
        end
    end

    temporalFeaturesError = features_error;
    temporalFeaturesCorrect = features_correct;

    if ~isempty(temporalFeaturesError)
        minValError = min(temporalFeaturesError);
        maxValError = max(temporalFeaturesError);
        temporalFeaturesError = (temporalFeaturesError - minValError) ./ (maxValError - minValError);
    end

    if ~isempty(temporalFeaturesCorrect)
        minValCorrect = min(temporalFeaturesCorrect);
        maxValCorrect = max(temporalFeaturesCorrect);
        temporalFeaturesCorrect = (temporalFeaturesCorrect - minValCorrect) ./ (maxValCorrect - minValCorrect);
    end

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

    combinedFeaturesError = [spectralFeaturesError, temporalFeaturesError];
    combinedFeaturesCorrect = [spectralFeaturesCorrect, temporalFeaturesCorrect];

    combinedFeatures = [combinedFeaturesError; combinedFeaturesCorrect];
    labels = [zeros(size(combinedFeaturesError, 1), 1); ones(size(combinedFeaturesCorrect, 1), 1)];

    numErrorSamples = size(combinedFeaturesError, 1);
    numCorrectSamples = size(combinedFeaturesCorrect, 1);

    weightError = 1 / numErrorSamples;
    weightCorrect = 1 / numCorrectSamples;

    totalSamples = numErrorSamples + numCorrectSamples;
    weightError = weightError * totalSamples / 2;
    weightCorrect = weightCorrect * totalSamples / 2;

    weights = zeros(size(labels));
    weights(labels == 0) = weightError;
    weights(labels == 1) = weightCorrect;

    SVMModel = fitcsvm(combinedFeatures, labels, 'KernelFunction', 'rbf', 'Weights', weights);

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

    % Spectral Feature Extraction for testing data
    testSpectralFeaturesError = extractPSDFeatures(testErrorIndices, testEegChannels, Fs, frequencyRange);
    testSpectralFeaturesCorrect = extractPSDFeatures(testCorrectIndices, testEegChannels, Fs, frequencyRange);

    % Temporal Feature Extraction for testing data
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

    testFeaturesError = zeros(length(testErrorIndices), 2);
    testFeaturesCorrect = zeros(length(testCorrectIndices), 2);

    for i = 1:length(testErrorIndices)
        index = testErrorIndices(i);
        if index + window_end <= size(testEEGData, 1)
            segment = testEEGData(index + window_start:index + window_end, [2,3,4,5,6,9,15,16]);
            testFeaturesError(i, 1) = mean(segment, 'all');
            testFeaturesError(i, 2) = var(segment, 0, 'all');
        end
    end

    for i = 1:length(testCorrectIndices)
        index = testCorrectIndices(i);
        if index + window_end <= size(testEEGData, 1)
            segment = testEEGData(index + window_start:index + window_end, [2,3,4,5,6,9,15,16]);
            testFeaturesCorrect(i, 1) = mean(segment, 'all');
            testFeaturesCorrect(i, 2) = var(segment, 0, 'all');
        end
    end

    testTemporalFeaturesError = testFeaturesError;
    testTemporalFeaturesCorrect = testFeaturesCorrect;

    if ~isempty(testTemporalFeaturesError)
        testMinValError = min(testTemporalFeaturesError);
        testMaxValError = max(testTemporalFeaturesError);
        testTemporalFeaturesError = (testTemporalFeaturesError - testMinValError) ./ (testMaxValError - testMinValError);
    end

    if ~isempty(testTemporalFeaturesCorrect)
        testMinValCorrect = min(testTemporalFeaturesCorrect);
        testMaxValCorrect = max(testTemporalFeaturesCorrect);
        testTemporalFeaturesCorrect = (testTemporalFeaturesCorrect - testMinValCorrect) ./ (testMaxValCorrect - testMinValCorrect);
    end

    if size(testSpectralFeaturesError, 1) ~= size(testTemporalFeaturesError, 1)
        minSize = min(size(testSpectralFeaturesError, 1), size(testTemporalFeaturesError, 1));
        testSpectralFeaturesError = testSpectralFeaturesError(1:minSize, :);
        testTemporalFeaturesError = testTemporalFeaturesError(1:minSize, :);
    end

    if size(testSpectralFeaturesCorrect, 1) ~= size(testTemporalFeaturesCorrect, 1)
        minSize = min(size(testSpectralFeaturesCorrect, 1), size(testTemporalFeaturesCorrect, 1));
        testSpectralFeaturesCorrect = testSpectralFeaturesCorrect(1:minSize, :);
        testTemporalFeaturesCorrect = testTemporalFeaturesCorrect(1:minSize, :);
    end

    testCombinedFeaturesError = [testSpectralFeaturesError, testTemporalFeaturesError];
    testCombinedFeaturesCorrect = [testSpectralFeaturesCorrect, testTemporalFeaturesCorrect];

    testCombinedFeatures = [testCombinedFeaturesError; testCombinedFeaturesCorrect];
    testLabels = [zeros(size(testCombinedFeaturesError, 1), 1); ones(size(testCombinedFeaturesCorrect, 1), 1)];

    % Predict and calculate accuracy
    predictions = predict(SVMModel, testCombinedFeatures);
    accuracy = sum(predictions == testLabels) / length(testLabels);
    accuracies(k) = accuracy;
end

% Display average accuracy
averageAccuracy = mean(accuracies);
disp(['Average Accuracy: ', num2str(averageAccuracy)]);


% % Define the path to your directory containing CSV files
% baseDirectoryPath = '/Users/raychen/Desktop/BCI project/psychopy/data';  % Update this with the correct path
% 
% % Get a list of all folders
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
%     % Spectral Feature Extraction for training data
%     Fs = 512;  % Sampling frequency
%     frequencyRange = [4 8]; % Frequency range of interest
%     spectralFeaturesError = extractPSDFeatures(errorIndices, eegChannels, Fs, frequencyRange);
%     spectralFeaturesCorrect = extractPSDFeatures(correctIndices, eegChannels, Fs, frequencyRange);
% 
%     % Temporal Feature Extraction for training data
%     fs = 512; % Sample rate (Hz) - adjust this as per your data
%     time_window = [0, 0.8]; % 0 to 0.8 seconds
%     window_samples = round(time_window * fs);
% 
%     error_segments = [];
%     correct_segments = [];
% 
%     for i = 1:length(errorIndices)
%         index = errorIndices(i);
%         if index + window_samples(2) <= size(trainEEGData, 1)
%             segment = trainEEGData(index + window_samples(1):index + window_samples(2), [2,3,4,5,6,9,15,16]);
%             error_segments = [error_segments; segment'];
%         end
%     end
% 
%     for i = 1:length(correctIndices)
%         index = correctIndices(i);
%         if index + window_samples(2) <= size(trainEEGData, 1)
%             segment = trainEEGData(index + window_samples(1):index + window_samples(2), [2,3,4,5,6,9,15,16]);
%             correct_segments = [correct_segments; segment'];
%         end
%     end
% 
%     mean_error_segment = mean(error_segments, 1);
%     mean_correct_segment = mean(correct_segments, 1);
% 
%     diff_segment = mean_error_segment - mean_correct_segment;
%     [~, max_diff_index] = max(diff_segment);
% 
%     max_diff_time = max_diff_index / fs;
% 
%     window_start = max(0, max_diff_time - 0.005) * fs;
%     window_end = min(max_diff_time + 0.005, time_window(2)) * fs;
% 
%     features_error = zeros(length(errorIndices), 2);
%     features_correct = zeros(length(correctIndices), 2);
% 
%     for i = 1:length(errorIndices)
%         index = errorIndices(i);
%         if index + window_end <= size(trainEEGData, 1)
%             segment = trainEEGData(index + window_start:index + window_end, [2,3,4,5,6,9,15,16]);
%             features_error(i, 1) = mean(segment, 'all');
%             features_error(i, 2) = var(segment, 0, 'all');
%         end
%     end
% 
%     for i = 1:length(correctIndices)
%         index = correctIndices(i);
%         if index + window_end <= size(trainEEGData, 1)
%             segment = trainEEGData(index + window_start:index + window_end, [2,3,4,5,6,9,15,16]);
%             features_correct(i, 1) = mean(segment, 'all');
%             features_correct(i, 2) = var(segment, 0, 'all');
%         end
%     end
% 
%     temporalFeaturesError = features_error;
%     temporalFeaturesCorrect = features_correct;
% 
%     if ~isempty(temporalFeaturesError)
%         minValError = min(temporalFeaturesError);
%         maxValError = max(temporalFeaturesError);
%         temporalFeaturesError = (temporalFeaturesError - minValError) ./ (maxValError - minValError);
%     end
% 
%     if ~isempty(temporalFeaturesCorrect)
%         minValCorrect = min(temporalFeaturesCorrect);
%         maxValCorrect = max(temporalFeaturesCorrect);
%         temporalFeaturesCorrect = (temporalFeaturesCorrect - minValCorrect) ./ (maxValCorrect - minValCorrect);
%     end
% 
%     if size(spectralFeaturesError, 1) ~= size(temporalFeaturesError, 1)
%         minSize = min(size(spectralFeaturesError, 1), size(temporalFeaturesError, 1));
%         spectralFeaturesError = spectralFeaturesError(1:minSize, :);
%         temporalFeaturesError = temporalFeaturesError(1:minSize, :);
%     end
% 
%     if size(spectralFeaturesCorrect, 1) ~= size(temporalFeaturesCorrect, 1)
%         minSize = min(size(spectralFeaturesCorrect, 1), size(temporalFeaturesCorrect, 1));
%         spectralFeaturesCorrect = spectralFeaturesCorrect(1:minSize, :);
%         temporalFeaturesCorrect = temporalFeaturesCorrect(1:minSize, :);
%     end
% 
%     combinedFeaturesError = [spectralFeaturesError, temporalFeaturesError];
%     combinedFeaturesCorrect = [spectralFeaturesCorrect, temporalFeaturesCorrect];
% 
%     combinedFeatures = [combinedFeaturesError; combinedFeaturesCorrect];
%     labels = [zeros(size(combinedFeaturesError, 1), 1); ones(size(combinedFeaturesCorrect, 1), 1)];
% 
%     numErrorSamples = size(combinedFeaturesError, 1);
%     numCorrectSamples = size(combinedFeaturesCorrect, 1);
% 
%     weightError = 1 / numErrorSamples;
%     weightCorrect = 1 / numCorrectSamples;
% 
%     totalSamples = numErrorSamples + numCorrectSamples;
%     weightError = weightError * totalSamples / 2;
%     weightCorrect = weightCorrect * totalSamples / 2;
% 
%     weights = zeros(size(labels));
%     weights(labels == 0) = weightError;
%     weights(labels == 1) = weightCorrect;
% 
%     SVMModel = fitcsvm(combinedFeatures, labels, 'KernelFunction', 'rbf', 'Weights', weights);
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
%     % Spectral Feature Extraction for testing data
%     testSpectralFeaturesError = extractPSDFeatures(testErrorIndices, testEegChannels, Fs, frequencyRange);
%     testSpectralFeaturesCorrect = extractPSDFeatures(testCorrectIndices, testEegChannels, Fs, frequencyRange);
% 
%     % Temporal Feature Extraction for testing data
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
%     testFeaturesError = zeros(length(testErrorIndices), 2);
%     testFeaturesCorrect = zeros(length(testCorrectIndices), 2);
% 
%     for i = 1:length(testErrorIndices)
%         index = testErrorIndices(i);
%         if index + window_end <= size(testEEGData, 1)
%             segment = testEEGData(index + window_start:index + window_end, [2,3,4,5,6,9,15,16]);
%             testFeaturesError(i, 1) = mean(segment, 'all');
%             testFeaturesError(i, 2) = var(segment, 0, 'all');
%         end
%     end
% 
%     for i = 1:length(testCorrectIndices)
%         index = testCorrectIndices(i);
%         if index + window_end <= size(testEEGData, 1)
%             segment = testEEGData(index + window_start:index + window_end, [2,3,4,5,6,9,15,16]);
%             testFeaturesCorrect(i, 1) = mean(segment, 'all');
%             testFeaturesCorrect(i, 2) = var(segment, 0, 'all');
%         end
%     end
% 
%     testTemporalFeaturesError = testFeaturesError;
%     testTemporalFeaturesCorrect = testFeaturesCorrect;
% 
%     if ~isempty(testTemporalFeaturesError)
%         testMinValError = min(testTemporalFeaturesError);
%         testMaxValError = max(testTemporalFeaturesError);
%         testTemporalFeaturesError = (testTemporalFeaturesError - testMinValError) ./ (testMaxValError - testMinValError);
%     end
% 
%     if ~isempty(testTemporalFeaturesCorrect)
%         testMinValCorrect = min(testTemporalFeaturesCorrect);
%         testMaxValCorrect = max(testTemporalFeaturesCorrect);
%         testTemporalFeaturesCorrect = (testTemporalFeaturesCorrect - testMinValCorrect) ./ (testMaxValCorrect - testMinValCorrect);
%     end
% 
%     if size(testSpectralFeaturesError, 1) ~= size(testTemporalFeaturesError, 1)
%         minSize = min(size(testSpectralFeaturesError, 1), size(testTemporalFeaturesError, 1));
%         testSpectralFeaturesError = testSpectralFeaturesError(1:minSize, :);
%         testTemporalFeaturesError = testTemporalFeaturesError(1:minSize, :);
%     end
% 
%     if size(testSpectralFeaturesCorrect, 1) ~= size(testTemporalFeaturesCorrect, 1)
%         minSize = min(size(testSpectralFeaturesCorrect, 1), size(testTemporalFeaturesCorrect, 1));
%         testSpectralFeaturesCorrect = testSpectralFeaturesCorrect(1:minSize, :);
%         testTemporalFeaturesCorrect = testTemporalFeaturesCorrect(1:minSize, :);
%     end
% 
%     testCombinedFeaturesError = [testSpectralFeaturesError, testTemporalFeaturesError];
%     testCombinedFeaturesCorrect = [testSpectralFeaturesCorrect, testTemporalFeaturesCorrect];
% 
%     testCombinedFeatures = [testCombinedFeaturesError; testCombinedFeaturesCorrect];
%     testLabels = [zeros(size(testCombinedFeaturesError, 1), 1); ones(size(testCombinedFeaturesCorrect, 1), 1)];
% 
%     % Predict and calculate accuracy
%     predictions = predict(SVMModel, testCombinedFeatures);
%     accuracy = sum(predictions == testLabels) / length(testLabels);
%     accuracies(k) = accuracy;
% end
% 
% % Display average accuracy
% averageAccuracy = mean(accuracies);
% disp(['Average Accuracy: ', num2str(averageAccuracy)]);

