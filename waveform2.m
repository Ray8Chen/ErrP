% Define the path to your directory containing CSV files
directoryPath = '/Users/raychen/Desktop/BCI project/psychopy/data/subject-4';  % Update this with the correct path

% Get a list of all CSV files in the directory
csvFiles = dir(fullfile(directoryPath, '*.csv'));

% Initialize variables to hold concatenated data
allEEGData = [];
allEventIds = [];

% Loop over each CSV file
for i = 1:length(csvFiles)
    % Get the full path to the current file
    filePath = fullfile(directoryPath, csvFiles(i).name);
    
    % Load the data
    data = readtable(filePath);
    
    % Convert table to array 
    eegData = table2array(data);
    
    % Replace NaN values with -1
    eegData(isnan(eegData)) = -1;
    
    % Extract EEG channel data and Event Id
    eegChannels = eegData(:, 3:18);  % Adjust the indices based on specific columns
    eventIds = eegData(:, 19);
    
    % Concatenate the data
    allEEGData = [allEEGData; eegChannels];  % Append EEG channel data
    allEventIds = [allEventIds; eventIds];   % Append Event Ids
end

% Assume allEEGData is the concatenated data matrix, samplingRate is 512 Hz
% and allEventIds is the concatenated vector of Event Ids

% Step 1: Filter the data
filteredEEG = filteredData(allEEGData, 512);

% Step 2: Extract error/correct epochs; call function 'errorepochs' or
% 'correctepochs'; error epochs marked by eventId == 0, correct epochs
% marked by eventId == 1
Epochs = correctepochs(filteredEEG, allEventIds, 512);

% Step 3: Plot ErrP
plotErrP(Epochs);
