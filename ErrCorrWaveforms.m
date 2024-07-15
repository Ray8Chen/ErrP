% Define the path to your directory containing CSV files
directoryPath = '/Users/raychen/Desktop/BCI project/psychopy/data/subject-13';  % Update this with the correct path

% Get a list of all CSV files in the directory
csvFiles = dir(fullfile(directoryPath, '*.csv'));

% Initialize variables to hold concatenated data
allEEGData = [];
allEventIds = [];
allTime = [];

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
    
    % Extract time, EEG channel data, and Event Id
    time = eegData(:, 1);        % Time points
    eegChannels = eegData(:, 3:18);  % EEG channel data (adjust indices if needed)
    eventIds = eegData(:, 19);    % Event Ids
    
    % Concatenate the data
    allTime = [allTime; time];        % Append time points
    allEEGData = [allEEGData; eegChannels];  % Append EEG channel data
    allEventIds = [allEventIds; eventIds];   % Append Event Ids
end

filteredEEGData = filteredData(allEEGData, 512);
% Extract the columns of interest
channel_6 = filteredEEGData(:, 15); % Channel 6 is the 6th column of eegChannels (since eegChannels starts from column 3)

% Define the time window
pre_event_time = -0.2; % 200 ms before the event
post_event_time = 0.8; % 800 ms after the event

% Sampling rate (assuming it's constant and in Hz, adjust as necessary)
sampling_rate = 512; % Example: 1000 Hz
pre_event_samples = round(pre_event_time * sampling_rate);
post_event_samples = round(post_event_time * sampling_rate);

% Find the indices of error and correct events
error_indices = find(allEventIds == 0);
correct_indices = find(allEventIds == 1);

% Initialize arrays to hold the waveforms
error_waveforms = [];
correct_waveforms = [];

% Extract waveforms around error events
for i = 1:length(error_indices)
    idx = error_indices(i);
    if idx + pre_event_samples > 0 && idx + post_event_samples <= length(allTime)
        error_waveforms = [error_waveforms; channel_6((idx + pre_event_samples):(idx + post_event_samples))'];
    end
end

% Extract waveforms around correct events
for i = 1:length(correct_indices)
    idx = correct_indices(i);
    if idx + pre_event_samples > 0 && idx + post_event_samples <= length(allTime)
        correct_waveforms = [correct_waveforms; channel_6((idx + pre_event_samples):(idx + post_event_samples))'];
    end
end

% Create time vector for the plot
time_vector = linspace(pre_event_time, post_event_time, length(pre_event_samples:post_event_samples));

% Calculate mean and standard deviation for error and correct events
mean_error_waveform = mean(error_waveforms, 1);
std_error_waveform = std(error_waveforms, [], 1);
mean_correct_waveform = mean(correct_waveforms, 1);
std_correct_waveform = std(correct_waveforms, [], 1);


% Plot the waveforms with shaded areas for standard deviation
figure;
hold on;

% Plot mean and std for error events
fill([time_vector fliplr(time_vector)], [mean_error_waveform + std_error_waveform fliplr(mean_error_waveform - std_error_waveform)], 'r', 'FaceAlpha', 0.3, 'EdgeColor', 'none', 'DisplayName', 'Error Events (Std Dev)');
plot(time_vector, mean_error_waveform, 'r', 'LineWidth', 1.5, 'DisplayName', 'Error Events (Mean)');

% Plot mean and std for correct events
fill([time_vector fliplr(time_vector)], [mean_correct_waveform + std_correct_waveform fliplr(mean_correct_waveform - std_correct_waveform)], 'b', 'FaceAlpha', 0.3, 'EdgeColor', 'none', 'DisplayName', 'Correct Events (Std Dev)');
plot(time_vector, mean_correct_waveform, 'b', 'LineWidth', 1.5, 'DisplayName', 'Correct Events (Mean)');

xlabel('Time (s)');
ylabel('Voltage (µV)');
title('EEG Waveforms Around Events');
legend;
hold off;

