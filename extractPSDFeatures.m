% Using All EEG Channels, Epoch=1s After Event.

% function psdFeatures = extractPSDFeatures(indices, eegChannels, Fs, freqRange)
%     psdFeatures = [];
%     numSamples = Fs; % Number of samples in 1 second
%     for i = 1:length(indices)
%         % Calculate the start and end indices for 1 second after the event
%         startIndex = indices(i);
%         endIndex = startIndex + numSamples - 1;
% 
%         % Ensure endIndex does not exceed the size of the matrix
%         if endIndex > size(eegChannels, 1)
%             continue; % Skip this iteration if the segment goes beyond the data
%         end
% 
%         % Extract the signal segment for all channels
%         signalSegment = eegChannels(startIndex:endIndex, :);
% 
%         % Average the signal across all channels (if necessary) or handle each channel separately
%         % Here, I assume you want to average across channels for simplicity
%         averagedSignal = mean(signalSegment, 2); % Averages across columns (each column is a channel)
% 
%         % Compute the power spectral density (PSD) for the averaged signal
%         [pxx, f] = pwelch(averagedSignal, [], [], [], Fs);
% 
%         % Keep only the frequencies within the desired range
%         freqMask = f >= freqRange(1) & f <= freqRange(2);
%         psdFeatures = [psdFeatures; sum(pxx(freqMask))];
%     end
%     return
% end



% Using Fz and FCz, Epoch=1s after event
function psdFeatures = extractPSDFeatures(indices, eegChannels, Fs, freqRange)
    psdFeatures = [];
    numSamples = Fs;  % Number of samples in 1 second
    for i = 1:length(indices)
        % Calculate the start and end indices for 1 second after the event
        startIndex = indices(i);
        endIndex = startIndex + numSamples - 1;

        % Ensure endIndex does not exceed the size of the matrix
        if endIndex > size(eegChannels, 1)
            continue;  % Skip this iteration if the segment goes beyond the data
        end

        % Extract the signal segment for channels 6 and 15
        signalSegment = eegChannels(startIndex:endIndex, 6); % last index [,]

        % You can either average the signals from the two channels or handle them separately
        % If you choose to average:
        averagedSignal = mean(signalSegment, 2);  % Averages across the specified channels

        % Compute the power spectral density (PSD) for the averaged signal
        [pxx, f] = pwelch(averagedSignal, [], [], [], Fs);

        % Keep only the frequencies within the desired range
        freqMask = f >= freqRange(1) & f <= freqRange(2);
        psdFeatures = [psdFeatures; sum(pxx(freqMask))];
    end
    return
end

