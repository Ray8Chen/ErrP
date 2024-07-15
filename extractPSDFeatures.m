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

        % Extract the signal segment 
        signalSegment = eegChannels(startIndex:endIndex, [2,3,4,5,6,9,15,16]); % last index [,]

        % Average the signals 
        averagedSignal = mean(signalSegment, 8);  % Averages across the specified channels

        % Compute the power spectral density (PSD) for the averaged signal
        [pxx, f] = pwelch(averagedSignal, [], [], [], Fs);

        % Keep only the frequencies within the desired range
        freqMask = f >= freqRange(1) & f <= freqRange(2);
        psdFeatures = [psdFeatures; sum(pxx(freqMask))];
    end

    
    % Apply min-max scaling to psdFeatures
    if ~isempty(psdFeatures)
        minVal = min(psdFeatures);
        maxVal = max(psdFeatures);
        psdFeatures = (psdFeatures - minVal) / (maxVal - minVal);
    end
    return
end

