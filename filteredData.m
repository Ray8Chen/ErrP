function filtereddata = filteredData(data, samplingRate)
    % Design the band-pass filter
    lowCutoff = 1;  % Low frequency cutoff (Hz)
    highCutoff = 10;  % High frequency cutoff (Hz)
    [b, a] = butter(4, [lowCutoff highCutoff] / (samplingRate / 2), 'bandpass');
    
    % Apply the filter
    filtereddata = filtfilt(b, a, data);
end