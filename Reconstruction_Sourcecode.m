%% Spectrum reconstruction from responsivity matrix and photocurrent matrix
% Adapted from Yoon et al. (https://doi.org/10.1126/science.add8544)
% This script reconstructs optical spectra from measured photocurrents using 
% Tikhonov regularization and Generalized Cross-Validation (GCV).
clear; clc; close all;

%% ===================== USER SETTINGS =====================
% Define input file names containing the system's characterization and raw data
cfg.responseFile         = 'ResponsivityMatrix.csv';
cfg.measuredCurrentFile  = 'MeasuredSignals.csv';
cfg.measuredSpectrumFile = 'SpectrumMatrix.csv';

% Reconstruction resolution settings
cfg.numGaussianBasis     = 121;  % Number of Gaussian functions used to build the spectrum
cfg.numPlotPoints        = 1010; % High-resolution grid points for smooth plotting

% Candidate FWHM (Full Width at Half Maximum) pools for the Gaussian basis functions.
% Different light sources require differently shaped basis functions for optimal reconstruction.
cfg.fwhmNarrowPool       = 3:1:15;   % Best for monochromatic sources (e.g., lasers)
cfg.fwhmMidPool          = 20:5:80;  % Best for LEDs or narrow-broad sources
cfg.fwhmBroadPool        = 85:5:200; % Best for true broadband sources (e.g., sunlight, halogen)
cfg.fwhmCandidates       = [cfg.fwhmNarrowPool, cfg.fwhmMidPool, cfg.fwhmBroadPool];

% Diagnostic classifier settings
% A quick "first pass" is done using a fixed FWHM to guess the type of light source
cfg.diagnosticFWHM       = 10; % FWHM used for the initial diagnostic pass
% Thresholds (in nm) to classify the standard deviation of the preliminary spectrum
cfg.stdMidThreshold      = 10; 
cfg.stdBroadThreshold    = 40; 

% Boundary padding for Gaussian centers
% Adds padding beyond the measured wavelength range to prevent edge artifacts
cfg.padNm                = 50; 

% GCV (Generalized Cross-Validation) settings for finding the optimal regularization parameter
cfg.gcvLowerBound        = 1e-12; 
cfg.gcvUpperMult         = 1e2;   

% Interpolation settings for upsampling data
cfg.responseInterpMethod = 'pchip'; % Piecewise cubic Hermite interpolating polynomial preserves shape
cfg.plotInterpMethod     = 'pchip';

% Output file names for the results
cfg.outputReconSpectrumHiRes = 'reconstructed_spectrum_matrix_hires.csv';
cfg.outputSimulatedCurrent   = 'simulated_current_matrix.csv';
cfg.outputSummary            = 'reconstruction_summary.csv';

%% ===================== LOAD INPUT DATA =====================
responseMatrix         = readNumericCsv(cfg.responseFile);
measuredCurrentMatrix  = readNumericCsv(cfg.measuredCurrentFile);
measuredSpectrumMatrix = readNumericCsv(cfg.measuredSpectrumFile);

% Ground truth wavelength range (e.g., 400nm to 900nm in 5nm steps)
wavelengthNm           = 400:5:900; 

% Normalize the responsivity matrix to a maximum of 1 to ensure numerical stability 
% during matrix inversions and SVD calculations later.
responseMatrix = normalizeToMax(responseMatrix);

%% ===================== VALIDATE INPUTS =====================
% Ensure matrix dimensions match the physical system setup to prevent runtime crashes
validateInputs(responseMatrix, measuredCurrentMatrix, measuredSpectrumMatrix, wavelengthNm);

numWavelengths = numel(wavelengthNm);
numVoltages    = size(responseMatrix, 1);    % Number of sensor channels/voltages
numSamples     = size(measuredCurrentMatrix, 2); % Number of distinct light sources to reconstruct

%% ===================== BUILD WAVELENGTH GRIDS =====================
% Normalize wavelengths (0 to 1) to prevent numerical scaling issues in Gaussian calculations
lambdaScale = max(wavelengthNm);
wavelengthNorm     = wavelengthNm / lambdaScale;

% Create a high-resolution wavelength grid for smooth interpolation and plotting
wavelengthPlotNm   = linspace(min(wavelengthNm), max(wavelengthNm), cfg.numPlotPoints).';
wavelengthPlotNorm = wavelengthPlotNm / lambdaScale;

%% ===================== INTERPOLATE RESPONSIVITY =====================
% Upsample the physical responsivity matrix to match the high-resolution plotting grid
responseMatrixHiRes = interpolateRows( ...
    responseMatrix, ...
    wavelengthNorm, ...
    wavelengthPlotNorm, ...
    cfg.responseInterpMethod);
numRowsForward = size(responseMatrixHiRes, 1);

%% ===================== BUILD GAUSSIAN CENTERS =====================
% Distribute the centers of the Gaussian basis functions evenly across the wavelength range,
% including the padded boundaries to handle signals near 400nm or 900nm smoothly.
minCenterNm = min(wavelengthNm) - cfg.padNm;
maxCenterNm = max(wavelengthNm) + cfg.padNm;

gaussianCentersNm   = linspace(minCenterNm, maxCenterNm, cfg.numGaussianBasis).';
gaussianCentersNorm = gaussianCentersNm / lambdaScale;

%% ===================== REGULARIZATION MATRIX =====================
% Create a second-difference operator (Laplacian) matrix.
% This is used in Tikhonov regularization to penalize "roughness" (rapid changes) 
% in the reconstructed spectrum, forcing the solution to be physically smooth.
e = ones(cfg.numGaussianBasis, 1);
regularizationMatrix = spdiags([e, -2*e, e], 0:2, cfg.numGaussianBasis - 2, cfg.numGaussianBasis);

% Suppress command window output from the least-squares solver
lsqOptions = optimset('Display', 'off');

%% ===================== PRECOMPUTE MODELS FOR ALL FWHMs =====================
% To save time during the reconstruction loop, we precompute the Forward matrices 
% and their Singular Value Decompositions (SVD) for every candidate FWHM.
numFWHM = numel(cfg.fwhmCandidates);

% Preallocate structure to hold precomputed models
models(numFWHM) = struct( ...
    'fwhm', [], ...
    'sigma', [], ...
    'basisHiRes', [], ...
    'forward', [], ...
    'U', [], ...
    'singularValues', []);

for k = 1:numFWHM
    sigma = fwhmToSigma(cfg.fwhmCandidates(k));
    
    % Build the dictionary of Gaussian curves
    basisHiRes = buildGaussianBasis(wavelengthPlotNorm, gaussianCentersNorm, sigma);
    
    % Forward matrix (A) maps spectrum coefficients to expected sensor currents
    forwardMat = responseMatrixHiRes * basisHiRes;
    
    % Precompute SVD (Economy size). This drastically speeds up the GCV calculation later.
    [U, S, ~]  = svd(forwardMat, 'econ');
    
    models(k).fwhm           = cfg.fwhmCandidates(k);
    models(k).sigma          = sigma;
    models(k).basisHiRes     = basisHiRes;
    models(k).forward        = forwardMat;
    models(k).U              = U;
    models(k).singularValues = diag(S);
end

% Group the indices so we can easily search only the relevant subset later
candidateIdx.narrow = find(ismember(cfg.fwhmCandidates, cfg.fwhmNarrowPool));
candidateIdx.mid    = find(ismember(cfg.fwhmCandidates, cfg.fwhmMidPool));
candidateIdx.broad  = find(ismember(cfg.fwhmCandidates, cfg.fwhmBroadPool));

%% ===================== PRECOMPUTE DIAGNOSTIC MODEL =====================
% Compute a specific model used only for the first diagnostic pass
sigmaDiag = fwhmToSigma(cfg.diagnosticFWHM);
basisDiag = buildGaussianBasis(wavelengthPlotNorm, gaussianCentersNorm, sigmaDiag);
fwDiag    = responseMatrixHiRes * basisDiag;
[UDiag, SDiag, ~] = svd(fwDiag, 'econ');
singValsDiag = diag(SDiag);

%% ===================== PREALLOCATE OUTPUTS =====================
% Preallocating arrays prevents MATLAB from dynamically resizing them in the loop,
% which improves performance and memory management.
reconstructedSpectrumHiRes = zeros(cfg.numPlotPoints, numSamples);
simulatedCurrentMatrix     = zeros(numVoltages, numSamples);
bestFWHMUsed          = zeros(numSamples, 1);
bestGammaUsed         = zeros(numSamples, 1);
measuredStds          = zeros(numSamples, 1);
dataMismatchNorm2     = zeros(numSamples, 1);
roughnessNorm2        = zeros(numSamples, 1);
totalObjective        = zeros(numSamples, 1);
sourceClassification  = strings(numSamples, 1);

%% ===================== RECONSTRUCT EACH SAMPLE =====================
% Change "for" to "parfor" if you want parallel execution.
for sampleIdx = 1:numSamples
    
    % Extract and normalize the measurement vector for the current sample
    currentMeasured = normalizeToMax(measuredCurrentMatrix(:, sampleIdx));
    
    % -------------------------------------------------
    % PASS 1: DIAGNOSTIC RECONSTRUCTION FOR CLASSIFICATION
    % -------------------------------------------------
    % Goal: Do a quick, rough reconstruction to determine if the light is 
    % a laser, an LED, or a broad lamp.
    
    % Find optimal regularization parameter (gamma) for diagnostic pass
    gammaDiag = findOptimalGammaGCV(currentMeasured, UDiag, singValsDiag, numRowsForward, cfg);
    
    % Augment matrices for Tikhonov regularization (combines data fitting and smoothing)
    augmentedADiag = [
        fwDiag
        gammaDiag * regularizationMatrix
    ];
    augmentedBDiag = [
        currentMeasured
        zeros(size(regularizationMatrix, 1), 1)
    ];
    
    % Solve Non-Negative Least Squares (coefficients cannot be negative, as light can't be negative)
    coeffsDiag = lsqnonneg(augmentedADiag, augmentedBDiag, lsqOptions);
    trialSpectrum = basisDiag * coeffsDiag;
    
    % Calculate the standard deviation of the preliminary spectrum
    trialStdNm = measureSpectrumStd(trialSpectrum, wavelengthPlotNm);
    measuredStds(sampleIdx) = trialStdNm;
    
    % Classify the source based on spectral width and select the appropriate FWHM pool
    if trialStdNm < cfg.stdMidThreshold
        poolIdx = candidateIdx.narrow;
        sourceClassification(sampleIdx) = "Monochromatic";
    elseif trialStdNm < cfg.stdBroadThreshold
        poolIdx = candidateIdx.mid;
        sourceClassification(sampleIdx) = "LED / Narrow Broad";
    else
        poolIdx = candidateIdx.broad;
        sourceClassification(sampleIdx) = "True Broadband";
    end
    
    % -------------------------------------------------
    % PASS 2: FINAL RECONSTRUCTION IN SELECTED POOL
    % -------------------------------------------------
    % Goal: Loop only through the FWHM candidates that match our classification
    % to find the absolute best fit.
    
    bestCost        = inf;
    bestCoeffs      = [];
    bestModelIdx    = poolIdx(1);
    bestGammaLocal  = NaN;
    bestMismatch    = NaN;
    bestRoughness   = NaN;
    
    for idx = poolIdx
        model = models(idx);
        
        % Find the best smoothing parameter for this specific FWHM
        gamma = findOptimalGammaGCV( ...
            currentMeasured, ...
            model.U, ...
            model.singularValues, ...
            numRowsForward, ...
            cfg);
            
        % Set up the regularized least squares problem
        augmentedA = [
            model.forward
            gamma * regularizationMatrix
        ];
        augmentedB = [
            currentMeasured
            zeros(size(regularizationMatrix, 1), 1)
        ];
        
        % Solve for coefficients
        coeffs = lsqnonneg(augmentedA, augmentedB, lsqOptions);
        
        % Calculate how well our reconstructed spectrum recreates the measured current
        simCurrent   = model.forward * coeffs;
        mismatch     = norm(simCurrent - currentMeasured)^2; % Data fidelity term
        roughness    = norm(regularizationMatrix * coeffs)^2; % Smoothness penalty term
        
        % Total objective cost function (we want to minimize this)
        totalCost    = mismatch + (gamma^2) * roughness;
        
        % If this FWHM gives a better cost, save it as the current best
        if totalCost < bestCost
            bestCost       = totalCost;
            bestCoeffs     = coeffs;
            bestModelIdx   = idx;
            bestGammaLocal = gamma;
            bestMismatch   = mismatch;
            bestRoughness  = roughness;
        end
    end
    
    % Finalize the best model for this sample
    bestModel = models(bestModelIdx);
    
    % Reconstruct the final high-resolution spectrum using the best coefficients
    reconstructedSpectrumHiRes(:, sampleIdx) = normalizeToMax(bestModel.basisHiRes * bestCoeffs);
    simulatedCurrentMatrix(:, sampleIdx)     = bestModel.forward * bestCoeffs;
    
    % Store metadata for the summary report
    bestFWHMUsed(sampleIdx)      = bestModel.fwhm;
    bestGammaUsed(sampleIdx)     = bestGammaLocal;
    dataMismatchNorm2(sampleIdx) = bestMismatch;
    roughnessNorm2(sampleIdx)    = bestRoughness;
    totalObjective(sampleIdx)    = bestCost;
end

%% ===================== SAVE OUTPUTS =====================
% Write matrices to CSV for downstream analysis
writematrix(reconstructedSpectrumHiRes, cfg.outputReconSpectrumHiRes);
writematrix(simulatedCurrentMatrix,     cfg.outputSimulatedCurrent);

% Create and save a comprehensive summary table
summaryTable = table( ...
    (1:numSamples).', ...
    measuredStds, ...
    sourceClassification, ...
    bestFWHMUsed, ...
    bestGammaUsed, ...
    dataMismatchNorm2, ...
    roughnessNorm2, ...
    totalObjective, ...
    'VariableNames', { ...
        'Sample', ...
        'TrialSpectrumStdNm', ...
        'Classification', ...
        'SelectedFWHM', ...
        'SelectedGamma', ...
        'DataMismatchNorm2', ...
        'RoughnessNorm2', ...
        'TotalObjective'});
writetable(summaryTable, cfg.outputSummary);

%% ===================== PLOT COMPARISON =====================
% Visually compare the reconstructed spectra against the ground-truth measurements
plotSpectrumComparison( ...
    wavelengthNm, ...
    measuredSpectrumMatrix, ...
    wavelengthPlotNm, ...
    reconstructedSpectrumHiRes, ...
    sourceClassification, ...
    bestFWHMUsed, ...
    cfg.plotInterpMethod);

%% ===================== DISPLAY SUMMARY =====================
% Print a formatted summary to the command window
fprintf('\n--- RECONSTRUCTION SUMMARY ---\n');
for i = 1:numSamples
    fprintf(['Sample %2d | Std = %6.1f nm | Class = %-18s | FWHM = %3d | ' ...
             'Gamma = %.3e | Cost = %.3e\n'], ...
        i, measuredStds(i), sourceClassification(i), bestFWHMUsed(i), bestGammaUsed(i), totalObjective(i));
end

fprintf('\nSaved files:\n');
fprintf('  %s\n', cfg.outputReconSpectrumHiRes);
fprintf('  %s\n', cfg.outputSimulatedCurrent);
fprintf('  %s\n', cfg.outputSummary);


%% ===================== LOCAL FUNCTIONS =====================

% Helper to safely read a numeric CSV, throwing descriptive errors if it fails
function X = readNumericCsv(filename)
    if ~isfile(filename)
        error('File not found: %s', filename);
    end
    X = readmatrix(filename);
    if isempty(X) || ~isnumeric(X)
        error('File %s is empty or not numeric.', filename);
    end
    if any(~isfinite(X(:)))
        error('File %s contains NaN or Inf values.', filename);
    end
end

% Helper to load a specific column vector from a .mat file
function x = readMatColumnVector(filename, variableName)
    if ~isfile(filename)
        error('File not found: %s', filename);
    end
    s = load(filename, variableName);
    if ~isfield(s, variableName)
        error('Variable "%s" not found in %s.', variableName, filename);
    end
    x = s.(variableName);
    x = x(:);
    if isempty(x) || ~isnumeric(x)
        error('Variable "%s" in %s must be numeric.', variableName, filename);
    end
    if any(~isfinite(x(:)))
        error('Variable "%s" in %s contains NaN or Inf.', variableName, filename);
    end
end

% Validates that all input matrices align with physical dimensions
function validateInputs(responseMatrix, measuredCurrentMatrix, measuredSpectrumMatrix, wavelengthNm)
    numWavelengths = numel(wavelengthNm);
    numVoltages    = size(responseMatrix, 1);
    if size(responseMatrix, 2) ~= numWavelengths
        error('Responsivity matrix must have %d columns, but has %d.', ...
            numWavelengths, size(responseMatrix, 2));
    end
    if size(measuredCurrentMatrix, 1) ~= numVoltages
        error('Measured current matrix must have %d rows, but has %d.', ...
            numVoltages, size(measuredCurrentMatrix, 1));
    end
    if size(measuredSpectrumMatrix, 1) ~= numWavelengths
        error('Measured spectrum matrix must have %d rows, but has %d.', ...
            numWavelengths, size(measuredSpectrumMatrix, 1));
    end
    if size(measuredSpectrumMatrix, 2) ~= size(measuredCurrentMatrix, 2)
        error('Measured spectrum matrix and measured current matrix must have the same number of columns.');
    end
end

% Converts Full Width at Half Maximum (FWHM) to standard deviation (sigma)
% The division by 1000 scales it to match the normalized wavelength grids
function sigma = fwhmToSigma(fwhmValue)
    sigma = fwhmValue / 1000 / (2 * sqrt(2 * log(2)));
end

% Interpolates rows of a matrix from an old grid to a new grid
function Xout = interpolateRows(Xin, xOld, xNew, method)
    numRows = size(Xin, 1);
    Xout = zeros(numRows, numel(xNew));
    for rowIdx = 1:numRows
        Xout(rowIdx, :) = interp1(xOld, Xin(rowIdx, :), xNew, method);
    end
end

% Constructs a dictionary matrix of Gaussian curves
% Each column is a Gaussian centered at a specific wavelength
function basis = buildGaussianBasis(wavelengthGridNorm, centersNorm, sigma)
    delta = wavelengthGridNorm - centersNorm.';
    basis = exp(-0.5 * (delta ./ sigma).^2);
    basis = basis / (sigma * sqrt(2 * pi));
end

% Normalizes a vector/matrix by its absolute maximum value
function x = normalizeToMax(x)
    maxValue = max(abs(x(:)));
    if maxValue > 0
        x = x / maxValue;
    end
end

% Wraps the GCV objective function in fminbnd to find the gamma that minimizes the GCV score.
% Operates in log-space to handle the vastly different scales gamma can take.
function gamma = findOptimalGammaGCV(signalVector, U, singularValues, numRowsForward, cfg)
    lowerBound = cfg.gcvLowerBound;
    upperBound = max([singularValues; 1]) * cfg.gcvUpperMult;
    
    objective = @(logGamma) computeGCV( ...
        exp(logGamma), U, singularValues, signalVector, numRowsForward);
        
    gamma = exp(fminbnd(objective, log(lowerBound), log(upperBound)));
end

% Calculates the Generalized Cross-Validation score.
% GCV provides a mathematical way to find a regularization parameter (gamma) 
% that balances fitting the data closely vs avoiding overfitting to noise.
function gcvValue = computeGCV(gamma, U, singularValues, b, numRowsForward)
    beta = U' * b;
    filterFactor = singularValues.^2 ./ (singularValues.^2 + gamma.^2);
    residualNorm = sum(((1 - filterFactor) .* beta).^2);
    gcvValue = residualNorm / (numRowsForward - sum(filterFactor))^2;
end

% Estimates the standard deviation (width) of a given spectrum.
% Used to classify if the source is monochromatic, LED, or broad.
function stdNm = measureSpectrumStd(spectrum, wavelengthNm)
    % Baseline subtraction
    s = spectrum - min(spectrum(:));
    
    % Zero out noise floor (anything below 1% of max peak)
    if max(s) > 0
        s(s < 0.01 * max(s)) = 0;
    end
    
    if sum(s) == 0
        stdNm = 0;
        return;
    end
    
    % Treat the spectrum like a probability density function to find mean and std
    sPdf = s / sum(s);
    meanLambda = sum(wavelengthNm .* sPdf);
    stdNm = sqrt(sum(((wavelengthNm - meanLambda).^2) .* sPdf));
end

% Generates a tiled figure comparing ground truth vs reconstructed spectra
function plotSpectrumComparison(wavelengthNm, measuredSpectrumMatrix, wavelengthPlotNm, reconstructedSpectrumHiRes, classifications, bestFWHMUsed, interpMethod)
    numSamples = size(measuredSpectrumMatrix, 2);
    numCols = min(4, numSamples); % Max 4 columns
    numRows = ceil(numSamples / numCols);
    
    figure('Color', 'white', 'Position', [50 50 1400 900]);
    tiledlayout(numRows, numCols, 'TileSpacing', 'compact', 'Padding', 'compact');
    
    for sampleIdx = 1:numSamples
        nexttile;
        
        % Interpolate measured spectrum to high-res grid for an apples-to-apples plot
        measuredInterp = interp1( ...
            wavelengthNm, ...
            measuredSpectrumMatrix(:, sampleIdx), ...
            wavelengthPlotNm, ...
            interpMethod);
            
        measuredInterp = normalizeToMax(measuredInterp);
        reconstructed  = normalizeToMax(reconstructedSpectrumHiRes(:, sampleIdx));
        
        plot(wavelengthPlotNm, measuredInterp, 'k-',  'LineWidth', 1.4); hold on;
        plot(wavelengthPlotNm, reconstructed,  'r--', 'LineWidth', 1.4);
        
        title(sprintf('Sample %d | %s | FWHM %d', ...
            sampleIdx, classifications(sampleIdx), bestFWHMUsed(sampleIdx)), ...
            'FontSize', 10);
        xlabel('\lambda (nm)');
        ylabel('a.u.');
        xlim([min(wavelengthNm), max(wavelengthNm)]);
        grid on;
        
        if sampleIdx == 1
            legend('Measured', 'Reconstructed', 'Location', 'best');
        end
    end
    sgtitle('Measured vs Reconstructed Spectra');
end