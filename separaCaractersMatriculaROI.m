function [chars, boxes, info] = separaCaractersMatriculaROI(roiInput, params)

    if nargin < 2 || isempty(params)
        params = defaultParams();
    else
        params = mergeParams(defaultParams(), params);
    end

    if ischar(roiInput) || isstring(roiInput)
        roiOriginal = imread(roiInput);
    else
        roiOriginal = roiInput;
    end

    if params.useAjustROI
        [roiAjustada, ajustInfo] = ajustaROIMatriculaConservadorV1(roiOriginal, params.ajustROI);
    else
        roiAjustada = roiOriginal;
        ajustInfo = struct();
        ajustInfo.didCrop = false;
        ajustInfo.reason = "Ajust desactivat";
        ajustInfo.cropBox = [1 1 size(roiOriginal,2) size(roiOriginal,1)];
    end

    [chars, boxes, dbg] = separaCaractersMatriculaInternal(roiAjustada, params);

    info = dbg;
    info.roiOriginal = roiOriginal;
    info.roiAjustada = roiAjustada;
    info.infoAjust = ajustInfo;
end

function params = defaultParams()
    params = struct();

    params.useAjustROI = true;
    params.ajustROI = struct();
    params.ajustROI.debug = false;

    params.targetHeight = 80;
    params.sensitivity = 0.55;
    params.minAreaFrac = 0.00015;
    params.closeSize = [2 2];
    params.rowThreshFrac = 0.035;
    params.bandPad = 5;

    params.minCharHeightFrac = 0.18;
    params.maxCharHeightFrac = 1.10;
    params.minCharWidthFrac = 0.004;
    params.maxCharWidthFrac = 0.45;
    params.minCharAreaFracBand = 0.0007;
    params.minCharDensity = 0.015;

    params.hLineLenFrac = 0.18;
    params.hLineMinLenPx = 20;
    params.hLineDilateWidth = 3;
    params.hLineDilateHeight = 2;

    params.verticalCloseFrac = 0.12;
    params.colProjThreshFrac = 0.12;
    params.maxGapCols = 2;
    params.splitWideFactor = 1.9;
    params.minWideRunWidth = 10;
    params.valleyFrac = 0.20;

    params.cropPadFrac = 0.010;
    params.charCanvas = [48 32];
    params.connectivity = 8;

    params.maxCharsAbansFiltreAlcada = 7;
    params.minCharsDespresFiltreAlcada = 4;

    params.heightCloseVerticalRel = 0.04;
    params.heightMinRelMedian = 0.55;
    params.heightMaxRelMedian = 1.65;

    params.heightMinFracBand = 0.22;
    params.heightMaxFracBand = 0.98;

    params.heightMinAreaFracBand = 0.0008;
    params.heightMinWidthFracBand = 0.004;
    params.heightMaxWidthFracBand = 0.45;
    params.heightMinDensity = 0.01;

    params.postBlobFilterIfMoreThan = 7;
    params.postBlobMinCharsAfter = 4;
    params.postBlobMinRelMedian = 0.45;
    params.postBlobMinHeightPx = 8;
end

function [chars, boxes, dbg] = separaCaractersMatriculaInternal(inputImage, params)

    if nargin < 2 || isempty(params)
        params = defaultParams();
    end

    if ischar(inputImage) || isstring(inputImage)
        I0 = imread(inputImage);
    else
        I0 = inputImage;
    end

    scale = params.targetHeight / size(I0,1);
    I = imresize(I0, scale);

    if size(I,3) == 3
        gray0 = rgb2gray(I);
    else
        gray0 = I;
    end

    gray0 = im2uint8(gray0);

    if exist('adapthisteq', 'file')
        gray = adapthisteq(gray0, 'ClipLimit', 0.02);
    else
        gray = gray0;
    end

    gray = medfilt2(gray, [3 3]);

    [bwRaw, binDbg] = binaritzaCaractersRobust(gray, params);
    bwClean = netejaBinariCaracters(bwRaw, params);

    rowProj = sum(bwClean, 2);

    if max(rowProj) > 0
        rows = find(rowProj > params.rowThreshFrac * max(rowProj));
    else
        rows = [];
    end

    if isempty(rows)
        y1 = 1;
        y2 = size(bwClean,1);
    else
        y1 = max(1, rows(1) - params.bandPad);
        y2 = min(size(bwClean,1), rows(end) + params.bandPad);
    end

    bwBand = bwClean(y1:y2, :);

    [BW2, lineDbg] = eliminaLiniesHoritzontalsMorfologiques(bwBand, params);

    H = size(bwBand, 1);
    W = size(bwBand, 2);

    BWdetail = BW2;

    lineLen = max(3, round(params.verticalCloseFrac * H));
    BWproj = imclose(BW2, strel('line', lineLen, 90));

    colProj = sum(BWproj, 1);

    if max(colProj) > 0
        thrCol = max(1, params.colProjThreshFrac * max(colProj));
        activeCols = colProj >= thrCol;
    else
        activeCols = false(1, W);
    end

    runs = findRuns1D(activeCols);
    runs = mergeRuns1D(runs, params.maxGapCols);
    runs = splitWideRunsByValleys(runs, colProj, params);

    boxesBand = [];

    for i = 1:size(runs, 1)

        x1 = max(1, runs(i, 1));
        x2 = min(W, runs(i, 2));

        patchDetail = BWdetail(:, x1:x2);
        [yy, ~] = find(patchDetail);

        if isempty(yy)
            patchProj = BWproj(:, x1:x2);
            [yy, ~] = find(patchProj);

            if isempty(yy)
                continue;
            end

            patchForStats = patchProj;
        else
            patchForStats = patchDetail;
        end

        y1b = min(yy);
        y2b = max(yy);

        bw = x2 - x1 + 1;
        bh = y2b - y1b + 1;

        area = nnz(patchForStats);
        density = area / max(numel(patchForStats), eps);

        isTallEnough = bh >= params.minCharHeightFrac * H;
        isNotTooTall = bh <= params.maxCharHeightFrac * H;
        isWideEnough = bw >= params.minCharWidthFrac * W;
        isNotTooWide = bw <= params.maxCharWidthFrac * W;
        hasArea = area >= params.minCharAreaFracBand * H * W;
        hasDensity = density >= params.minCharDensity;

        looksLikeChar = isNotTooTall && isWideEnough && isNotTooWide && ...
            (isTallEnough || hasArea || hasDensity);

        if looksLikeChar
            boxesBand = [boxesBand; x1 y1b bw bh]; %#ok<AGROW>
        end
    end

    if ~isempty(boxesBand)
        [~, order] = sort(boxesBand(:,1));
        boxesBand = boxesBand(order,:);
    end

    heightFilterInfo = struct();
    heightFilterInfo.aplicat = false;
    heightFilterInfo.numAbans = size(boxesBand, 1);
    heightFilterInfo.numDespres = size(boxesBand, 1);
    heightFilterInfo.motiu = "No cal filtre per alçada";
    heightFilterInfo.hRef = [];
    heightFilterInfo.minH = [];
    heightFilterInfo.maxH = [];

    if size(boxesBand, 1) > params.maxCharsAbansFiltreAlcada
        [boxesFiltrades, heightFilterInfo] = novaPassadaComponentsPerAlcada(BWdetail, params);

        if heightFilterInfo.aplicat
            boxesBand = boxesFiltrades;
        end
    end

    charsTemp = {};
    boxesTemp = [];
    blobInfos = [];

    for i = 1:size(boxesBand,1)

        bb = boxesBand(i,:);

        bbGlobal = bb;
        bbGlobal(2) = bbGlobal(2) + y1 - 1;

        charImgGris = cropBoxWithPadding(gray0, bbGlobal, params.cropPadFrac);

        [charNorm, blobInfo] = normalitzaCaracterGrisAmbInfo(charImgGris, params.charCanvas);

        charsTemp{end+1} = charNorm; %#ok<AGROW>
        boxesTemp = [boxesTemp; bb]; %#ok<AGROW>
        blobInfos = [blobInfos; blobInfo]; %#ok<AGROW>
    end

    postBlobFilterInfo = struct();
    postBlobFilterInfo.aplicat = false;
    postBlobFilterInfo.numAbans = numel(charsTemp);
    postBlobFilterInfo.numDespres = numel(charsTemp);
    postBlobFilterInfo.motiu = "No cal filtre post-blob";
    postBlobFilterInfo.hRef = [];
    postBlobFilterInfo.minH = [];

    if numel(charsTemp) > params.postBlobFilterIfMoreThan
        [keepBlob, postBlobFilterInfo] = filtraPerAlcadaBlobFinal(blobInfos, params);

        if postBlobFilterInfo.aplicat
            charsTemp = charsTemp(keepBlob);
            boxesTemp = boxesTemp(keepBlob, :);
        end
    end

    chars = charsTemp;
    boxesBand = boxesTemp;

    boxesScaled = boxesBand;

    if ~isempty(boxesScaled)
        boxesScaled(:,2) = boxesScaled(:,2) + y1 - 1;
        boxes = boxesScaled ./ scale;
    else
        boxes = zeros(0,4);
    end

    dbg = struct();
    dbg.gray = gray;
    dbg.gray0 = gray0;
    dbg.bwRaw = bwRaw;
    dbg.bwClean = bwClean;
    dbg.bwBand = bwBand;
    dbg.bwaux = BW2;
    dbg.bwChars = BWproj;
    dbg.bwDetail = BWdetail;
    dbg.yBand = [y1 y2];
    dbg.boxesBand = boxesBand;
    dbg.boxesScaled = boxesScaled;
    dbg.boxesROI = boxes;
    dbg.binDbg = binDbg;
    dbg.rowProj = rowProj;
    dbg.colProj = colProj;
    dbg.runs = runs;
    dbg.lineDbg = lineDbg;
    dbg.heightFilterInfo = heightFilterInfo;
    dbg.postBlobFilterInfo = postBlobFilterInfo;
    dbg.blobInfos = blobInfos;
end

function [keep, info] = filtraPerAlcadaBlobFinal(blobInfos, params)

    n = numel(blobInfos);

    keep = true(n, 1);

    info = struct();
    info.aplicat = false;
    info.numAbans = n;
    info.numDespres = n;
    info.motiu = "Filtre post-blob no aplicat";
    info.hRef = [];
    info.minH = [];

    if n == 0
        info.motiu = "No hi ha blobs";
        return;
    end

    found = [blobInfos.found]';
    heights = [blobInfos.blobHeight]';

    validHeights = heights(found & heights > 0);

    if isempty(validHeights)
        info.motiu = "No hi ha alçades vàlides";
        return;
    end

    hMedian = percentileLocal(validHeights, 50);
    highHeights = validHeights(validHeights >= hMedian);

    if isempty(highHeights)
        hRef = median(validHeights);
    else
        hRef = median(highHeights);
    end

    minH = max(params.postBlobMinHeightPx, params.postBlobMinRelMedian * hRef);

    keep = found & heights >= minH;

    info.hRef = hRef;
    info.minH = minH;
    info.numDespres = sum(keep);

    if sum(keep) < params.postBlobMinCharsAfter
        keep = true(n, 1);
        info.numDespres = n;
        info.motiu = "El filtre post-blob deixaria massa pocs caràcters";
        return;
    end

    if sum(keep) == n
        info.motiu = "El filtre post-blob no elimina res";
        return;
    end

    info.aplicat = true;
    info.motiu = "Filtre post-blob per alçada aplicat";
end

function [boxesOut, info] = novaPassadaComponentsPerAlcada(BW, params)

    BW = logical(BW);

    [H, W] = size(BW);

    info = struct();
    info.aplicat = false;
    info.numAbans = 0;
    info.numDespres = 0;
    info.motiu = "Filtre no aplicat";
    info.hRef = [];
    info.minH = [];
    info.maxH = [];

    if nnz(BW) == 0
        boxesOut = zeros(0, 4);
        info.motiu = "Imatge binària buida";
        return;
    end

    closeLen = max(2, round(params.heightCloseVerticalRel * H));
    BWcc = imclose(BW, strel('line', closeLen, 90));

    CC = bwconncomp(BWcc, params.connectivity);
    stats = regionprops(CC, 'BoundingBox', 'Area');

    if isempty(stats)
        boxesOut = zeros(0, 4);
        info.motiu = "No hi ha components connexes";
        return;
    end

    comps = [];

    for i = 1:numel(stats)

        bb = stats(i).BoundingBox;

        x1 = max(1, floor(bb(1)));
        y1 = max(1, floor(bb(2)));
        x2 = min(W, ceil(bb(1) + bb(3) - 1));
        y2 = min(H, ceil(bb(2) + bb(4) - 1));

        bw = x2 - x1 + 1;
        bh = y2 - y1 + 1;

        area = stats(i).Area;
        density = area / max(bw * bh, eps);

        if area < params.heightMinAreaFracBand * H * W
            continue;
        end

        if bw < params.heightMinWidthFracBand * W
            continue;
        end

        if bw > params.heightMaxWidthFracBand * W
            continue;
        end

        if bh < 0.08 * H || bh > 1.05 * H
            continue;
        end

        if density < params.heightMinDensity
            continue;
        end

        comps = [comps; x1 y1 bw bh area density]; %#ok<AGROW>
    end

    info.numAbans = size(comps, 1);

    if isempty(comps)
        boxesOut = zeros(0, 4);
        info.motiu = "Cap component supera els filtres bàsics";
        return;
    end

    heights = comps(:,4);

    hMedian = percentileLocal(heights, 50);
    heightsAltes = heights(heights >= hMedian);

    if isempty(heightsAltes)
        hRef = median(heights);
    else
        hRef = median(heightsAltes);
    end

    minH = max(params.heightMinFracBand * H, params.heightMinRelMedian * hRef);
    maxH = min(params.heightMaxFracBand * H, params.heightMaxRelMedian * hRef);

    keep = comps(:,4) >= minH & comps(:,4) <= maxH;

    compsFiltrades = comps(keep, :);

    info.hRef = hRef;
    info.minH = minH;
    info.maxH = maxH;
    info.numDespres = size(compsFiltrades, 1);

    if size(compsFiltrades, 1) < params.minCharsDespresFiltreAlcada
        boxesOut = zeros(0, 4);
        info.motiu = "El filtre deixaria massa pocs caràcters";
        return;
    end

    boxesOut = compsFiltrades(:, 1:4);

    [~, order] = sort(boxesOut(:,1));
    boxesOut = boxesOut(order, :);

    info.aplicat = true;
    info.motiu = "Filtre per alçada aplicat";
end

function [out, blobInfo] = normalitzaCaracterGrisAmbInfo(imgGris, canvasSize)

    targetH = canvasSize(1);
    targetW = canvasSize(2);

    blobInfo = struct();
    blobInfo.found = false;
    blobInfo.blobHeight = 0;
    blobInfo.blobWidth = 0;
    blobInfo.blobArea = 0;

    umbral = graythresh(imgGris);
    bw = imbinarize(imgGris, umbral);

    if sum(bw(:)) > numel(bw)/2
        bw = ~bw;
    end

    cc = bwconncomp(bw, 8);

    if cc.NumObjects > 0
        stats = regionprops(cc, 'Area');
        [~, idxMax] = max([stats.Area]);

        bwNeta = false(size(bw));
        bwNeta(cc.PixelIdxList{idxMax}) = true;
    else
        bwNeta = bw;
    end

    [r, c] = find(bwNeta);

    if isempty(r)
        out = true(targetH, targetW);
        return;
    end

    r1 = min(r);
    r2 = max(r);
    c1 = min(c);
    c2 = max(c);

    blobInfo.found = true;
    blobInfo.blobHeight = r2 - r1 + 1;
    blobInfo.blobWidth = c2 - c1 + 1;
    blobInfo.blobArea = nnz(bwNeta);

    bwRetallada = bwNeta(r1:r2, c1:c2);

    usableH = targetH;
    usableW = targetW;

    scale = min(usableH / size(bwRetallada,1), usableW / size(bwRetallada,2));

    newH = max(1, round(size(bwRetallada,1) * scale));
    newW = max(1, round(size(bwRetallada,2) * scale));

    resized = imresize(bwRetallada, [newH newW], 'nearest');

    out = true(targetH, targetW);

    r0 = floor((targetH - newH)/2) + 1;
    c0 = floor((targetW - newW)/2) + 1;

    out(r0:r0+newH-1, c0:c0+newW-1) = ~resized;
end

function [bestBW, dbg] = binaritzaCaractersRobust(gray, params)

    G = im2double(gray);

    [H, W] = size(G);

    candidates = {};
    names = {};

    try
        BW1 = imbinarize(gray, 'adaptive', ...
            'ForegroundPolarity', 'dark', ...
            'Sensitivity', params.sensitivity);
    catch
        T = graythresh(gray);
        BW1 = ~imbinarize(gray, T);
    end

    candidates{end+1} = BW1;
    names{end+1} = 'adaptive_dark';

    try
        BW2light = imbinarize(gray, 'adaptive', ...
            'ForegroundPolarity', 'bright', ...
            'Sensitivity', params.sensitivity);
        BW2 = ~BW2light;
    catch
        T = graythresh(gray);
        BW2 = ~imbinarize(gray, T);
    end

    candidates{end+1} = BW2;
    names{end+1} = 'adaptive_light_inverted';

    seH = max(5, makeOddLocal(round(0.45 * H)));
    seW = max(7, makeOddLocal(round(0.10 * W)));

    BH = imbothat(G, strel('rectangle', [seH seW]));
    BH = mat2gray(BH);

    if exist('imgaussfilt', 'file')
        BH = imgaussfilt(BH, 0.45);
    end

    if exist('adaptthresh', 'file')
        neigh = makeOddLocal(max(9, round(0.35 * min(H, W))));
        Tbh = adaptthresh(BH, 0.42, ...
            'ForegroundPolarity', 'bright', ...
            'NeighborhoodSize', neigh);
        BW3 = imbinarize(BH, Tbh);
    else
        BW3 = imbinarize(BH, graythresh(BH));
    end

    q = percentileLocal(BH(:), 55);
    BW3 = BW3 & (BH > q);

    candidates{end+1} = BW3;
    names{end+1} = 'blackhat';

    vals = G(:);
    tg = percentileLocal(vals, 42);
    BW4 = G < tg;

    candidates{end+1} = BW4;
    names{end+1} = 'global_dark';

    scores = zeros(numel(candidates), 1);

    for i = 1:numel(candidates)

        BW = logical(candidates{i});

        [scoreA, BWA] = scoreBinariCaracter(BW, params);
        [scoreB, BWB] = scoreBinariCaracter(~BW, params);

        if scoreB > scoreA + 0.08
            candidates{i} = BWB;
            scores(i) = scoreB;
        else
            candidates{i} = BWA;
            scores(i) = scoreA;
        end
    end

    [~, bestIdx] = max(scores);

    bestBW = logical(candidates{bestIdx});

    dbg = struct();
    dbg.candidates = candidates;
    dbg.names = names;
    dbg.scores = scores;
    dbg.bestIdx = bestIdx;
    dbg.bestName = names{bestIdx};
end

function [score, BWout] = scoreBinariCaracter(BW, params)

    BW = logical(BW);

    [H, W] = size(BW);

    minArea = max(3, round(0.00025 * H * W));

    BWtest = bwareaopen(BW, minArea, params.connectivity);
    BWtest = eliminaComponentsExtrems(BWtest, params);

    density = nnz(BWtest) / numel(BWtest);
    densityScore = exp(-((density - 0.16) / 0.16)^2);

    if density > 0.55
        densityScore = densityScore * 0.1;
    end

    if density < 0.01
        densityScore = densityScore * 0.2;
    end

    CC = bwconncomp(BWtest, params.connectivity);
    stats = regionprops(CC, 'BoundingBox', 'Area', 'Centroid');

    if isempty(stats)
        score = -Inf;
        BWout = BWtest;
        return;
    end

    bb = vertcat(stats.BoundingBox);
    cent = vertcat(stats.Centroid);

    widths = bb(:,3);
    heights = bb(:,4);

    cx = cent(:,1);
    cy = cent(:,2);

    charLike = heights >= 0.18 * H & ...
               heights <= 0.95 * H & ...
               widths >= 0.006 * W & ...
               widths <= 0.35 * W;

    nCharLike = sum(charLike);
    nScore = min(1, nCharLike / 6);

    if nCharLike >= 2
        alignScore = exp(-(std(cy(charLike)) / ...
            max(1, 0.35 * median(heights(charLike))))^2);

        span = (max(cx(charLike)) - min(cx(charLike))) / max(W, eps);
        spanScore = min(1, span / 0.45);
    else
        alignScore = 0;
        spanScore = 0;
    end

    r1 = max(1, round(0.15 * H));
    r2 = min(H, round(0.88 * H));

    centralDensity = nnz(BWtest(r1:r2, :)) / numel(BWtest(r1:r2, :));
    centralScore = exp(-((centralDensity - 0.18) / 0.18)^2);

    border = false(H, W);

    b = max(1, round(0.04 * min(H, W)));

    border(1:b,:) = true;
    border(end-b+1:end,:) = true;
    border(:,1:b) = true;
    border(:,end-b+1:end) = true;

    borderDensity = nnz(BWtest & border) / max(nnz(border), eps);

    borderPenalty = 1;

    if borderDensity > 0.35
        borderPenalty = 0.45;
    end

    score = borderPenalty * ...
        (0.30 * densityScore + ...
         0.25 * nScore + ...
         0.20 * alignScore + ...
         0.15 * spanScore + ...
         0.10 * centralScore);

    BWout = BWtest;
end

function BWout = netejaBinariCaracters(BW, params)

    BW = logical(BW);

    minArea = max(6, round(params.minAreaFrac * numel(BW)));

    BWout = bwareaopen(BW, minArea, params.connectivity);
    BWout = eliminaComponentsExtrems(BWout, params);
    BWout = imclose(BWout, strel('rectangle', params.closeSize));
end

function BWout = eliminaComponentsExtrems(BW, params)

    [H, W] = size(BW);

    CC = bwconncomp(BW, params.connectivity);
    stats = regionprops(CC, 'BoundingBox', 'Area');

    BWout = false(size(BW));

    for i = 1:numel(stats)

        bb = stats(i).BoundingBox;
        area = stats(i).Area;

        bw = bb(3);
        bh = bb(4);

        if area < 2
            continue;
        end

        if area > 0.35 * H * W
            continue;
        end

        if bw > 0.55 * W && bh < 0.25 * H
            continue;
        end

        if bw > 0.92 * W
            continue;
        end

        if bh > 0.96 * H && bw > 0.15 * W
            continue;
        end

        BWout(CC.PixelIdxList{i}) = true;
    end
end

function [BWout, lineDbg] = eliminaLiniesHoritzontalsMorfologiques(BW, params)

    BW = logical(BW);

    [~, W] = size(BW);

    lineLen = max(params.hLineMinLenPx, round(params.hLineLenFrac * W));
    lineLen = min(lineLen, W);

    seLine = strel('line', lineLen, 0);
    lineMask = imopen(BW, seLine);

    seDil = strel('rectangle', [params.hLineDilateHeight params.hLineDilateWidth]);
    lineMaskDil = imdilate(lineMask, seDil);

    BWout = BW;
    BWout(lineMaskDil) = 0;

    lineDbg = struct();
    lineDbg.lineLen = lineLen;
    lineDbg.lineMask = lineMask;
    lineDbg.lineMaskDil = lineMaskDil;
end

function runs = findRuns1D(mask)

    mask = logical(mask(:))';

    if isempty(mask)
        runs = zeros(0, 2);
        return;
    end

    d = diff([false mask false]);

    starts = find(d == 1);
    ends = find(d == -1) - 1;

    runs = [starts(:) ends(:)];
end

function runsOut = mergeRuns1D(runs, maxGap)

    if isempty(runs)
        runsOut = runs;
        return;
    end

    runsOut = runs(1, :);

    for i = 2:size(runs, 1)

        gap = runs(i, 1) - runsOut(end, 2) - 1;

        if gap <= maxGap
            runsOut(end, 2) = runs(i, 2);
        else
            runsOut(end+1, :) = runs(i, :); %#ok<AGROW>
        end
    end
end

function runsOut = splitWideRunsByValleys(runs, colProj, params)

    runsOut = zeros(0, 2);

    if isempty(runs)
        return;
    end

    widths = runs(:, 2) - runs(:, 1) + 1;
    medW = median(widths);

    for i = 1:size(runs, 1)

        x1 = runs(i, 1);
        x2 = runs(i, 2);
        w = x2 - x1 + 1;

        if w < max(params.splitWideFactor * medW, params.minWideRunWidth)
            runsOut(end+1, :) = [x1 x2]; %#ok<AGROW>
            continue;
        end

        segment = colProj(x1:x2);

        if max(segment) == 0
            runsOut(end+1, :) = [x1 x2]; %#ok<AGROW>
            continue;
        end

        valleyThr = params.valleyFrac * max(segment);
        valleys = find(segment <= valleyThr);

        margin = max(2, round(0.18 * w));
        valleys = valleys(valleys > margin & valleys < w - margin);

        if isempty(valleys)
            runsOut(end+1, :) = [x1 x2]; %#ok<AGROW>
            continue;
        end

        center = w / 2;
        [~, idx] = min(abs(valleys - center));

        splitRel = valleys(idx);
        splitX = x1 + splitRel - 1;

        leftRun = [x1 max(x1, splitX - 1)];
        rightRun = [min(x2, splitX + 1) x2];

        if leftRun(2) >= leftRun(1)
            runsOut(end+1, :) = leftRun; %#ok<AGROW>
        end

        if rightRun(2) >= rightRun(1)
            runsOut(end+1, :) = rightRun; %#ok<AGROW>
        end
    end
end

function crop = cropBoxWithPadding(BW, bb, padFrac)

    H = size(BW,1);
    W = size(BW,2);

    x = floor(bb(1));
    y = floor(bb(2));
    w = ceil(bb(3));
    h = ceil(bb(4));

    pad = round(padFrac * H);

    x1 = max(1, x - pad);
    y1 = max(1, y - pad);
    x2 = min(W, x + w + pad);
    y2 = min(H, y + h + pad);

    crop = BW(y1:y2, x1:x2);
end

function [roiAdj, info] = ajustaROIMatriculaConservadorV1(roiInput, params)

    if nargin < 2
        params = struct();
    end

    params = completaParamsAjust(params);

    if ischar(roiInput) || isstring(roiInput)
        I0 = imread(roiInput);
    else
        I0 = roiInput;
    end

    roiAdj = I0;

    info = struct();
    info.didRotate = false;
    info.didCrop = false;
    info.angleDeg = 0;
    info.candidateAngleDeg = 0;
    info.reason = "";
    info.initialCandidate = [];
    info.finalCandidate = [];
    info.cropBox = [];
    info.I0 = I0;
    info.roiAlreadyTight = false;

    if isempty(I0)
        info.reason = "ROI buida";
        return;
    end

    G0 = toGrayDoubleAjust(I0);

    [H0, W0] = size(G0);

    cand0 = detectaPlacaClaraAjust(G0, params);

    info.initialCandidate = cand0;
    info.finalCandidate = cand0;

    if cand0.found
        info.candidateAngleDeg = cand0.angleDeg;
    end

    if ~cand0.found || cand0.score < params.minCandidateScore
        info.reason = "No hi ha candidat fiable; es retorna original";
        info.cropBox = [1 1 W0 H0];
        return;
    end

    cropBox = expandeixBoxAjust(cand0.box, H0, W0, params);

    info.cropBox = cropBox;

    roiAlreadyTight = cropGairebeIgualAjust(cropBox, H0, W0, params);
    info.roiAlreadyTight = roiAlreadyTight;

    cropArea = cropBox(3) * cropBox(4);
    areaRel = cropArea / max(H0 * W0, eps);

    cropTooSmall = areaRel < params.minCropAreaRel;

    if roiAlreadyTight
        roiAdj = I0;
        info.didCrop = false;
        info.cropBox = [1 1 W0 H0];
        info.reason = "ROI ja sembla ben acotada; no es retalla";
    elseif cropTooSmall
        roiAdj = I0;
        info.didCrop = false;
        info.cropBox = [1 1 W0 H0];
        info.reason = "Retall proposat massa petit; es conserva original";
    else
        roiAdj = imcrop(I0, cropBox);
        info.didCrop = true;
        info.reason = "Ajust aplicat";
    end
end

function cand = detectaPlacaClaraAjust(G, params)

    [H, W] = size(G);

    cand = struct();
    cand.found = false;
    cand.score = 0;
    cand.box = [1 1 W H];
    cand.angleDeg = 0;
    cand.mask = false(H, W);
    cand.allMask = false(H, W);
    cand.Gs = G;

    if exist('adapthisteq', 'file')
        Gc = adapthisteq(G, 'NumTiles', [4 4], 'ClipLimit', 0.01);
    else
        Gc = mat2gray(G);
    end

    if exist('imgaussfilt', 'file')
        Gs = imgaussfilt(Gc, 0.8);
    else
        Gs = Gc;
    end

    r1 = max(1, round(0.08 * H));
    r2 = min(H, round(0.92 * H));
    c1 = max(1, round(0.04 * W));
    c2 = min(W, round(0.96 * W));

    vals = Gs(r1:r2, c1:c2);

    pBright = percentileValueAjust(vals(:), params.brightPercentile);
    thr = max(params.minBrightThreshold, pBright);

    bright = Gs >= thr;

    seH = max(3, makeOddAjust(round(params.closeHRel * H)));
    seW = max(3, makeOddAjust(round(params.closeWRel * W)));

    brightClosed = imclose(bright, strel('rectangle', [seH seW]));
    brightClosed = imfill(brightClosed, 'holes');

    minArea = max(4, round(params.minPlateAreaRel * H * W));
    brightClosed = bwareaopen(brightClosed, minArea);

    CC = bwconncomp(brightClosed, 8);
    stats = regionprops(CC, Gs, ...
        'BoundingBox', 'Area', 'Centroid', 'Orientation', 'Extent', 'MeanIntensity');

    if isempty(stats)
        return;
    end

    bestScore = -Inf;
    bestIdx = NaN;

    for i = 1:numel(stats)

        bb = stats(i).BoundingBox;

        bw = bb(3);
        bh = bb(4);

        ratio = bw / max(bh, eps);
        areaBox = bw * bh;
        areaRel = areaBox / max(H * W, eps);

        cx = stats(i).Centroid(1);
        cy = stats(i).Centroid(2);

        centerDistX = abs(cx - W/2) / max(W/2, eps);
        centerDistY = abs(cy - H/2) / max(H/2, eps);

        if ratio < params.minPlateRatio || ratio > params.maxPlateRatio
            continue;
        end

        if bw < params.minPlateWidthRel * W
            continue;
        end

        if bh < params.minPlateHeightRel * H
            continue;
        end

        if areaRel < params.minPlateBoxAreaRel
            continue;
        end

        ratioScore = exp(-((ratio - params.expectedPlateRatio) / params.ratioSigma)^2);
        widthScore = min(1, bw / max(1, params.goodPlateWidthRel * W));
        heightScore = min(1, bh / max(1, params.goodPlateHeightRel * H));

        centerScore = exp(-((centerDistX / 0.70)^2 + (centerDistY / 0.85)^2));
        brightnessScore = min(1, stats(i).MeanIntensity / params.goodBrightMean);
        extentScore = min(1, stats(i).Extent / 0.65);

        score = 0.30 * ratioScore + ...
                0.20 * widthScore + ...
                0.15 * heightScore + ...
                0.15 * centerScore + ...
                0.15 * brightnessScore + ...
                0.05 * extentScore;

        if score > bestScore
            bestScore = score;
            bestIdx = i;
        end
    end

    if isnan(bestIdx)
        return;
    end

    bb = stats(bestIdx).BoundingBox;

    cand.found = true;
    cand.score = bestScore;
    cand.box = bb;
    cand.angleDeg = stats(bestIdx).Orientation;
    cand.mask = false(H, W);
    cand.mask(CC.PixelIdxList{bestIdx}) = true;
    cand.allMask = brightClosed;
    cand.Gs = Gs;
end

function cropBox = expandeixBoxAjust(box, H, W, params)

    x1 = box(1);
    y1 = box(2);
    x2 = box(1) + box(3) - 1;
    y2 = box(2) + box(4) - 1;

    padX = max(params.minPadPx, round(params.padXRel * box(3)));
    padY = max(params.minPadPx, round(params.padYRel * box(4)));

    x1 = max(1, round(x1 - padX));
    y1 = max(1, round(y1 - padY));
    x2 = min(W, round(x2 + padX));
    y2 = min(H, round(y2 + padY));

    cropBox = [x1 y1 x2 - x1 + 1 y2 - y1 + 1];
end

function sameEnough = cropGairebeIgualAjust(cropBox, H, W, params)

    leftMargin = cropBox(1) - 1;
    topMargin = cropBox(2) - 1;
    rightMargin = W - (cropBox(1) + cropBox(3) - 1);
    bottomMargin = H - (cropBox(2) + cropBox(4) - 1);

    maxMarginX = max(leftMargin, rightMargin) / max(W, eps);
    maxMarginY = max(topMargin, bottomMargin) / max(H, eps);

    sameEnough = maxMarginX < params.minCropChangeRel && ...
                 maxMarginY < params.minCropChangeRel;
end

function params = completaParamsAjust(params)

    defaults = struct();

    defaults.debug = false;
    defaults.enableRotation = false;
    defaults.minCandidateScore = 0.48;

    defaults.brightPercentile = 58;
    defaults.minBrightThreshold = 0.28;

    defaults.closeHRel = 0.20;
    defaults.closeWRel = 0.12;

    defaults.minPlateAreaRel = 0.015;
    defaults.minPlateRatio = 1.7;
    defaults.maxPlateRatio = 8.5;
    defaults.expectedPlateRatio = 4.6;
    defaults.ratioSigma = 2.2;

    defaults.minPlateWidthRel = 0.35;
    defaults.minPlateHeightRel = 0.16;
    defaults.minPlateBoxAreaRel = 0.075;

    defaults.goodPlateWidthRel = 0.65;
    defaults.goodPlateHeightRel = 0.35;
    defaults.goodBrightMean = 0.62;

    defaults.padXRel = 0.07;
    defaults.padYRel = 0.16;
    defaults.minPadPx = 2;

    defaults.minCropChangeRel = 0.055;
    defaults.minCropAreaRel = 0.22;

    params = mergeParams(defaults, params);
end

function G = toGrayDoubleAjust(I)

    if size(I, 3) == 3
        G = rgb2gray(I);
    else
        G = I;
    end

    G = im2double(G);
end

function v = makeOddAjust(v)

    v = round(v);

    if v < 1
        v = 1;
    end

    if mod(v, 2) == 0
        v = v + 1;
    end
end

function q = percentileValueAjust(v, p)

    v = sort(v(:));

    if isempty(v)
        q = 0;
        return;
    end

    idx = round(1 + (p / 100) * (numel(v) - 1));
    idx = max(1, min(numel(v), idx));

    q = v(idx);
end

function v = makeOddLocal(v)

    v = round(v);

    if v < 1
        v = 1;
    end

    if mod(v, 2) == 0
        v = v + 1;
    end
end

function q = percentileLocal(v, p)

    v = sort(v(:));

    if isempty(v)
        q = 0;
        return;
    end

    idx = round(1 + (p / 100) * (numel(v) - 1));
    idx = max(1, min(numel(v), idx));

    q = v(idx);
end

function out = mergeParams(defaults, custom)

    out = defaults;

    if isempty(custom)
        return;
    end

    f = fieldnames(custom);

    for i = 1:numel(f)

        name = f{i};

        if isstruct(custom.(name)) && ...
           isfield(out, name) && ...
           isstruct(out.(name))

            out.(name) = mergeParams(out.(name), custom.(name));
        else
            out.(name) = custom.(name);
        end
    end
end