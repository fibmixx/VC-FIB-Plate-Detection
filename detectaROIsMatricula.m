function [rois, boxes] = detectaROIsMatricula(imgInput)

    if ischar(imgInput) || isstring(imgInput)
        I0 = imread(imgInput);
    else
        I0 = imgInput;
    end

    if size(I0, 3) == 3
        G = rgb2gray(I0);
    else
        G = I0;
    end

    G = im2double(G);
    [H, W] = size(G);

    if exist('adapthisteq', 'file')
        Gc = adapthisteq(G, 'NumTiles', [8 8], 'ClipLimit', 0.01);
    else
        Gc = mat2gray(G);
    end

    seH1 = max(5, makeOdd(round(0.025 * H)));
    seW1 = max(15, makeOdd(round(0.060 * W)));

    seH2 = max(3, makeOdd(round(0.015 * H)));
    seW2 = max(11, makeOdd(round(0.035 * W)));

    BH1 = imbothat(Gc, strel('rectangle', [seH1 seW1]));
    BH2 = imbothat(Gc, strel('rectangle', [seH2 seW2]));

    BH = mat2gray(max(BH1, BH2));

    if exist('imgaussfilt', 'file')
        BH = imgaussfilt(BH, 0.6);
    end

    neigh = makeOdd(max(21, round(0.08 * min(H, W))));

    if exist('adaptthresh', 'file')
        T = adaptthresh(BH, 0.42, ...
            'ForegroundPolarity', 'bright', ...
            'NeighborhoodSize', neigh);

        BW = imbinarize(BH, T);
    else
        BW = imbinarize(BH, graythresh(BH));
    end

    q = percentileValue(BH(:), 65);
    BW = BW & (BH > q);

    minArea = max(2, round(0.00001 * H * W));
    BW = bwareaopen(BW, minArea);

    CC = bwconncomp(BW, 8);

    stats = regionprops(CC, ...
        'BoundingBox', ...
        'Centroid', ...
        'Area', ...
        'Extent');

    if isempty(stats)
        rois = {};
        boxes = zeros(0, 4);
        return;
    end

    bb = vertcat(stats.BoundingBox);
    cent = vertcat(stats.Centroid);
    areas = vertcat(stats.Area);
    extents = vertcat(stats.Extent);

    x = bb(:, 1);
    y = bb(:, 2);
    w = bb(:, 3);
    h = bb(:, 4);
    cx = cent(:, 1);
    cy = cent(:, 2);

    aspectBlob = w ./ max(h, eps);

    minBlobH = max(4, round(0.008 * H));
    maxBlobH = max(12, round(0.18 * H));

    minBlobW = 2;
    maxBlobW = max(8, round(0.12 * W));

    maxBlobArea = 0.020 * H * W;

    keep = ...
        h >= minBlobH & h <= maxBlobH & ...
        w >= minBlobW & w <= maxBlobW & ...
        areas >= 2 & areas <= maxBlobArea & ...
        aspectBlob >= 0.04 & aspectBlob <= 1.80 & ...
        extents >= 0.03;

    B = [x y w h cx cy areas];
    B = B(keep, :);

    if isempty(B)
        rois = {};
        boxes = zeros(0, 4);
        return;
    end

    allBoxes = [];
    allScores = [];

    n = size(B, 1);

    for i = 1:n
        groupIdx = growHorizontalGroupFromSeed(i, B, H, W);

        if numel(groupIdx) < 3
            continue;
        end

        [box, score] = scoreGroupAndBuildBox(groupIdx, B, BW, H, W);

        if score < 0.28
            continue;
        end

        allBoxes = [allBoxes; box]; %#ok<AGROW>
        allScores = [allScores; score]; %#ok<AGROW>
    end

    primaryAllBoxes = allBoxes;
    primaryAllScores = allScores;

    [boxesPrimary, scoresPrimary] = finalizeCandidateBoxes(primaryAllBoxes, primaryAllScores, H, W);

    runFallback = shouldRunWhitePlateFallback(boxesPrimary, scoresPrimary, B, BW, G, H, W);

    if runFallback
        [fallbackBoxes, fallbackScores] = generateWhitePlateFallback(B, BW, G, H, W);

        if ~isempty(fallbackBoxes)
            [fallbackScores, ordFb] = sort(fallbackScores, 'descend');
            fallbackBoxes = fallbackBoxes(ordFb, :);

            maxFallback = min(25, numel(fallbackScores));

            fallbackBoxes = fallbackBoxes(1:maxFallback, :);
            fallbackScores = fallbackScores(1:maxFallback);

            allBoxes = [primaryAllBoxes; fallbackBoxes];
            allScores = [primaryAllScores; fallbackScores];

            [boxes, ~] = finalizeCandidateBoxes(allBoxes, allScores, H, W);
        else
            boxes = boxesPrimary;
        end
    else
        boxes = boxesPrimary;
    end

    if isempty(boxes)
        rois = {};
        boxes = zeros(0, 4);
        return;
    end

    rois = cell(size(boxes, 1), 1);

    for i = 1:size(boxes, 1)
        rois{i} = imcrop(I0, boxes(i, :));
    end
end

function groupIdx = growHorizontalGroupFromSeed(seedIdx, B, H, W)
    %#ok<INUSD>

    x = B(:, 1);
    w = B(:, 3);
    h = B(:, 4);
    cx = B(:, 5);
    cy = B(:, 6);

    x1 = x;
    x2 = x + w - 1;

    h0 = h(seedIdx);
    cy0 = cy(seedIdx);

    heightRatio = h ./ max(h0, eps);

    rowMask = ...
        heightRatio >= 0.45 & heightRatio <= 2.30 & ...
        abs(cy - cy0) <= max(10, 0.85 * max(h, h0));

    cand = find(rowMask);

    if numel(cand) < 3
        groupIdx = seedIdx;
        return;
    end

    [~, ord] = sort(cx(cand));
    cand = cand(ord);

    seedPos = find(cand == seedIdx, 1);

    if isempty(seedPos)
        groupIdx = seedIdx;
        return;
    end

    left = seedPos;
    right = seedPos;

    while left > 1
        a = cand(left - 1);
        b = cand(left);

        if areNeighbourBlobsCompatible(a, b, x1, x2, h, cy)
            left = left - 1;
        else
            break;
        end
    end

    while right < numel(cand)
        a = cand(right);
        b = cand(right + 1);

        if areNeighbourBlobsCompatible(a, b, x1, x2, h, cy)
            right = right + 1;
        else
            break;
        end
    end

    groupIdx = cand(left:right);
end

function ok = areNeighbourBlobsCompatible(a, b, x1, x2, h, cy)

    medH = median([h(a), h(b)]);
    gap = x1(b) - x2(a);

    heightRatio = max(h(a), h(b)) / max(min(h(a), h(b)), eps);
    verticalDiff = abs(cy(a) - cy(b));

    maxGapNormal = max(8, 2.8 * medH);
    maxGapWide = max(14, 5.2 * medH);

    normalGapOK = ...
        gap <= maxGapNormal & ...
        gap >= -0.9 * medH & ...
        heightRatio <= 2.40 & ...
        verticalDiff <= max(10, 0.95 * medH);

    wideGapOK = ...
        gap > maxGapNormal & ...
        gap <= maxGapWide & ...
        gap >= -0.4 * medH & ...
        heightRatio <= 1.70 & ...
        verticalDiff <= max(6, 0.45 * medH);

    ok = normalGapOK || wideGapOK;
end

function [box, score] = scoreGroupAndBuildBox(groupIdx, B, BW, H, W)

    x = B(:, 1);
    y = B(:, 2);
    w = B(:, 3);
    h = B(:, 4);
    cx = B(:, 5);
    cy = B(:, 6);

    x1 = min(x(groupIdx));
    y1 = min(y(groupIdx));
    x2 = max(x(groupIdx) + w(groupIdx) - 1);
    y2 = max(y(groupIdx) + h(groupIdx) - 1);

    medH = median(h(groupIdx));

    rawBox = [x1 y1 x2 - x1 + 1 y2 - y1 + 1];

    rawBox = expandBoxWithBinaryEvidence(BW, rawBox, medH, H, W);

    padX = max(round(0.8 * medH), round(0.08 * rawBox(3)));
    padY = max(2, round(0.40 * medH));

    bx1 = max(1, rawBox(1) - padX);
    by1 = max(1, rawBox(2) - padY);
    bx2 = min(W, rawBox(1) + rawBox(3) - 1 + padX);
    by2 = min(H, rawBox(2) + rawBox(4) - 1 + padY);

    box = [bx1 by1 bx2 - bx1 + 1 by2 - by1 + 1];

    nBlobs = numel(groupIdx);

    [~, ord] = sort(cx(groupIdx));
    g = groupIdx(ord);

    gx1 = x(g);
    gx2 = x(g) + w(g) - 1;

    if numel(g) >= 2
        gaps = gx1(2:end) - gx2(1:end-1);
    else
        gaps = [];
    end

    roiRatio = box(3) / max(box(4), eps);

    nScore = min(1, (nBlobs - 2) / 4);

    alignScore = exp(- (std(cy(groupIdx)) / max(1, 0.65 * medH)) ^ 2);

    heightScore = exp(- (mad(h(groupIdx), 1) / max(1, 0.45 * medH)) ^ 2);

    if isempty(gaps)
        gapScore = 0;
    else
        normalGaps = gaps <= 2.8 * medH & gaps >= -0.8 * medH;
        wideGaps = gaps > 2.8 * medH & gaps <= 5.2 * medH;

        gapValues = zeros(size(gaps));
        gapValues(normalGaps) = 1.0;
        gapValues(wideGaps) = 0.55;

        reasonableGaps = mean(gapValues);
        regularity = exp(- (std(max(gaps, 0)) / max(1, 2.2 * medH)) ^ 2);

        gapScore = 0.65 * reasonableGaps + 0.35 * regularity;
    end

    aspectScore = exp(- ((roiRatio - 4.5) / 2.0) ^ 2);

    if roiRatio < 1.6 || roiRatio > 8.5
        aspectScore = aspectScore * 0.2;
    end

    rx1 = max(1, round(box(1)));
    ry1 = max(1, round(box(2)));
    rx2 = min(W, round(box(1) + box(3) - 1));
    ry2 = min(H, round(box(2) + box(4) - 1));

    patch = BW(ry1:ry2, rx1:rx2);
    density = nnz(patch) / numel(patch);

    densityScore = exp(- ((density - 0.09) / 0.12) ^ 2);

    if density > 0.45
        densityScore = densityScore * 0.3;
    end

    score = ...
        0.28 * nScore + ...
        0.22 * alignScore + ...
        0.18 * heightScore + ...
        0.15 * gapScore + ...
        0.12 * aspectScore + ...
        0.05 * densityScore;
end

function box = expandBoxWithBinaryEvidence(BW, box, medH, H, W)

    x1 = round(box(1));
    y1 = round(box(2));
    x2 = round(box(1) + box(3) - 1);
    y2 = round(box(2) + box(4) - 1);

    cy = round((y1 + y2) / 2);

    bandHalf = max(3, round(0.75 * medH));
    by1 = max(1, cy - bandHalf);
    by2 = min(H, cy + bandHalf);

    colSum = sum(BW(by1:by2, :), 1);

    thr = max(1, round(0.08 * (by2 - by1 + 1)));

    maxExpand = round(3.0 * medH);
    maxEmptyGap = max(2, round(0.45 * medH));

    newX1 = x1;
    emptyGap = 0;

    for xx = x1-1:-1:max(1, x1 - maxExpand)
        if colSum(xx) >= thr
            newX1 = xx;
            emptyGap = 0;
        else
            emptyGap = emptyGap + 1;
        end

        if emptyGap > maxEmptyGap
            break;
        end
    end

    newX2 = x2;
    emptyGap = 0;

    for xx = x2+1:min(W, x2 + maxExpand)
        if colSum(xx) >= thr
            newX2 = xx;
            emptyGap = 0;
        else
            emptyGap = emptyGap + 1;
        end

        if emptyGap > maxEmptyGap
            break;
        end
    end

    box = [newX1 y1 newX2 - newX1 + 1 y2 - y1 + 1];
end

function [boxes, scores] = finalizeCandidateBoxes(candidateBoxes, candidateScores, H, W)

    if isempty(candidateBoxes)
        boxes = zeros(0, 4);
        scores = [];
        return;
    end

    ratios = candidateBoxes(:, 3) ./ max(candidateBoxes(:, 4), eps);

    valid = ...
        candidateBoxes(:, 3) >= max(20, 0.04 * W) & ...
        candidateBoxes(:, 4) >= max(8, 0.015 * H) & ...
        ratios >= 1.6 & ratios <= 8.5;

    candidateBoxes = candidateBoxes(valid, :);
    candidateScores = candidateScores(valid);

    if isempty(candidateBoxes)
        boxes = zeros(0, 4);
        scores = [];
        return;
    end

    [candidateScores, ord] = sort(candidateScores, 'descend');
    candidateBoxes = candidateBoxes(ord, :);

    roundedBoxes = round(candidateBoxes);
    [~, ia] = unique(roundedBoxes, 'rows', 'stable');

    candidateBoxes = candidateBoxes(ia, :);
    candidateScores = candidateScores(ia);

    keepIdx = filtraBoxesSolapades(candidateBoxes, candidateScores, 0.35, 30);

    boxes = round(candidateBoxes(keepIdx, :));
    scores = candidateScores(keepIdx);

    [scores, order] = sort(scores, 'descend');
    boxes = boxes(order, :);
end

function runFallback = shouldRunWhitePlateFallback(boxes, scores, B, BW, G, H, W)

    runFallback = true;

    if isempty(boxes)
        return;
    end

    nCheck = min(8, size(boxes, 1));

    bestPlateConfidence = 0;

    for i = 1:nCheck
        box = boxes(i, :);

        ratio = box(3) / max(box(4), eps);

        if ratio < 1.7 || ratio > 8.5
            continue;
        end

        blobScore = computeBlobEvidenceInBox(box, B);
        densityScore = computeDensityScoreInBox(box, BW, H, W);
        whiteScore = computeWhitePlateScore(G, box);

        if isempty(scores) || numel(scores) < i
            scoreNorm = 0;
        else
            scoreNorm = min(1, max(0, scores(i) / 0.55));
        end

        plateConfidence = ...
            0.35 * scoreNorm + ...
            0.35 * blobScore + ...
            0.20 * whiteScore + ...
            0.10 * densityScore;

        bestPlateConfidence = max(bestPlateConfidence, plateConfidence);
    end

    numPrimary = size(boxes, 1);

    runFallback = ...
        (numPrimary <= 10) || ...
        (numPrimary <= 15 && bestPlateConfidence < 0.85) || ...
        (bestPlateConfidence < 0.62);
end

function blobScore = computeBlobEvidenceInBox(box, B)

    if isempty(B)
        blobScore = 0;
        return;
    end

    h = B(:, 4);
    cx = B(:, 5);
    cy = B(:, 6);

    x1 = box(1);
    y1 = box(2);
    x2 = box(1) + box(3) - 1;
    y2 = box(2) + box(4) - 1;

    mx = 0.08 * box(3);
    my = 0.20 * box(4);

    inside = ...
        cx >= x1 - mx & cx <= x2 + mx & ...
        cy >= y1 - my & cy <= y2 + my;

    idx = find(inside);

    if numel(idx) < 3
        blobScore = 0;
        return;
    end

    medH = median(h(idx));

    nScore = min(1, (numel(idx) - 2) / 5);

    alignScore = exp(- (std(cy(idx)) / max(1, 0.80 * medH)) ^ 2);

    heightScore = exp(- (mad(h(idx), 1) / max(1, 0.65 * medH)) ^ 2);

    horizontalSpan = (max(cx(idx)) - min(cx(idx))) / max(box(3), eps);
    spanScore = min(1, horizontalSpan / 0.45);

    blobScore = ...
        0.35 * nScore + ...
        0.25 * alignScore + ...
        0.20 * heightScore + ...
        0.20 * spanScore;

    blobScore = max(0, min(1, blobScore));
end

function densityScore = computeDensityScoreInBox(box, BW, H, W)

    x1 = max(1, round(box(1)));
    y1 = max(1, round(box(2)));
    x2 = min(W, round(box(1) + box(3) - 1));
    y2 = min(H, round(box(2) + box(4) - 1));

    if x2 <= x1 || y2 <= y1
        densityScore = 0;
        return;
    end

    patch = BW(y1:y2, x1:x2);

    density = nnz(patch) / numel(patch);

    densityScore = exp(- ((density - 0.10) / 0.16) ^ 2);

    if density > 0.50
        densityScore = densityScore * 0.3;
    end

    densityScore = max(0, min(1, densityScore));
end

function keep = filtraBoxesSolapades(boxes, scores, thr, maxKeep)

    if isempty(boxes)
        keep = [];
        return;
    end

    [~, order] = sort(scores, 'descend');

    keep = [];

    while ~isempty(order) && numel(keep) < maxKeep
        i = order(1);
        keep(end+1) = i; %#ok<AGROW>

        if numel(order) == 1
            break;
        end

        rest = order(2:end);
        solapaments = zeros(numel(rest), 1);

        for k = 1:numel(rest)
            solapaments(k) = calculaSolapament(boxes(i, :), boxes(rest(k), :));
        end

        order = rest(solapaments < thr);
    end
end

function [boxes, scores] = generateWhitePlateFallback(B, BW, G, H, W)

    boxes = [];
    scores = [];

    if isempty(B) || size(B, 1) < 4
        return;
    end

    x = B(:, 1);
    w = B(:, 3);
    h = B(:, 4);
    cx = B(:, 5);
    cy = B(:, 6);

    n = size(B, 1);

    for seed = 1:n
        h0 = h(seed);
        cy0 = cy(seed);

        heightRatio = h ./ max(h0, eps);

        bandMask = ...
            abs(cy - cy0) <= max(10, 1.30 * h0) & ...
            heightRatio >= 0.30 & heightRatio <= 3.30;

        cand = find(bandMask);

        if numel(cand) < 4
            continue;
        end

        [~, ord] = sort(cx(cand));
        cand = cand(ord);

        segments = {};
        current = cand(1);

        for k = 2:numel(cand)
            prev = cand(k - 1);
            cur = cand(k);

            medH = median(h(current));

            prevX2 = x(prev) + w(prev) - 1;
            gap = x(cur) - prevX2;

            maxGap = max([14, 4.8 * medH, 0.040 * W]);

            if gap > maxGap
                segments{end + 1} = current; %#ok<AGROW>
                current = cur;
            else
                current = [current; cur]; %#ok<AGROW>
            end
        end

        segments{end + 1} = current; %#ok<AGROW>

        for s = 1:numel(segments)
            seg = segments{s};

            if numel(seg) < 4
                continue;
            end

            maxLen = min(12, numel(seg));

            for len = 4:maxLen
                for a = 1:(numel(seg) - len + 1)
                    groupIdx = seg(a:a + len - 1);

                    [box, baseScore] = scoreGroupAndBuildBox(groupIdx, B, BW, H, W);

                    ratio = box(3) / max(box(4), eps);

                    if ratio < 1.35 || ratio > 10.0
                        continue;
                    end

                    whiteScore = computeWhitePlateScore(G, box);

                    finalScore = 0.72 * baseScore + 0.28 * whiteScore;

                    if finalScore >= 0.18
                        boxes = [boxes; box]; %#ok<AGROW>
                        scores = [scores; finalScore]; %#ok<AGROW>
                    end
                end
            end
        end
    end

    if isempty(boxes)
        return;
    end

    roundedBoxes = round(boxes);
    [~, ia] = unique(roundedBoxes, 'rows', 'stable');

    boxes = boxes(ia, :);
    scores = scores(ia);
end

function whiteScore = computeWhitePlateScore(G, box)

    H = size(G, 1);
    W = size(G, 2);

    x1 = max(1, round(box(1)));
    y1 = max(1, round(box(2)));
    x2 = min(W, round(box(1) + box(3) - 1));
    y2 = min(H, round(box(2) + box(4) - 1));

    if x2 <= x1 || y2 <= y1
        whiteScore = 0;
        return;
    end

    patch = G(y1:y2, x1:x2);

    if isempty(patch)
        whiteScore = 0;
        return;
    end

    p20 = percentileValue(patch(:), 20);
    p70 = percentileValue(patch(:), 70);
    p85 = percentileValue(patch(:), 85);
    p95 = percentileValue(patch(:), 95);

    localMean = mean(patch(:));
    localStd = std(patch(:));

    brightMask = patch > max(localMean + 0.15 * localStd, p70);
    brightFraction = nnz(brightMask) / numel(patch);

    brightnessScore = min(1, p85 / 0.72);
    fractionScore = min(1, brightFraction / 0.42);
    contrastScore = min(1, (p95 - p20) / 0.45);

    whiteScore = ...
        0.45 * brightnessScore + ...
        0.35 * fractionScore + ...
        0.20 * contrastScore;

    whiteScore = max(0, min(1, whiteScore));
end

function s = calculaSolapament(a, b)

    ax1 = a(1);
    ay1 = a(2);
    ax2 = a(1) + a(3) - 1;
    ay2 = a(2) + a(4) - 1;

    bx1 = b(1);
    by1 = b(2);
    bx2 = b(1) + b(3) - 1;
    by2 = b(2) + b(4) - 1;

    xx1 = max(ax1, bx1);
    yy1 = max(ay1, by1);
    xx2 = min(ax2, bx2);
    yy2 = min(ay2, by2);

    iw = max(0, xx2 - xx1 + 1);
    ih = max(0, yy2 - yy1 + 1);

    inter = iw * ih;

    areaA = a(3) * a(4);
    areaB = b(3) * b(4);

    s = inter / max(areaA + areaB - inter, eps);
end

function v = makeOdd(v)

    v = round(v);

    if v < 1
        v = 1;
    end

    if mod(v, 2) == 0
        v = v + 1;
    end
end

function q = percentileValue(v, p)

    v = sort(v(:));

    if isempty(v)
        q = 0;
        return;
    end

    idx = round(1 + (p / 100) * (numel(v) - 1));
    idx = max(1, min(numel(v), idx));

    q = v(idx);
end