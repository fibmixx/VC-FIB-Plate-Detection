%% PracticaPipeline.m

clear; close all; clc;

carpetaImatges = fullfile(pwd, "JocProvesComplet");
 
extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"];

fitxers = [];

for i = 1:numel(extensions)
    fitxers = [fitxers; dir(fullfile(carpetaImatges, extensions(i)))]; %#ok<AGROW>
end

if isempty(fitxers)
    error("No s'han trobat imatges a la carpeta JocProvesComplet.");
end

resultats = struct();

for k = 1:numel(fitxers)

    nomImatge = fitxers(k).name;
    pathImatge = fullfile(fitxers(k).folder, nomImatge);

    fprintf("Processant %s...\n", nomImatge);

    I = imread(pathImatge);

    %% 1) Detecció de ROIs candidates de matrícula

    [rois, boxes] = detectaROIsMatricula(I);

    %% 2) Classificador de matrícula
   

    [roiMatricula, boxMatricula, idxROI] = seleccionaROIMatricula(rois, boxes, I);

    if isempty(roiMatricula)
        matriculaLlegida = "";
        chars = {};
        charBoxes = zeros(0, 4);
    else

        %% 3) Separació de caràcters

        [chars, charBoxes, infoSeparacio] = separaCaractersMatriculaROI(roiMatricula); %#ok<NASGU>

        %% 4) OCR dels caràcters

        caractersOCR = strings(1, numel(chars));

        for j = 1:numel(chars)
            caractersOCR(j) = reconeixCaracterOCR(chars{j});
        end

        matriculaLlegida = strjoin(caractersOCR, "");
    end

    %% 5) Guardar resultat

    resultats(k).nomImatge = nomImatge;
    resultats(k).pathImatge = pathImatge;
    resultats(k).numROIs = numel(rois);
    resultats(k).boxesROIs = boxes;
    resultats(k).idxROISeleccionada = idxROI;
    resultats(k).boxMatricula = boxMatricula;
    resultats(k).numCaracters = numel(chars);
    resultats(k).charBoxes = charBoxes;
    resultats(k).matricula = matriculaLlegida;

    fprintf("  ROIs: %d | ROI seleccionada: %d | Caracters: %d | Resultat: %s\n", ...
        numel(rois), idxROI, numel(chars), matriculaLlegida);

end

save("resultats_pipeline.mat", "resultats");

taulaResultats = struct2table(resultats);
writetable(taulaResultats(:, ["nomImatge", "numROIs", "idxROISeleccionada", "numCaracters", "matricula"]), ...
    "resultats_pipeline.csv");

fprintf("\nPipeline acabat.\n");
fprintf("Resultats guardats a resultats_pipeline.mat i resultats_pipeline.csv\n");

%% ========================================================================
%% ===================== CLASSIFICADOR DE MATRÍCULA ========================
%% ========================================================================

function [roiSeleccionada, boxSeleccionada, idxROI] = seleccionaROIMatricula(rois, boxes, I) %#ok<INUSD>

    if isempty(rois)
        roiSeleccionada = [];
        boxSeleccionada = [];
        idxROI = 0;
        return;
    end

    idxROI = 1;
    
    for i = 1:numel(rois)
        if es_matricula(rois{i})
            idxROI = i;
            break; 
        end
    end
    
    
    roiSeleccionada = rois{idxROI};
    boxSeleccionada = boxes(idxROI, :);
end

%% ========================================================================
%% ============================== OCR ======================================
%% ========================================================================

function caracter = reconeixCaracterOCR(charImg) %#ok<INUSD>

    [top_lletres, top_scores] = deteccio_lletra(charImg);
    
    lletraMesProbable = top_lletres(1);
    maximScore = top_scores(1);

    lletra2 = top_lletres(2);

    es_conflicte_O_0 = (lletraMesProbable == "O" && lletra2 == "0") || (lletraMesProbable == "0" && lletra2 == "O");

    if es_conflicte_O_0
        caracter = lletraMesProbable;
    elseif maximScore < 60
        caracter = "?";
    else
        caracter = lletraMesProbable;
    end
end