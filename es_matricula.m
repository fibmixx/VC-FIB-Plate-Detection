function resultado = es_matricula(im)
    % ES_MATRICULA Determina si un ROI en color es una matrícula o no.
    %   resultado = es_matricula(im) devuelve true si el ROI 'im' (RGB) 
    %   se clasifica como matrícula ('SI') y false en caso contrario ('NO').
    
    persistent modelo_guardado;
    if isempty(modelo_guardado)
        nombre_mat = 'modeloF.mat';
        if exist(nombre_mat, 'file')
            data = load(nombre_mat);
            campos = fieldnames(data);
            modelo_guardado = data.(campos{1});
        else
            error('No se encuentra el archivo ''%s'' en el directorio actual.', nombre_mat);
        end
    end
    
    % 2. Preprocesamiento de la imagen
    if size(im, 3) == 1
        im = cat(3, im, im, im);
    end
    
    img_gray = rgb2gray(im);
    bw = imbinarize(img_gray);
    if sum(bw(:)) > numel(bw)/2
        bw = ~bw; % Forzar a que el texto/bordes sean blancos (1) y fondo negro (0)
    end
    

    % 3. EXTRACCIÓN DE CARACTERÍSTICAS
    
    % A. Características de forma/textura
    bordes = edge(img_gray, 'sobel');
    caract_densidad_bordes = sum(bordes(:)) / numel(bordes);
    caract_transiciones = sum(abs(diff(bw, 1, 2)), 'all') / numel(bw);
    
    % B. Características de color
    img_hsv = rgb2hsv(im);
    H = img_hsv(:,:,1); 
    S = img_hsv(:,:,2);
    
    caract_saturacion_media = mean(S(:));
    caract_std_saturacion = std(S(:));
    
    pixeles_azules = (H > 0.55 & H < 0.75) & (S > 0.4);
    caract_porcentaje_azul = sum(pixeles_azules, 'all') / numel(H);
    
    pixeles_amarillos = (H > 0.10 & H < 0.20) & (S > 0.4);
    caract_porcentaje_amarillo = sum(pixeles_amarillos, 'all') / numel(H);
    
    % C. Características geométricas

    caract_proporcion = size(im, 2) / size(im, 1);
    
    stats = regionprops(bw, 'Solidity', 'Area', 'BoundingBox', 'Eccentricity', 'Extent');
    
    if ~isempty(stats)
        [~, idx_max_area] = max([stats.Area]);
        caract_solidesa = stats(idx_max_area).Solidity;
        
        caixa = stats(idx_max_area).BoundingBox;
        caract_proporcion_interna = caixa(3) / caixa(4); % Evita el coche de los lados
        
        caract_excentricidad = stats(idx_max_area).Eccentricity;
        caract_extent = stats(idx_max_area).Extent;
    else
        caract_solidesa = 0;
        caract_proporcion_interna = 0;
        caract_excentricidad = 0;
        caract_extent = 0;
    end
    
    mitja_columna = round(size(bw, 2) / 2);
    meitat_esquerra = bw(:, 1:mitja_columna);
    meitat_dreta = bw(:, mitja_columna+1:end);
    mida_minima = min(size(meitat_esquerra, 2), size(meitat_dreta, 2));
    
    caract_asimetria = sum(abs(meitat_esquerra(:, 1:mida_minima) - fliplr(meitat_dreta(:, 1:mida_minima))), 'all') / numel(bw);
   
    trans_verticales = sum(abs(diff(bw, 1, 1)), 'all');
    trans_horizontales_bruto = sum(abs(diff(bw, 1, 2)), 'all');
    caract_ratio_transiciones = trans_horizontales_bruto / (trans_verticales + 1e-5);
    

    % 4. Crear la tabla idéntica a la utilizada en el entrenamiento

    datos_roi = table(caract_densidad_bordes, ...
                      caract_transiciones, ...
                      caract_saturacion_media, ...
                      caract_std_saturacion, ...
                      caract_porcentaje_azul, ...
                      caract_porcentaje_amarillo, ...
                      caract_proporcion, ...
                      caract_proporcion_interna, ...
                      caract_excentricidad, ...
                      caract_extent, ...
                      caract_solidesa, ...
                      caract_asimetria, ...
                      caract_ratio_transiciones, ...
                      'VariableNames', {...
                          'DensidadBordes', 'Transiciones', ...
                          'SatMedia', 'StdSat', 'PorcentajeAzul', 'PorcentajeAmarillo', ...
                          'ProporcionGlobal', 'ProporcionInterna', 'Eccentricity', 'Extent', ...
                          'Solidity', 'Asimetria', 'RatioTransiciones'});
                      

    % 5. Predicción del modelo

    prediccion_categorical = modelo_guardado.predictFcn(datos_roi);
    
    % Convertir el resultado categórico ('SI'/'NO') a un booleano (true/false)
    if strcmp(char(prediccion_categorical), 'SI')
        resultado = true;
    else
        resultado = false;
    end
end