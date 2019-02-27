function [maxweight] = type_of_truck(centers)
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
    switch len(centers)
        case 2
            weights = [1];
            b = repmat(ratios(centers), length(weights),1);
            c = corr(b, weights);
            lim = [18];
        case 3
            weights = [3 2; 1 1;5 1;1 5];
            b = repmat(ratios(centers), length(weights),1);
            c = corr(b, weights);
            lim = [27 27 25 25];
        case 4
            weights = [4 2 1; 4 1 3; 4 4 1; 4 2 3;1 4 1];
            b = repmat(ratios(centers), length(weights),1);
            c = corr(b, weights);
            lim = [36 35 36 36 31];
        case 5
            weights = [3 1 3 1; 3 7 5 2; 5 1 3 1; 1 3 3 1;4 3 1 1;5 1 2 3;1 5 3 3];
            b = repmat(ratios(centers), length(weights),1);
            c = corr(b, weights);
            lim = [43 43 43 43 42 43 43];
        case 6
            %中间挂钩无法识别
            weights = [6 2 3 2 2; 1 3 1 3 1; 3 1 2 1 1;1 4 2 1 1;1 4 1 3 3];
            b = repmat(ratios(centers), length(weights),1);
            c = corr(b, weights);
            lim = [46 46 46 46 46];
            
        otherwise
             
    
end

