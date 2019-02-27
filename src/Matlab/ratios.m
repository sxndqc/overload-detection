function [outputArg1,outputArg2] = ratios(centers)
%UNTITLED2 此处显示有关此函数的摘要
%   此处显示详细说明
    ratios=zeros(1,length(centers)-1);
    for i=2:1:centers(max)
        ratios(i-1) = centers(i)-centers(i-1);
    end
end