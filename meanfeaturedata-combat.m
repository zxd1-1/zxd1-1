clc;
clear;
close all;
tic
%%
dbstop if error
Path = 'D:\程序';
data = importdata([Path filesep 'meanfeaturedata.xlsx']);
[subNum,featureNum] = size(data);
% load('D:\程序\.featuredataxlsx');
data=xlsread("meanfeaturedata.xlsx");
a2=xlsread("data.xlsx");
% ECN = data(:,2)+data(:,3);
% DMN = data(:,5)+data(:,6);
% SN = data(:,1)+data(:,4);
% D = [ECN DMN SN];
%%
site = a2(:,2);
TR = a2(:,5);
sex = a2(:,7);
%hand = a2(77:85,11);
%clinic = cell2mat(group(:,9));
% t_index = t_index' .* TR;
age  = a2(:,6);
%IQ = cell2mat(group(:,1));
%eyes = cell2mat(group(:,62));
%%
Ddata = combat(data(:,:).',sex,age,1);
Ddata = combat(Ddata,TR,age,1);
Ddata = combat(Ddata,site,age,1);
%Ddata = combat(Ddata,site,age,1);
Ddata = Ddata';
%[b,bint,r,stats]=regress(Ddata(1:884,2),a2(:,6));

dlmwrite([Path filesep 'intensity_combat_eyes_TR_site.txt'], Ddata, 'delimiter' , ' ' , 'precision', '%0.4f');
%%
