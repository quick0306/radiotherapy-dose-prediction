new %% read matrix data and indexS 
%% global planC include all cntour and dose matrix
clear all
load('Tran_TMI.mat')
global planC
indexS = planC{end};
structureListC = {planC{indexS.structures}.structureName};

%% Define PTV and OARs - The Howell is planned differentla
PTV = ["PTV_Ribs", "PTV_VExP", "PTV_SpCord", "PTV_LN", "PTV_Spleen", "PTV_Liver"];
%%%%%%%Howell,
PTV = ["PTV_Ribs", "PTV_Bone_Total", "PTV_SpCord", "PTV_LN", "PTV_Spleen", "PTV_Liver"];
%%%%% Pim,RABACA, SALCIDO
%PTV = [".PTV3_Ribs", ".PTV2_Bone", ".PTV1_Cord", ".PTV4_LN's", ".PTV6_Spleen", ".PTV7_Liver"];
%%%%%% No breast : Chavez, Colon, HOwell, Kane
OARs=["Lungs","Heart","Esophagus","GI_Upper","Rectum"];
%%%%%%Howell, 
%OARs=["Lungs_Total","Heart","Esophagus","Stomach","Breasts"];
% At Lung area the avoid structure, Pim-Avoid1, Howell-NA, Dameris-NA,
% Salcido-Avoid1,Rabaca-Avoid3, Chavez-Avoid_M, Rivera-NA, Evans-Avoid2,
% NGUYEN-Avoid2, LARASH-NA, COLON-AvoidMed, Tran-Avoid2,
% Lee-Avoid2, Perez-NA, Tompodung-NA, Kane-NA
OptStructure = ["NA"]
SKIN ='BODY';

%% get PTVs mask 
structNum_PTV = getMatchingIndex(lower(PTV{1}),lower(structureListC),'exact');
mask3M_PTV_Ribs  =  getUniformStr(structNum_PTV, planC);

structNum_PTV = getMatchingIndex(lower(PTV{2}),lower(structureListC),'exact');
mask3M_PTV_VExP  =  getUniformStr(structNum_PTV, planC);

structNum_PTV = getMatchingIndex(lower(PTV{3}),lower(structureListC),'exact');
mask3M_PTV_SpCord  =  getUniformStr(structNum_PTV, planC);

structNum_PTV = getMatchingIndex(lower(PTV{4}),lower(structureListC),'exact');
mask3M_PTV_LN  =  getUniformStr(structNum_PTV, planC);

structNum_PTV = getMatchingIndex(lower(PTV{5}),lower(structureListC),'exact');
mask3M_PTV_Spleen  =  getUniformStr(structNum_PTV, planC);

structNum_PTV = getMatchingIndex(lower(PTV{6}),lower(structureListC),'exact');
mask3M_PTV_Liver  =  getUniformStr(structNum_PTV, planC);

structNum_SKIN = getMatchingIndex(lower(SKIN),lower(structureListC),'exact');
mask3M_Body  =  getUniformStr(structNum_SKIN, planC);


%% Get OARs mask
structNum = getMatchingIndex(lower(OARs{1}),lower(structureListC),'exact');
mask3M_Lungs  =  getUniformStr(structNum, planC);

structNum = getMatchingIndex(lower(OARs{2}),lower(structureListC),'exact');
mask3M_Heart  =  getUniformStr(structNum, planC);

structNum = getMatchingIndex(lower(OARs{3}),lower(structureListC),'exact');
mask3M_Esophagus  =  getUniformStr(structNum, planC);

structNum = getMatchingIndex(lower(OARs{4}),lower(structureListC),'exact');
mask3M_GIUpper  =  getUniformStr(structNum, planC);

structNum = getMatchingIndex(lower(OARs{5}),lower(structureListC),'exact');
mask3M_Breasts  =  getUniformStr(structNum, planC);

%% Get optimization structure mask

% structNum = getMatchingIndex(lower(OptStructure{1}),lower(structureListC),'exact');
% mask3M_Avoid1  =  getUniformStr(structNum, planC);

  
mask3M_Avoid1 = zeros(size(mask3M_GIUpper));
mask3M_Avoid1 = logical(mask3M_Avoid1);

%% The dose is medium which is 1/2 of the CT resolution 
%%dose3M  = getDoseArray(1,planC);
%%[xV, yV, zV] = getUniformScanXYZVals(planC{indexS.scan}(scanNum));

dose3M_original = getDoseOnCT(1, 1, 'uniform'); % get dose on CT grid
scan3M = getScanArray(planC{indexS.scan}(1));

%[x,y,z]=size(dose3M);
%mask3M_sum = zeros(x,y,z);
% construct dose logical matrix for 33Gy;
%mask3M_dose= dose3M;
% index = find(dose3M>=33);
% mask3M_dose(index)=1;
volVox = planC{indexS.scan}(1).uniformScanInfo.grid1Units * planC{indexS.scan}(1).uniformScanInfo.grid2Units * planC{indexS.scan}(1).uniformScanInfo.sliceThickness;
res_x=planC{indexS.scan}(1).uniformScanInfo.grid1Units
res_y=planC{indexS.scan}(1).uniformScanInfo.grid2Units
res_z=planC{indexS.scan}(1).uniformScanInfo.sliceThickness
res  = [res_x res_y res_z]

%% resample the strucutre and scan to dose grid (256x256)

res_x_resamp = res_x*2; %cm
res_y_resamp = res_y*2 %cm
res_z_resamp = res_z; %cm

numrows = size(scan3M,1)*res_x/res_x_resamp;
numcols = size(scan3M,2)*res_y/res_y_resamp;
numplanes = size(scan3M,3)*res_z/res_z_resamp;

scan3M_resamp = imresize3(scan3M,[numrows numcols numplanes]);
mask3M_PTV_Ribs = double(mask3M_PTV_Ribs);
mask3M_PTV_Ribs_resamp = imresize3(mask3M_PTV_Ribs,[numrows numcols numplanes],'nearest');
mask3M_PTV_Ribs_resamp = logical(mask3M_PTV_Ribs_resamp);

mask3M_PTV_VExP = double(mask3M_PTV_VExP);
mask3M_PTV_VExP_resamp = imresize3(mask3M_PTV_VExP,[numrows numcols numplanes],'nearest');
mask3M_PTV_VExP_resamp = logical(mask3M_PTV_VExP_resamp);

mask3M_PTV_SpCord = double(mask3M_PTV_SpCord);
mask3M_PTV_SpCord_resamp = imresize3(mask3M_PTV_SpCord,[numrows numcols numplanes],'nearest');
mask3M_PTV_SpCord_resamp = logical(mask3M_PTV_SpCord_resamp);

mask3M_PTV_LN = double(mask3M_PTV_LN);
mask3M_PTV_LN_resamp = imresize3(mask3M_PTV_LN,[numrows numcols numplanes],'nearest');
mask3M_PTV_LN_resamp = logical(mask3M_PTV_LN_resamp);

mask3M_PTV_Spleen = double(mask3M_PTV_Spleen);
mask3M_PTV_Spleen_resamp = imresize3(mask3M_PTV_Spleen,[numrows numcols numplanes],'nearest');
mask3M_PTV_Spleen_resamp = logical(mask3M_PTV_Spleen_resamp);

mask3M_PTV_Liver = double(mask3M_PTV_Liver);
mask3M_PTV_Liver_resamp = imresize3(mask3M_PTV_Liver,[numrows numcols numplanes],'nearest');
mask3M_PTV_Liver_resamp = logical(mask3M_PTV_Liver_resamp);

mask3M_Lungs = double(mask3M_Lungs);
mask3M_Lungs_resamp = imresize3(mask3M_Lungs,[numrows numcols numplanes],'nearest');
mask3M_Lungs_resamp = logical(mask3M_Lungs_resamp);

mask3M_Heart = double(mask3M_Heart);
mask3M_Heart_resamp = imresize3(mask3M_Heart,[numrows numcols numplanes],'nearest');
mask3M_Heart_resamp = logical(mask3M_Heart_resamp);

mask3M_Esophagus = double(mask3M_Esophagus);
mask3M_Esophagus_resamp = imresize3(mask3M_Esophagus,[numrows numcols numplanes],'nearest');
mask3M_Esophagus_resamp = logical(mask3M_Esophagus_resamp);

mask3M_GIUpper = double(mask3M_GIUpper);
mask3M_GIUpper_resamp = imresize3(mask3M_GIUpper,[numrows numcols numplanes],'nearest');
mask3M_GIUpper_resamp = logical(mask3M_GIUpper_resamp);

mask3M_Breasts = double(mask3M_Breasts);
mask3M_Breasts_resamp = imresize3(mask3M_Breasts,[numrows numcols numplanes],'nearest');
mask3M_Breasts_resamp = logical(mask3M_Breasts_resamp);

mask3M_Avoid1 = double(mask3M_Avoid1);
mask3M_Avoid1_resamp = imresize3(mask3M_Avoid1,[numrows numcols numplanes],'nearest');
mask3M_Avoid1_resamp = logical(mask3M_Avoid1_resamp);


mask3M_Body = double(mask3M_Body);
mask3M_Body_resamp = imresize3(mask3M_Body,[numrows numcols numplanes],'nearest');
mask3M_Body_resamp = logical(mask3M_Body_resamp);

dose3M = imresize3(dose3M_original,[numrows numcols numplanes],'nearest');

%% Find the skin range to reduce the data size
index_Lung = find(mask3M_Lungs_resamp);
[x1_Lungs,y1_Lungs,z1_Lungs] = ind2sub(size(mask3M_Body_resamp),index_Lung);

index_Body = find(mask3M_Body_resamp);
[x1_SKIN,y1_SKIN,z1_SKIN] = ind2sub(size(mask3M_Body_resamp),index_Body);
center_x = floor((max(x1_SKIN)+min(x1_SKIN))/2)+10; % do not cut x direction, cut only y and z direction.
x_range = max(x1_SKIN)-center_x+1
x_range = 64

center_y = (max(y1_SKIN)+min(y1_SKIN))/2;
y_range = 64;

center_z = (max(z1_Lungs)+min(z1_Lungs))/2;
z_range = max(z1_Lungs)-center_z+1;

mask3M_Body_cut = mask3M_Body_resamp((center_x-x_range): (center_x+x_range-1),:,(center_z-z_range): (center_z+z_range));
mask3M_PTV_Ribs_cut = mask3M_PTV_Ribs_resamp((center_x-x_range): (center_x+x_range-1),:,(center_z-z_range): (center_z+z_range));
mask3M_PTV_VExP_cut = mask3M_PTV_VExP_resamp((center_x-x_range): (center_x+x_range-1),:,(center_z-z_range): (center_z+z_range));
mask3M_PTV_SpCord_cut = mask3M_PTV_SpCord_resamp((center_x-x_range): (center_x+x_range-1),:,(center_z-z_range): (center_z+z_range));
mask3M_PTV_LN_cut = mask3M_PTV_LN_resamp((center_x-x_range): (center_x+x_range-1),:,(center_z-z_range): (center_z+z_range));
mask3M_PTV_Spleen_cut = mask3M_PTV_Spleen_resamp((center_x-x_range): (center_x+x_range-1),:,(center_z-z_range): (center_z+z_range));
mask3M_PTV_Liver_cut = mask3M_PTV_Liver_resamp((center_x-x_range): (center_x+x_range-1),:,(center_z-z_range): (center_z+z_range));
mask3M_Heart_cut = mask3M_Heart_resamp((center_x-x_range): (center_x+x_range-1),:,(center_z-z_range): (center_z+z_range));
mask3M_Lungs_cut = mask3M_Lungs_resamp((center_x-x_range): (center_x+x_range-1),:,(center_z-z_range): (center_z+z_range));
mask3M_Esophagus_cut = mask3M_Esophagus_resamp((center_x-x_range): (center_x+x_range-1),:,(center_z-z_range): (center_z+z_range));
mask3M_GIUpper_cut = mask3M_GIUpper_resamp((center_x-x_range): (center_x+x_range-1),:,(center_z-z_range): (center_z+z_range));
mask3M_Breasts_cut = mask3M_Breasts_resamp((center_x-x_range): (center_x+x_range-1),:,(center_z-z_range): (center_z+z_range));
mask3M_Avoid1_cut = mask3M_Avoid1_resamp((center_x-x_range): (center_x+x_range-1),:,(center_z-z_range): (center_z+z_range));

dose3M_cut = dose3M((center_x-x_range): (center_x+x_range-1),:,(center_z-z_range): (center_z+z_range));
scan3M_cut = scan3M_resamp((center_x-x_range): (center_x+x_range-1),:,(center_z-z_range): (center_z+z_range));

mask3M_PTV_cut = mask3M_PTV_Ribs_cut | mask3M_PTV_VExP_cut | mask3M_PTV_SpCord_cut | mask3M_PTV_LN_cut | mask3M_PTV_Spleen_cut | mask3M_PTV_Liver_cut ;

%% test DVH
clear DVH1 DVH2 
clear data
data1=dose3M(find(mask3M_PTV_Ribs_resamp));
data2=dose3M_cut(find(mask3M_Lungs_cut));
data3=dose3M(find(mask3M_GIUpper_resamp));
data4=dose3M_cut(find(mask3M_Avoid1_cut));
data5=dose3M(find(mask3M_Breasts_resamp));

for i=1:1:300
    x(i) = (i-1)*0.1;
    DVH1(i) = length(find(data1>=((i-1)*0.1)))/length(data1);
    DVH2(i) = length(find(data2>=((i-1)*0.1)))/length(data2);
    DVH3(i) = length(find(data3>=((i-1)*0.1)))/length(data3);
    DVH4(i) = length(find(data4>=((i-1)*0.1)))/length(data4);
    DVH5(i) = length(find(data5>=((i-1)*0.1)))/length(data5);
end

figure()
plot(x,DVH1, x,DVH2,x,DVH5,x,DVH4)



%% Plot the scan and strucrues
scan3M_rgb = zeros(size(scan3M_cut,1),size(scan3M_cut,2),size(scan3M_cut,3),3);
pic = double(max(max(max(scan3M_cut))));
scan3M_rgb(:,:,:,1)=double(scan3M_cut)/pic;
scan3M_rgb(:,:,:,2)=double(scan3M_cut)/pic;
scan3M_rgb(:,:,:,3)=double(scan3M_cut)/pic;

all_structures = ["PTV_Ribs", "PTV_VExP", "PTV_SpCord", "PTV_LN", "PTV_Spleen", "PTV_Liver", "Lungs","Heart","Esophagus"]

% Get surface mask for structNum
surfPoints = getSurfacePoints(mask3M_PTV_Ribs_cut);
surf3M = repmat(logical(0), size(mask3M_PTV_Ribs_cut));
for i=1:size(surfPoints,1)
      scan3M_rgb(surfPoints(i,1),surfPoints(i,2), surfPoints(i,3),1)=1;
      scan3M_rgb(surfPoints(i,1),surfPoints(i,2), surfPoints(i,3),2)=0;
      scan3M_rgb(surfPoints(i,1),surfPoints(i,2), surfPoints(i,3),3)=0;
end

surfPoints = getSurfacePoints(mask3M_PTV_SpCord_cut);
surf3M = repmat(logical(0), size(mask3M_PTV_SpCord_cut));
for i=1:size(surfPoints,1)
      scan3M_rgb(surfPoints(i,1),surfPoints(i,2), surfPoints(i,3),1)=0.2;
      scan3M_rgb(surfPoints(i,1),surfPoints(i,2), surfPoints(i,3),2)=0.5;
      scan3M_rgb(surfPoints(i,1),surfPoints(i,2), surfPoints(i,3),3)=0.8;
end

surfPoints = getSurfacePoints(mask3M_PTV_LN_cut);
surf3M = repmat(logical(0), size(mask3M_PTV_LN_cut));
for i=1:size(surfPoints,1)
      scan3M_rgb(surfPoints(i,1),surfPoints(i,2), surfPoints(i,3),1)=0.3;
      scan3M_rgb(surfPoints(i,1),surfPoints(i,2), surfPoints(i,3),2)=0.3;
      scan3M_rgb(surfPoints(i,1),surfPoints(i,2), surfPoints(i,3),3)=0.1;
end

surfPoints = getSurfacePoints(mask3M_PTV_Spleen_cut);
surf3M = repmat(logical(0), size(mask3M_PTV_Spleen_cut));
for i=1:size(surfPoints,1)
      scan3M_rgb(surfPoints(i,1),surfPoints(i,2), surfPoints(i,3),1)=1;
      scan3M_rgb(surfPoints(i,1),surfPoints(i,2), surfPoints(i,3),2)=1;
      scan3M_rgb(surfPoints(i,1),surfPoints(i,2), surfPoints(i,3),3)=1;
end

surfPoints = getSurfacePoints(mask3M_PTV_Liver_cut);
surf3M = repmat(logical(0), size(mask3M_PTV_Liver_cut));
for i=1:size(surfPoints,1)
      scan3M_rgb(surfPoints(i,1),surfPoints(i,2), surfPoints(i,3),1)=0.5;
      scan3M_rgb(surfPoints(i,1),surfPoints(i,2), surfPoints(i,3),2)=0.8;
      scan3M_rgb(surfPoints(i,1),surfPoints(i,2), surfPoints(i,3),3)=0;
end

surfPoints = getSurfacePoints(mask3M_Esophagus_cut);
surf3M = repmat(logical(0), size(mask3M_Esophagus_cut));
for i=1:size(surfPoints,1)
      scan3M_rgb(surfPoints(i,1),surfPoints(i,2), surfPoints(i,3),1)=0;
      scan3M_rgb(surfPoints(i,1),surfPoints(i,2), surfPoints(i,3),2)=0.8;
      scan3M_rgb(surfPoints(i,1),surfPoints(i,2), surfPoints(i,3),3)=0.9;
end

surfPoints = getSurfacePoints(mask3M_Heart_cut);
surf3M = repmat(logical(0), size(mask3M_Heart_cut));
for i=1:size(surfPoints,1)
      scan3M_rgb(surfPoints(i,1),surfPoints(i,2), surfPoints(i,3),1)=0.9;
      scan3M_rgb(surfPoints(i,1),surfPoints(i,2), surfPoints(i,3),2)=0;
      scan3M_rgb(surfPoints(i,1),surfPoints(i,2), surfPoints(i,3),3)=0.9;
end

surfPoints = getSurfacePoints(mask3M_Body_cut);
% Get surface mask for structNum
surf3M = repmat(logical(0), size(mask3M_Body_cut));
for i=1:size(surfPoints,1)
    scan3M_rgb(surfPoints(i,1),surfPoints(i,2), surfPoints(i,3),1)=0;
    scan3M_rgb(surfPoints(i,1),surfPoints(i,2), surfPoints(i,3),2)=1;
    scan3M_rgb(surfPoints(i,1),surfPoints(i,2), surfPoints(i,3),3)=0;
end

surfPoints = getSurfacePoints(mask3M_PTV_VExP_cut);
% Get surface mask for structNum
surf3M = repmat(logical(0), size(mask3M_PTV_VExP_cut));
for i=1:size(surfPoints,1)
    scan3M_rgb(surfPoints(i,1),surfPoints(i,2), surfPoints(i,3),1)=0;
    scan3M_rgb(surfPoints(i,1),surfPoints(i,2), surfPoints(i,3),2)=0;
    scan3M_rgb(surfPoints(i,1),surfPoints(i,2), surfPoints(i,3),3)=1;
end

surfPoints = getSurfacePoints(mask3M_Lungs_cut);
% Get surface mask for structNum
surf3M = repmat(logical(0), size(mask3M_Lungs_cut));
for i=1:size(surfPoints,1)
    scan3M_rgb(surfPoints(i,1),surfPoints(i,2), surfPoints(i,3),1)=0;
    scan3M_rgb(surfPoints(i,1),surfPoints(i,2), surfPoints(i,3),2)=1;
    scan3M_rgb(surfPoints(i,1),surfPoints(i,2), surfPoints(i,3),3)=1;
end

surfPoints = getSurfacePoints(mask3M_GIUpper_cut);
% Get surface mask for structNum
surf3M = repmat(logical(0), size(mask3M_GIUpper_cut));
for i=1:size(surfPoints,1)
    scan3M_rgb(surfPoints(i,1),surfPoints(i,2), surfPoints(i,3),1)=0.8;
    scan3M_rgb(surfPoints(i,1),surfPoints(i,2), surfPoints(i,3),2)=0.2;
    scan3M_rgb(surfPoints(i,1),surfPoints(i,2), surfPoints(i,3),3)=0.1;
end

surfPoints = getSurfacePoints(mask3M_Avoid1_cut);
% Get surface mask for structNum
surf3M = repmat(logical(0), size(mask3M_Avoid1_cut));
for i=1:size(surfPoints,1)
    scan3M_rgb(surfPoints(i,1),surfPoints(i,2), surfPoints(i,3),1)=0.1;
    scan3M_rgb(surfPoints(i,1),surfPoints(i,2), surfPoints(i,3),2)=0.9;
    scan3M_rgb(surfPoints(i,1),surfPoints(i,2), surfPoints(i,3),3)=0.1;
end

surfPoints = getSurfacePoints(mask3M_Breasts_cut);
% Get surface mask for structNum
surf3M = repmat(logical(0), size(mask3M_Breasts_cut));
for i=1:size(surfPoints,1)
    scan3M_rgb(surfPoints(i,1),surfPoints(i,2), surfPoints(i,3),1)=0.9;
    scan3M_rgb(surfPoints(i,1),surfPoints(i,2), surfPoints(i,3),2)=0.5;
    scan3M_rgb(surfPoints(i,1),surfPoints(i,2), surfPoints(i,3),3)=0.7;
end

figure()
imshow3D(scan3M_rgb)

%%

rgbImage = cat(4, dose3M_cut/28, dose3M_cut/28, dose3M_cut/28);

surfPoints = getSurfacePoints(mask3M_PTV_Ribs_cut);
surf3M = repmat(logical(0), size(mask3M_PTV_Ribs_cut));
for i=1:size(surfPoints,1)
      rgbImage(surfPoints(i,1),surfPoints(i,2), surfPoints(i,3),1)=1;
      rgbImage(surfPoints(i,1),surfPoints(i,2), surfPoints(i,3),2)=0;
      rgbImage(surfPoints(i,1),surfPoints(i,2), surfPoints(i,3),3)=0;
end

surfPoints = getSurfacePoints(mask3M_PTV_VExP_cut);
% Get surface mask for structNum
surf3M = repmat(logical(0), size(mask3M_PTV_VExP_cut));
for i=1:size(surfPoints,1)
    rgbImage(surfPoints(i,1),surfPoints(i,2), surfPoints(i,3),1)=0;
    rgbImage(surfPoints(i,1),surfPoints(i,2), surfPoints(i,3),2)=0;
    rgbImage(surfPoints(i,1),surfPoints(i,2), surfPoints(i,3),3)=1;
end

figure()
imshow3D(rgbImage)

%% cut the mask data into 2D slices and multiple channels
[x,y,z]=size(dose3M_cut);
clear structset doseset;
%depth = 1; % every 4 slices form a training dataset
%interval = 2;
%data_num = floor((z-depth)/interval);
clear structset_2d_channel doseset_2d
for i=1:z
%    start = 1+(i-1)*interval;
    structset_2d_channel(i,:,:,1) = mask3M_Body_cut(:,:,i);
    structset_2d_channel(i,:,:,2) = mask3M_PTV_Ribs_cut(:,:,i);
    structset_2d_channel(i,:,:,3) = mask3M_PTV_VExP_cut(:,:,i);
    structset_2d_channel(i,:,:,4) = mask3M_PTV_SpCord_cut(:,:,i);
    structset_2d_channel(i,:,:,5) = mask3M_PTV_LN_cut(:,:,i);
    structset_2d_channel(i,:,:,6) = mask3M_PTV_Spleen_cut(:,:,i);
    structset_2d_channel(i,:,:,7) = mask3M_PTV_Liver_cut(:,:,i);
    structset_2d_channel(i,:,:,8) = mask3M_Lungs_cut(:,:,i);
    structset_2d_channel(i,:,:,9) = mask3M_Heart_cut(:,:,i);
    structset_2d_channel(i,:,:,10) = mask3M_Esophagus_cut(:,:,i);
    structset_2d_channel(i,:,:,11) = mask3M_GIUpper_cut(:,:,i);
    structset_2d_channel(i,:,:,12) = mask3M_Breasts_cut(:,:,i);
    structset_2d_channel(i,:,:,13) = mask3M_Avoid1_cut(:,:,i);
    doseset_2d(i,:,:) = dose3M_cut(:,:,i);
end

%% save the dataset
%save('structset_2d_channel_TOMPODUNG.mat','structset_2d_channel')
%save('doseset_2d_TOMPODUNG.mat','doseset_2d')
save('data_Kane_NEW.mat','structset_2d_channel', 'doseset_2d')
display('done')

%% Save the PTV as a whole structure
clear all
load('data_Tran.mat')
z = size(structset_2d_channel,1)
for i=1:z
%    start = 1+(i-1)*interval;
    structset_2d_channel(i,:,:,11) = structset_2d_channel(i,:,:,2) | structset_2d_channel(i,:,:,3) | structset_2d_channel(i,:,:,4) | structset_2d_channel(i,:,:,5) | structset_2d_channel(i,:,:,6)| structset_2d_channel(i,:,:,7);
    structset_2d_channel(i,:,:,2:7) = [];
end

%% 
volVox = planC{indexS.scan}(1).uniformScanInfo.grid1Units * planC{indexS.scan}(1).uniformScanInfo.grid2Units * planC{indexS.scan}(1).uniformScanInfo.sliceThickness;
planC{indexS.scan}(1).uniformScanInfo.grid1Units
planC{indexS.scan}(1).uniformScanInfo.grid2Units
planC{indexS.scan}(1).uniformScanInfo.sliceThickness
%%
index = find(mask_diff_OAR);
[x1,y1,z1] = ind2sub(size(mask_diff_OAR),index);
mask3M_OAR_highrisk=mask_diff_OAR;
dose_point = 0;
for i=1:length(x1)
    if (dose3M(x1(i),y1(i),z1(i))>=33)
        dose_point=dose_point+1
        continue;  
    end
    i
    M2D_diff = mask_diff_OAR(:,:,z1(i));
    M2D_sum = mask3M_sum(:,:,z1(i));
    M2D_PTV = mask3M_PTV(:,:,z1(i));
    if (length(find(M2D_PTV))==0)
        display('no PTV on same slice')
        mask3M_OAR_highrisk(x1(i),y1(i),z1(i))=false;
        continue;
    end
    
    [x2,y2] = ind2sub(size(M2D_PTV),find(M2D_PTV));
    PTV_center(1)= round(mean(x2));
    PTV_center(2)= round(mean(y2));
    % raytrace
    trace_contour = []; % 0 nothing, 1-OAR,2-PTV;
    trace_dose = [];
    b=1;
    if(PTV_center(1)<x1(i))
        b=-1;
    end
    trace_index=1;
    for j=x1(i):b:PTV_center(1)
        if(j==x1(i)) 
            continue;
        end
        x_coord = j;
        y_coord = round((PTV_center(2)-y1(i))/(PTV_center(1)-x1(i))*(j-x1(i)) +y1(i));
        trace_dose(trace_index)=dose3M(x_coord,y_coord,z1(i));
        if(M2D_sum(x_coord,y_coord))
            trace_contour(trace_index)=1;
        elseif (M2D_PTV(x_coord,y_coord))
            trace_contour(trace_index)=2;
        else
            trace_contour(trace_index)=0;
        end
        trace_index = trace_index+1;
    end
    flag=false;
    for j=1:trace_index-1
        if(trace_contour(j)==1 && trace_dose(j)<33)
            mask3M_OAR_highrisk(x1(i),y1(i),z1(i))=false;
            break;
        end
        if(trace_contour(j)==2)
            if (~flag)
                flag = true;
                if(trace_dose(j)>=50)
                    mask3M_OAR_highrisk(x1(i),y1(i),z1(i))=false;
                    break;
                else
                    dose_est = 50/trace_dose(j) * dose3M(x1(i),y1(i),z1(i));
                    if (dose_est<33)
                        mask3M_OAR_highrisk(x1(i),y1(i),z1(i))=false;
                        
                        break;
                    else
                        display('find a high dose risk point')
                    end
                    
                end
            end
        end
    end
    if (~flag)
       mask3M_OAR_highrisk(x1(i),y1(i),z1(i))=false; 
    end
        
    
end

%% conver the high risk mask to structure
strname = 'high risk stomach';
isUniform = 1;
planC = maskToCERRStructure(mask3M_OAR_highrisk, isUniform, scanNum, strname, planC)

%%

mask3M_Body_cut = mask3M_Body_resamp((center_x-127): (center_x+128),(center_y-127): (center_y+128),(center_z-z_range): (center_z+z_range));
mask3M_PTV_Ribs_cut = mask3M_PTV_Ribs_resamp((center_x-127): (center_x+128),(center_y-127): (center_y+128),(center_z-z_range): (center_z+z_range));
mask3M_PTV_VExP_cut = mask3M_PTV_VExP_resamp((center_x-127): (center_x+128),(center_y-127): (center_y+128),(center_z-z_range): (center_z+z_range));
mask3M_PTV_SpCord_cut = mask3M_PTV_SpCord_resamp((center_x-127): (center_x+128),(center_y-127): (center_y+128),(center_z-z_range): (center_z+z_range));
mask3M_PTV_LN_cut = mask3M_PTV_LN_resamp((center_x-127): (center_x+128),(center_y-127): (center_y+128),(center_z-z_range): (center_z+z_range));
mask3M_PTV_Spleen_cut = mask3M_PTV_Spleen_resamp((center_x-127): (center_x+128),(center_y-127): (center_y+128),(center_z-z_range): (center_z+z_range));
mask3M_PTV_Liver_cut = mask3M_PTV_Liver_resamp((center_x-127): (center_x+128),(center_y-127): (center_y+128),(center_z-z_range): (center_z+z_range));
mask3M_Heart_cut = mask3M_Heart_resamp((center_x-127): (center_x+128),(center_y-127): (center_y+128),(center_z-z_range): (center_z+z_range));
mask3M_Lungs_cut = mask3M_Lungs_resamp((center_x-127): (center_x+128),(center_y-127): (center_y+128),(center_z-z_range): (center_z+z_range));
mask3M_Esophagus_cut = mask3M_Esophagus_resamp((center_x-127): (center_x+128),(center_y-127): (center_y+128),(center_z-z_range): (center_z+z_range));
dose3M_cut = dose3M_resamp((center_x-127): (center_x+128),(center_y-127): (center_y+128),(center_z-z_range): (center_z+z_range));

scan3M_cut = scan3M_resamp((center_x-127): (center_x+128),(center_y-127): (center_y+128),(center_z-79): (center_z+80));


