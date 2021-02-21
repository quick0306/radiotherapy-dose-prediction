clear all
load('GARATE_TMI.mat')
global planC
indexS = planC{end};
structureListC = {planC{indexS.structures}.structureName};
%%
info = dicominfo('RD.0PDUJNzR0zPBISeRip4As9616.TMI_Body.dcm');
dose = dicomread('RD.0PDUJNzR0zPBISeRip4As9616.TMI_Body.dcm');
dose = squeeze(dose);
dose = double(dose);
dose = dose * info.DoseGridScaling;
max(max(max(dose)));
%%
doseStruct = planC{indexS.dose}(1);
doseArray = doseStruct.doseArray;
planC{indexS.dose}(1).doseArray = dose;
save('GARATE_TMI.mat','planC')
display('done')