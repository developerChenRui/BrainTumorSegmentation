D = mha_read_volume('/Users/chenrui/Desktop/brats_2013_pat0001_1/VSD.Brain.XX.O.MR_T1.54513/VSD.Brain.XX.O.MR_T1.54513.mha');
org =squeeze(D(:,:,25));
for i = 25:10:125
org2=squeeze(D(:,:,i));
SliceMap = cat(2,org,org2);
org=SliceMap;
end
imshow(SliceMap,[]);title('T1');
title('Horizontal Slices');