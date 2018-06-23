filedirt1 = '/Users/chenrui/Desktop/BRATS2015/BRATS2015_Training/HGG_n4_slice/T1';
filedirt1c = '/Users/chenrui/Desktop/BRATS2015/BRATS2015_Training/HGG_n4_slice/T1c';
filedirt2 = '/Users/chenrui/Desktop/BRATS2015/BRATS2015_Training/HGG_n4_slice/T2';
filedirflair = '/Users/chenrui/Desktop/BRATS2015/BRATS2015_Training/HGG_n4_slice/Flair';
filedirtgt = '/Users/chenrui/Desktop/BRATS2015/BRATS2015_Training/HGG_n4_slice/Labels';

figure
subplot(2,3,1)
t1=load([filedirt1 '/' '5_90.mat']);
t1=t1.V;

imshow(t1,[])

subplot(2,3,2)
t1c=load([filedirt1c '/' '5_90.mat']);
t1c=t1c.V;

imshow(t1c,[])

subplot(2,3,3)
t2=load([filedirt2 '/' '5_90.mat']);
t2=t2.V;

imshow(t2,[])

subplot(2,3,4)
f=load([filedirflair '/' '5_90.mat']);
f=f.V;

imshow(f,[])

subplot(2,3,5)
gt=load([filedirtgt '/' '5_90.mat']);
gt=gt.V;

imshow(gt,[])


