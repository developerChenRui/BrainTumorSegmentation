begin=1;ending=22;
Dir_T1 = '/Users/chenrui/Desktop/BRATS2015/BRATS2015_Training/HGG_n4_norm_slice/T1';
Dir_T1c = '/Users/chenrui/Desktop/BRATS2015/BRATS2015_Training/HGG_n4_norm_slice/T1c';
Dir_T2 = '/Users/chenrui/Desktop/BRATS2015/BRATS2015_Training/HGG_n4_norm_slice/T2';
Dir_Flair = '/Users/chenrui/Desktop/BRATS2015/BRATS2015_Training/HGG_n4_norm_slice/Flair';
Dir_label = '/Users/chenrui/Desktop/BRATS2015/BRATS2015_Training/HGG_n4_norm_slice/labels';
traindata = zeros([22*155 4 240 240]);
train_label = zeros([22*155 1 240 240]);
traindata(:,1,:,:)=1; % channel = T1
traindata(:,2,:,:)=2; % channel = T1c
traindata(:,3,:,:)=3; % channel = T2
traindata(:,4,:,:)=4; % channel = Flair
train_label(:,1,:,:) = 1;
for p=1:10
%mat2hdf5 [N channel ]
if p==3
    check=0;
end
    for num=begin:ending
        for i=1:155
            filename_T1 = [Dir_T1 '/' int2str(num) '_' int2str(i) '.mat'];
            V_t1 = load(filename_T1);
            traindata(((num-(p-1)*22)-1)*155+i,1,:,:)= V_t1.V;
        end
    end
    for num=begin:ending
        for i=1:155
            filename_T1c = [Dir_T1c '/' int2str(num) '_' int2str(i) '.mat'];
            V_T1c = load(filename_T1c);
            traindata(((num-(p-1)*22)-1)*155+i,2,:,:)= V_T1c.V;
        end
    end
    for num=begin:ending
        for i=1:155
            filename_T2 = [Dir_T2 '/' int2str(num) '_' int2str(i) '.mat'];
            V_T2 = load(filename_T2);
            traindata(((num-(p-1)*22)-1)*155+i,3,:,:)= V_T2.V;
        end
    end    
    for num=begin:ending
        for i=1:155
            filename_Flair = [Dir_Flair '/' int2str(num) '_' int2str(i) '.mat'];
            V_Flair = load(filename_Flair);
            traindata(((num-(p-1)*22)-1)*155+i,4,:,:)= V_Flair.V;
        end
    end    
    for num=begin:ending
        for i=1:155
            filename_label = [Dir_label '/' int2str(num) '_' int2str(i) '.mat'];
            V_label = load(filename_label);
            train_label(((num-(p-1)*22)-1)*155+i,1,:,:)= V_label.V;
        end
    end
    % create database

    h5create(['/Users/chenrui/Desktop/BRATS2015/BRATS2015_Training/HGG_n4_norm_slice_h5/train_' int2str(p) '.hdf5'],'/data',size(traindata),'Datatype','int16');

    h5create(['/Users/chenrui/Desktop/BRATS2015/BRATS2015_Training/HGG_n4_norm_slice_h5/train_' int2str(p) '.hdf5'],'/label',size(train_label),'Datatype','int16');

    h5write(['/Users/chenrui/Desktop/BRATS2015/BRATS2015_Training/HGG_n4_norm_slice_h5/train_' int2str(p) '.hdf5'],'/data',traindata);

    h5write(['/Users/chenrui/Desktop/BRATS2015/BRATS2015_Training/HGG_n4_norm_slice_h5/train_' int2str(p) '.hdf5'],'/label',train_label);
    
    begin=ending+1;
    ending=begin+21;
end