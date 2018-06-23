dataDir = '/Users/chenrui/Desktop/BRATS2015/BRATS2015_Training/HGG_n4_norm/';
saveorg = '/Users/chenrui/Desktop/BRATS2015/BRATS2015_Training/HGG_n4_norm_slice/';
menu_1 = dir(dataDir);
T1_patient = 0;
T1c_patient = 0;
T2_patient = 0;
F_patient = 0;
gt=0;
for i =1:max(size(menu_1))
    if strcmp(menu_1(i).name(1),'l')||strcmp(menu_1(i).name(1),'T')||strcmp(menu_1(i).name(1),'F')
        file=dir(fullfile(dataDir,menu_1(i).name));
        for j=1:max(size(file))
            if strcmp(file(j).name(end),'t')
                if strcmp(menu_1(i).name,'T1')
                T1_patient = T1_patient+1;
                goal = load(fullfile(dataDir,menu_1(i).name,file(j).name));
                goal = goal.TransT1;
                savepath = [saveorg 'T1/' int2str(T1_patient) '_'];
                end
                if strcmp(menu_1(i).name,'T1c')
                T1c_patient = T1c_patient+1;
                goal = load(fullfile(dataDir,menu_1(i).name,file(j).name));
                goal = goal.TransT1c;
                savepath = [saveorg 'T1c/' int2str(T1c_patient) '_'];
                end
                if strcmp(menu_1(i).name,'T2')
                T2_patient = T2_patient+1;
                goal = load(fullfile(dataDir,menu_1(i).name,file(j).name));
                goal = goal.TransT2;
                savepath = [saveorg 'T2/' int2str(T2_patient) '_'];
                end
                if strcmp(menu_1(i).name,'Flair')
                F_patient = F_patient+1;
                goal = load(fullfile(dataDir,menu_1(i).name,file(j).name));
                goal = goal.TransF;
                savepath = [saveorg 'Flair/' int2str(F_patient) '_'];
                end
                if strcmp(menu_1(i).name,'labels')
                gt = gt+1;
                goal = load(fullfile(dataDir,menu_1(i).name,file(j).name));
                goal = goal.V;%TransF;
                savepath = [saveorg 'labels/' int2str(gt) '_'];
                end
                
                [length,width,height]=size(goal);
                for h=1:height
                    V = goal(:,:,h);
                    savefinal = [savepath int2str(h) '.mat' ];
                    save(savefinal,'V','-mat');
                end
                
            end            
        end
    end
end





