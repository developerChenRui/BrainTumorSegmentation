cd(fileparts(which('DataProcess.m')));dataDir = '/Users/chenrui/Desktop/BRATS2015/BRATS2015_Training/HGG';
menu_1 = dir('/Users/chenrui/Desktop/BRATS2015/BRATS2015_Training/HGG');
num_T1 = 0;
num_T1c = 0;
num_T2 = 0;
num_Flair = 0;
num_labels = 0;
T1_sum = 0;
T1_patient =0;            
T1c_sum = 0;
T1c_patient =0;
T2_sum = 0;
T2_patient =0;
F_sum = 0;
F_patient =0; 
% for n=1:size(menu_1)
%     dex = menu_1(n);
%     if strcmp(dex.name(1),'b')
%         menu_2 = dir(dex.name);        
%         for i=1:max(size(menu_2))
%             if max(size(menu_2(i).name))>5 && strcmp(menu_2(i).name(1),'V')                       
%             dex_2 = menu_2(i).name;
%             file = dir(fullfile(dataDir,dex.name,dex_2));
%             for dex_goal = 1:size(file)
%                 if max(size(file(dex_goal).name))>5
%                     if strcmp(file(dex_goal).name(end),'a') % for labels
%                         goalname = file(dex_goal).name;
%                     end
%                     if strcmp(file(dex_goal).name(end-4),'r')
%                         goalname = file(dex_goal).name;
%                     end 
%                 end
%             end
%             fullgoal = fullfile(dataDir,dex.name,dex_2,goalname);
%             V = int16(mha_read_volume(fullgoal));
%             %% T1
% 
%             if strcmp(dex_2(end-6),'1')
% %                 num_T1 = num_T1+1;
% %                 savepath = ['/Users/chenrui/Desktop/BRATS2015/BRATS2015_Training/HGG_n4/T1/' int2str(num_T1) '.mat'];
% %                 save(savepath,'V','-mat');
%                 [T1_u]=TrainMuJ_test(V,'mode');
%                 T1_sum = T1_sum + T1_u;
% 
%             end
%             %% T1c
% 
%             if strcmp(dex_2(end-6),'c')
% %                num_T1c = num_T1c+1;
% %                 savepath = ['/Users/chenrui/Desktop/BRATS2015/BRATS2015_Training/HGG_n4/T1c/' int2str(num_T1c) '.mat'];
% %                 save(savepath,'V','-mat');
%                 [T1c_u]=TrainMuJ_test(V,'mode');
%                 T1c_sum = T1c_sum + T1c_u;
%               
%             end
%             %% T2
%             
%             if strcmp(dex_2(end-6),'2')
% %                num_T2 = num_T2+1;
% %                 savepath = ['/Users/chenrui/Desktop/BRATS2015/BRATS2015_Training/HGG_n4/T2/' int2str(num_T2) '.mat'];
% %                 save(savepath,'V','-mat');
%                 [T2_u]=TrainMuJ_test(V,'mode');
%                 T2_sum = T2_sum + T2_u;
%               
%             end
%             %% Flair
%            
%             if strcmp(dex_2(end-6),'r')
% %                num_Flair = num_Flair+1;
% %                 savepath = ['/Users/chenrui/Desktop/BRATS2015/BRATS2015_Training/HGG_n4/Flair/' int2str(num_Flair) '.mat'];
% %                 save(savepath,'V','-mat');
%                 [F_u]=TrainMuJ_test(V,'mode');
%                 F_sum = F_sum + F_u;
%                
%             end
%             %% Labels
%             if strcmp(dex_2(end-6),'T')
% %                num_labels = num_labels+1;
% %                 savepath = ['/Users/chenrui/Desktop/BRATS2015/BRATS2015_Training/HGG_n4/labels/' int2str(num_labels) '.mat'];
% %                 save(savepath,'V','-mat');
%             end
%             end
%         end
%     end
% end
% u_T1 = T1_sum/220;
% u_T1c = T1c_sum/220;
% u_T2 = T2_sum/220;
% u_Flair = F_sum/220;

u_T1 = 2812.8;
u_T1c = 2019.8;
u_T2 = 1610.1;
u_Flair = 2067.7;

num_T1 = 0;
num_T1c = 0;
num_T2 = 0;
num_Flair = 0;
for n=1:size(menu_1)
    dex = menu_1(n);
    if strcmp(dex.name(1),'b')
        menu_2 = dir(dex.name);        
        for i=1:max(size(menu_2))
            if max(size(menu_2(i).name))>5 && strcmp(menu_2(i).name(1),'V')                       
            dex_2 = menu_2(i).name;
            file = dir(fullfile(dataDir,dex.name,dex_2));
            for dex_goal = 1:size(file)
                if max(size(file(dex_goal).name))>5
                    if strcmp(file(dex_goal).name(end),'a') % for labels
                        goalname = file(dex_goal).name;
                    end
                    if strcmp(file(dex_goal).name(end-4),'r')
                        goalname = file(dex_goal).name;
                    end 
                end
            end
            fullgoal = fullfile(dataDir,dex.name,dex_2,goalname);
            V = int16(mha_read_volume(fullgoal));
            %% T1

            if strcmp(dex_2(end-6),'1')
                num_T1 = num_T1+1;
                [ TransT1 ] = Transform(u_T1,V);
                savepath = ['/Users/chenrui/Desktop/BRATS2015/BRATS2015_Training/HGG_n4_norm/T1/' int2str(num_T1) '.mat'];
                save(savepath,'TransT1','-mat');

            end
            %% T1c

            if strcmp(dex_2(end-6),'c')
               num_T1c = num_T1c+1;
               [ TransT1c ] = Transform(u_T1c,V);
                savepath = ['/Users/chenrui/Desktop/BRATS2015/BRATS2015_Training/HGG_n4_norm/T1c/' int2str(num_T1c) '.mat'];
                save(savepath,'TransT1c','-mat');
               
            end
            %% T2
            
            if strcmp(dex_2(end-6),'2')
               num_T2 = num_T2+1;
               [ TransT2 ] = Transform(u_T2,V);              
                savepath = ['/Users/chenrui/Desktop/BRATS2015/BRATS2015_Training/HGG_n4_norm/T2/' int2str(num_T2) '.mat'];
                save(savepath,'TransT2','-mat');
              
            end
            %% Flair
           
            if strcmp(dex_2(end-6),'r')
               num_Flair = num_Flair+1;
               [ TransF ] = Transform(u_Flair,V);
                savepath = ['/Users/chenrui/Desktop/BRATS2015/BRATS2015_Training/HGG_n4_norm/Flair/' int2str(num_Flair) '.mat'];
                save(savepath,'TransF','-mat');
              
            end

            end
        end
    end
end







