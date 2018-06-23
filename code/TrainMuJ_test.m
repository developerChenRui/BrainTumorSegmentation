function [mu_J] = TrainMuJ_test( Image,type )
V = int16(Image);
% map His_without0's[p1_j p2_j]to[s1=1 s2=4095]
s1=1;s2=4095;
% histogram of V
[length,width,height] = size(V);
% every pixel
His=zeros(1,s2);
sumNot0 = 0;
for i=1:length
    for j=1:width
        for k=1:height
            if V(i,j,k)>s2
                V(i,j,k)=0;
            end
            if V(i,j,k)>0
            His(V(i,j,k))=His(V(i,j,k))+1;
            sumNot0 = sumNot0+1;
            end
        end
    end
end
% set m1 = 1 and calculate the num of non-0 

% for i = 1:height
%     sumNot0=sumNot0+nnz(V(:,:,i));
% end
% 
% check_4095=0;
% for i = 1:height
%     check_4095=check_4095+nnz(V(:,:,i)>4095);
% end
% b=check_4095
% percent start from 2 because m1 = g(v)>0
p = zeros(1,s2);
for i=1:s2
    p(i)=His(i)/(sumNot0 * 1.0);
end
% accumulated percent
c = zeros(1,s2);
c(1)=p(1);
for i=2:s2
    c(i) = c(i-1) + p(i);
end
% fixed values from the paper
% pc_1 = 0;
% pc_2 = 99.8;
% find the p1_j p2_j find the m1
p1_j = 1;
[MaxValue,p2_j]= max(round(c*10000)>=9980);
% p2_j=max(max(max(V)));
if strcmp(type,'mode')
[MaxValue,mu_j]=max(His);
end

if strcmp(type,'median')
    for threshold =1:10
    [MedianValue,mu_j]=max(abs(round(c*1000)-500)<=threshold);
    if mu_j~=1
        break;
    end
    if mu_j==0
        error='e';
    end
    
    end
end

% check_4095=0;
% for i = 1:height
%     check_4095=check_4095+nnz(V(:,:,i)>4095);
% end
% b=check_4095

for i=1:length
    for j=1:width
        for k=1:height            
             if V(i,j,k)>p2_j                 
                 V(i,j,k)=0;
%                  int16(double(V(i,j,k+1)+V(i,j+1,k)+V(i,j+1,k+1)+...
%                     V(i+1,j,k)+V(i+1,j,k+1)+V(i+1,j+1,k)+V(i+1,j+1,k+1))./7);
                
             end  
             if V(i,j,k)>0
                 sub=V(i,j,k)-p1_j;
                 mul=double(sub)*double(s2-s1);
                 chu=mul/(p2_j-p1_j);
                 plus=s1+chu;
                 V(i,j,k)=int16(plus); 
             end
        end
    end
end
% 
% check_4095=0;
% for i = 1:height
%     check_4095=check_4095+nnz(V(:,:,i)>4095);
% end
% a=check_4095
His_after=zeros(1,s2);
sumNot0 = 0;
for i=1:length
    for j=1:width
        for k=1:height  
            if V(i,j,k)>0
            His_after(V(i,j,k))=His_after(V(i,j,k))+1; 
            sumNot0=sumNot0+1;
            end
        end
    end
end

% percent start from 2 because m1 = g(v)>0
p_after = zeros(1,s2);
for i=1:s2
    p_after(i)=His_after(i)/(sumNot0 * 1.0);
end
% accumulated percent
c_after = zeros(1,s2);
c_after(1)=p_after(1);
for i=2:s2
    c_after(i) = c_after(i-1) + p_after(i);
end

if strcmp(type,'mode')
[MaxValue,mu_J] = max(His_after);
end
if strcmp(type,'median')
        for threshold =1:20
    [MedianValue,mu_J]=max(((round(c_after*1000)-500)<=threshold)&...
        ((round(c_after*1000)-500)>=0));
    if mu_J~=1
        break;
    end
        end
end
end




