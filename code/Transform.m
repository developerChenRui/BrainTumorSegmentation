function [ TransformedImage ] = Transform( mu_s,Image)
V=int16(Image);
s1=1;s2=4095;
p1_i = 1;

[length,width,height] = size(V);
% every pixel
His=zeros(1,s2);
sumNot0 = 0;
for i=1:length
    for j=1:width
        for k=1:height
            if V(i,j,k)>4095
                V(i,j,k)=4095;
            end
            if V(i,j,k)>0
                His(V(i,j,k))=His(V(i,j,k))+1;
                sumNot0=sumNot0+1;
            end
        end
    end
end
% set m1 = 1 and calculate the num of non-0 



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
[MaxValue,p2_i]= max(round(c*10000)>=9980);
[MaxValue,mu_i]=max(His);

for i =1:length
    for j=1:width
        for k=1:height            
            if V(i,j,k)>p2_i    % cannot set it to 0 !!!!!               
               V(i,j,k)=int16(mu_s+((double(V(i,j,k)-mu_i)...
                   *double(s2-mu_s))/double(p2_i-mu_i)));
               continue;
         elseif V(i,j,k)>=p1_i&&V(i,j,k)<=mu_i
                V(i,j,k)=int16(mu_s+((double(V(i,j,k)-mu_i)...
                    *double(s1-mu_s))/double(p1_i-mu_i)));
                continue;

            elseif V(i,j,k)>mu_i&&V(i,j,k)<=p2_i

                V(i,j,k)=int16(mu_s+((double(V(i,j,k)-mu_i)...
                    *double(s2-mu_s))/double(p2_i-mu_i)));
                continue;

            end

        end
    end
end
TransformedImage = V;
end

