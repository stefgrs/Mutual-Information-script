function [mi, j_ent] = MIpair2(x,y,k)
%% Implementation of Kraskov Estimator of mutual information (Kraskov et al Phys Rev E 2004).
%% Written by Stefania Garasto, laboratory of Simon Schultz, Imperial College London
%% copyright S Garasto 2014-16.
% Input:
% x,y= matrices of size "D x N". They are N realizations of a random
% variable Z=(X,Y), where the two variables X and Y have dimension D.
% i.e. for two time series with T points: D=1; N=T.
% k= size of the neighbourhood (k-nearest neighbour)
% Output:
% mi= mutual information between X and Y
% j_ent= joint entropy

%% Check input
if (mean(size(x)==size(y))~=1) || ~isreal(x) || ~isreal(y)
    warning('the input arguments must be of equal size')
    mi=NaN;
    return;
end
    
%% gather parameters
[dim, n]= size(x);
% dim= internal dimension of each variable (dimension of the x and of the y
% space)
% n= numbers of realizations
if dim>n
    x=x';
    y=y';
    [dim, n]= size(x);
end

if isempty(k)
    k=3;
end

%% normalize
x=x-repmat(mean(x,2),1,n);
x=x./repmat(std(x,0,2),1,n);

y=y-repmat(mean(y,2),1,n);
y=y./repmat(std(y,0,2),1,n);

%% find the smallest rectangle epsx,epsy that contains k nearest neighbours 
% for each z_i=(x_i,y_i) and related quantities
% Distmat is a matrix where (i,j) element is the distance between z_i and z_j

Distmatx= pdist(x','chebychev');
Distmaty= pdist(y','chebychev');
Distmat= max([Distmatx; Distmaty]);

Distmatx= squareform(Distmatx);
Distmaty= squareform(Distmaty);
Distmat= squareform(Distmat);

% sort distance matrices
[Distmatsort, P]= sort(Distmat,2);

epsz= Distmatsort(:, k+1);
Distmatsortx = zeros(size(Distmatsort));
Distmatsorty = zeros(size(Distmatsort));

for ii=1:n
    Distmatsortx(ii,:)= Distmatx(ii,P(ii,:));
    Distmatsorty(ii,:)= Distmaty(ii,P(ii,:));
end

epsx= Distmatsortx(:, k+1);
epsy= Distmatsorty(:, k+1);

for ii=1:n
    flagy=1;
    for jj=k:-1:2
        if Distmatsortx(ii,jj)>epsx(ii);
            epsx(ii)= Distmatsortx(ii,jj);
            flagy=0;
        end
    end
    if flagy
        for jj=k:-1:2
            if Distmatsorty(ii,jj)>epsy(ii);
                epsy(ii)= Distmatsorty(ii,jj);
            end
        end
    end
end
        
nx= arrayfun(@(jj) sum(Distmatx(jj,:)<=epsx(jj))-1, 1:n)';
ny= arrayfun(@(jj) sum(Distmaty(jj,:)<=epsy(jj))-1, 1:n)';

meanpsix= 1/n * sum(psi(nx)); 
meanpsiy= 1/n * sum(psi(ny));

%% MI
mi = psi(k) - 1/k - meanpsix - meanpsiy + psi(n);

% joint entropy
j_ent= -psi(k)+ psi(n) + 2*dim*mean(epsz);
