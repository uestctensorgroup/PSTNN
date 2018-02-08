%--------------Brief description-------------------------------------------
% This demo contains the implementation of the experiment for high order
% data comletion
% More details in:
% Tai-Xiang Jiang, Ting-Zhu Huang, Xi-Le Zhao, Liang-Jian Deng;
% ''A novel nonconvex approach to recover the low-tuba-rank tensor data:
% when t-SVD meets PSSV'' submitted to Applied Mathematical Modelling
% (AMM).
% Contact: taixiangjiang@gmail.com
% Date: 7th Feb. 2018


addpath(genpath(cd))
clear all;clc;close all;
rng(1)
%% load the testing high order data
data_num  = 1;   %     1->MRI data;     2->video data;
switch data_num
    case 1
        data_name = 'MRI.mat';
    case 2
        data_name = 'salesman.mat';
end
load(dataname);
X = X(:,:,51:150);%cropping
[n1,n2,n3] = size(X);



%%
p = 0.2; % The ratio of observed entries
maxP = max(abs(X(:)));
omega = find(rand(n1*n2*n3,1)<p);
M = zeros(n1,n2,n3);
M(omega) = X(omega);

%% index of the observed data
PSNR0 = psnr(M,X);
SSIM0 = ssim(M,X);

%% parameter setting
opts.mu = 10^-3;
opts.tol = 1e-7;
opts.rho = 1.2;
opts.max_iter = 200;
opts.DEBUG = 0;
opts.max_mu = 1e10;

%% Tensor completion by HaLRTC
    
    alpha = [1, 1, 1];
    alpha = alpha / sum(alpha);
    
    [Xhat1,~] = HaLRTC(M, omega, alpha, opts.mu, opts.max_iter, opts.tol);%lrtc_snn(M,omega,alpha,opts);
    
    Xhat1 = max(Xhat1,0);
    Xhat1 = min(Xhat1,maxP);
    SSIM1 = ssim(Xhat1,X);
    PSNR1 = psnr(Xhat1,X);

%%  Tensor completion based on TNN
    
    [Xhat2,~,~,~] = LRTC_TNN(M,omega,opts,X);

    Xhat2 = max(Xhat2,0);
    Xhat2 = min(Xhat2,maxP);
    RSE2 = norm(X(:)-Xhat2(:))/norm(X(:));
    SSIM2 = ssim(Xhat2,X);%,maxP)
    PSNR2 = psnr(Xhat2,X,maxP);%);%,maxP);

%% Tensor completion based on PSTNN

        [rankN,~] = prox_rankN(X,0.01);%n*ones(1,n3);%
        [Xhat3,~,~,~] =  LRTC_PSTNN(M,omega,opts,rankN,X);%,Xhat2);%
        
        Xhat3 = max(Xhat3,0);
        Xhat3 = min(Xhat3,maxP);
        SSIM3 = ssim(Xhat3,X);
        PSNR3 = psnr(Xhat3,X);

%% illustration of the results;
  
frame = 50; % the number of the frame
figure(data_num)
subplot(2,3,1);
imshow(X(:,:,frame));title('Original data');
subplot(2,3,2);
imshow(M(:,:,frame));title('Observed data');
subplot(2,3,3);
imshow(Xhat3(:,:,frame));title('Results by PSTNN');
subplot(2,3,5);
imshow(Xhat1(:,:,frame));title('Results by FaLRTC');
subplot(2,3,6);
imshow(Xhat2(:,:,frame));title('Results by TNN');

disp(['image name : ' data_name]);
disp(['Index || observed || FaLRTC ||   TNN   ||   PSTNN ']);
disp(['PSNR || ' num2str(PSNR0) ' ||  ' num2str(PSNR1) ' ||  ' num2str(PSNR2) ' ||  ' num2str(PSNR3) ]);
disp(['SSIM || ' num2str(SSIM0) ' ||  ' num2str(SSIM1) ' ||  ' num2str(SSIM2) ' ||  ' num2str(SSIM3)  ]);




