% Main m file for equivalent Robust Precoding

clear
clc

addpath(genpath('quadriga_src/'));
addpath(genpath('fun/'));

t_start = datetime;

%% Load parameter
UserNum = 9;
RxAntNum = 1;
TxAntNum = 64; % vertical first, horizotal second, Polar last
SampleNum = 2000; % The number of subframes
IterNum = 20;
PowerConst = 1;

%% Load 5D input channel matrix H_freq (TxAntNum,RxAntNum,RBNum,SampleNum,UserNum)%%%
Speed = 5;
load(".\Channel\qd_ch_NLOS_20M_192_"+num2str(Speed) + "kmph.mat");
OSFactorH = 2; % The horizontal oversampling factor
OSFactorV = 2; % The vertical oversampling factor
BeamNum = TxAntNum * OSFactorH * OSFactorV;
MeanSparisity = 256;
Nh = 8;
Nv = 4;

EsN0 = 0:10:50; % 0 to 50 dB
RBNum = 1; % The number of RB used
max_SlotNum = 10;
SlotNum = 10; % Calculate the number of slots accroding to SampleNum

bn_list = [5 10 20 40 80 160]; % The number of blocks at each slot

% 数据-->mu
% mu-->p
for indx_bn = 1:length(bn_list)
    BlockNum = bn_list(indx_bn);
    Block_indx_list = 1:BlockNum;
    srs_indx_list = 1:BlockNum:SlotNum * BlockNum;
    
    % jake's model %——为了计算beta
    alpha = abs(besselj(0, 2 * pi * Speed * 1e3 * 3.5e9 * (BlockNum + Block_indx_list) * 1e-3 / (3600 * 3e8))).^2;
    beta = sqrt((mean(alpha))) * ones(UserNum, 1);
    beta0 = zeros(UserNum, 1);
    beta1 = ones(UserNum, 1);
    
   %% Channel Mean and Variance
    % -------- MFOCUSS -------- %——为了计算Ut, Omega
    [HFreqBeam, Ut, Omega] = CCMGenComp ...
        (H_freq(:, 1:RxAntNum, :, 1:max_SlotNum * BlockNum, :), OSFactorH, OSFactorV, BlockNum);
    HFreqBeam = HFreqBeam(:, :, 1:RBNum, 1:SlotNum , :);
    
   %% Precoding  
    % -------- RZF -------- %
    Sum_Rate_RZF = zeros(1, length(EsN0));
    for nSNR = 1:length(EsN0)
        PRZF{nSNR} = RZF_Precoder ...
            (H_freq(:, 1:RxAntNum, 1:RBNum, srs_indx_list, :), EsN0(nSNR), PowerConst);
        [Sum_Rate_RZF(nSNR), ~, ~] = CalSumRate ...
            (PRZF{nSNR}, H_freq(:, 1:RxAntNum, 1:RBNum, 1:SlotNum * BlockNum, :), EsN0(nSNR), BlockNum);
    end
    
        % -------- Robust -------- %——不用插值
    Sum_Rate_Robust = zeros(1, length(EsN0));
    for nSNR = 1:length(EsN0)
        PRobust_MFOCUSS{nSNR} = Robust_Precoder_MFOCUSS ...
            (PRZF{nSNR}, H_freq(:, 1:RxAntNum, 1:RBNum, srs_indx_list, :), Ut, Omega, beta, EsN0(nSNR), MeanSparisity, IterNum, PowerConst);
        mu_list_beta{nSNR} = get_mu_list(PRobust_MFOCUSS{nSNR}, H_freq(:, 1:RxAntNum, 1:RBNum, srs_indx_list, :), Ut, Omega, beta, EsN0(nSNR));
        [Sum_Rate_Robust(nSNR), ~, ~] = CalSumRate ...
            (PRobust_MFOCUSS{nSNR}, H_freq(:, 1:RxAntNum, 1:RBNum, 1:SlotNum * BlockNum, :), EsN0(nSNR), BlockNum);
    end

    % -------- Robust ML Optimal -------- %——利用差值好的mu，计算预编码矩阵
    Sum_Rate_ML = zeros(1, length(EsN0));
    for nSNR = 1:length(EsN0)
        PRobust_ML{nSNR} = Robust_Precoder_MLOptimal_with_mu ...
            (mu_list_beta{nSNR}, H_freq(:, 1:RxAntNum, 1:RBNum, srs_indx_list, :), Ut, Omega, beta, EsN0(nSNR), MeanSparisity, PowerConst);
        [Sum_Rate_ML(nSNR), ~, ~] = CalSumRate ...
            (PRobust_ML{nSNR}, H_freq(:, 1:RxAntNum, 1:RBNum, 1:SlotNum * BlockNum, :), EsN0(nSNR), BlockNum);
    end
    
        % -------- Robust -------- %
%     Sum_Rate_Robust = zeros(1, length(EsN0));
    for nSNR = 1:length(EsN0)
%         PRobust_MFOCUSS{nSNR} = Robust_Precoder_MFOCUSS ...
%             (PRZF{nSNR}, H_freq(:, 1:RxAntNum, 1:RBNum, srs_indx_list, :), Ut, Omega, beta, EsN0(nSNR), MeanSparisity, IterNum, PowerConst);
        mu_list_beta_interp{nSNR} = get_mu_list_chazhi(PRobust_MFOCUSS{nSNR}, H_freq(:, 1:RxAntNum, 1:RBNum, srs_indx_list, :), Ut, Omega, beta, EsN0(nSNR));
%         [Sum_Rate_Robust(nSNR), ~, ~] = CalSumRate ...
%             (PRobust_MFOCUSS{nSNR}, H_freq(:, 1:RxAntNum, 1:RBNum, 1:SlotNum * BlockNum, :), EsN0(nSNR), BlockNum);
    end

    % -------- Robust ML Optimal -------- %——利用差值好的mu，计算预编码矩阵
    Sum_Rate_ML_interp = zeros(1, length(EsN0));
    for nSNR = 1:length(EsN0)
        PRobust_ML_interp{nSNR} = Robust_Precoder_MLOptimal_with_mu ...
            (mu_list_beta_interp{nSNR}, H_freq(:, 1:RxAntNum, 1:RBNum, srs_indx_list, :), Ut, Omega, beta, EsN0(nSNR), MeanSparisity, PowerConst);
        [Sum_Rate_ML_interp(nSNR), ~, ~] = CalSumRate ...
            (PRobust_ML_interp{nSNR}, H_freq(:, 1:RxAntNum, 1:RBNum, 1:SlotNum * BlockNum, :), EsN0(nSNR), BlockNum);
    end
    
    
    figure;
    hold on
    plot(EsN0, real(Sum_Rate_RZF), '-*', EsN0, real(Sum_Rate_Robust), '-o', ...
        EsN0, real(Sum_Rate_ML), '-s', EsN0, real(Sum_Rate_ML_interp), '-d');
%     , EsN0, real(Sum_Rate_SLNR_CG), '-d'
    legend('RZF', 'Robust', 'eq-Robust','eq-Robust-interp', 'Location', 'NorthWest');
%     'SLNR-CG(l=5)'
    xlabel('SNR(dB)'), ylabel('Sum-Rate(bit/s/Hz)');
    title(num2str(BlockNum) + "ms");
    grid on
    grid on
end
% %     PRobust_ML即为所求p



% % 数据-->mu,rho
% % mu,rho-->p
% for indx_bn = 1:length(bn_list)
%     BlockNum = bn_list(indx_bn);
%     Block_indx_list = 1:BlockNum;
%     srs_indx_list = 1:BlockNum:SlotNum * BlockNum;
%     
%     % jake's model %——为了计算beta
%     alpha = abs(besselj(0, 2 * pi * Speed * 1e3 * 3.5e9 * (BlockNum + Block_indx_list) * 1e-3 / (3600 * 3e8))).^2;
%     beta = sqrt((mean(alpha))) * ones(UserNum, 1);
%     beta0 = zeros(UserNum, 1);
%     beta1 = ones(UserNum, 1);
%     
%    %% Channel Mean and Variance
%     % -------- MFOCUSS -------- %——为了计算Ut, Omega
%     [HFreqBeam, Ut, Omega] = CCMGenComp ...
%         (H_freq(:, 1:RxAntNum, :, 1:max_SlotNum * BlockNum, :), OSFactorH, OSFactorV, BlockNum);
%     HFreqBeam = HFreqBeam(:, :, 1:RBNum, 1:SlotNum , :);
%     
%     %% Precoding
%     % -------- RZF -------- %
% %     Sum_Rate_RZF = zeros(1, length(EsN0));
%     for nSNR = 1:length(EsN0)
%         PRZF{nSNR} = RZF_Precoder ...
%             (H_freq(:, 1:RxAntNum, 1:RBNum, srs_indx_list, :), EsN0(nSNR), PowerConst);
% %         [Sum_Rate_RZF(nSNR), ~, ~] = CalSumRate ...
% %             (PRZF{nSNR}, H_freq(:, 1:RxAntNum, 1:RBNum, 1:SlotNum * BlockNum, :), EsN0(nSNR), BlockNum);
%     end
%     
%     % -------- Robust -------- %
% %     Sum_Rate_Robust = zeros(1, length(EsN0));
%     for nSNR = 1:length(EsN0)
%         PRobust_MFOCUSS{nSNR} = Robust_Precoder_MFOCUSS ...
%             (PRZF{nSNR}, H_freq(:, 1:RxAntNum, 1:RBNum, srs_indx_list, :), Ut, Omega, beta, EsN0(nSNR), MeanSparisity, IterNum, PowerConst);
%         mu_list_beta{nSNR} = get_mu_list(PRobust_MFOCUSS{nSNR}, H_freq(:, 1:RxAntNum, 1:RBNum, srs_indx_list, :), Ut, Omega, beta, EsN0(nSNR));
% %         [Sum_Rate_Robust(nSNR), ~, ~] = CalSumRate ...
% %             (PRobust_MFOCUSS{nSNR}, H_freq(:, 1:RxAntNum, 1:RBNum, 1:SlotNum * BlockNum, :), EsN0(nSNR), BlockNum);
%     end
%     
%     % -------- Robust (beta = 1) --------%——为了得到mu_list_beta1和rho_list_beta1
% %     Sum_Rate_Robust_beta1 = zeros(1, length(EsN0));
%     rho_list_beta1 = zeros(length(EsN0), UserNum, RBNum, max_SlotNum - 1);
%     for nSNR = 1:length(EsN0)
%         PRobust_MFOCUSS1{nSNR} = Robust_Precoder_MFOCUSS ...
%             (PRZF{nSNR}, H_freq(:, 1:RxAntNum, 1:RBNum, srs_indx_list, :), Ut, Omega, beta1, EsN0(nSNR), MeanSparisity, IterNum, PowerConst);
%         mu_list_beta1{nSNR} = get_mu_list(PRobust_MFOCUSS1{nSNR}, H_freq(:, 1:RxAntNum, 1:RBNum, srs_indx_list, :), Ut, Omega, beta1, EsN0(nSNR));
%         for i = 1:RBNum
%             for j = 1:SlotNum - 1
%                 for k = 1:UserNum
%                     rho_list_beta1(nSNR,k,i,j) = mean(abs(squeeze(PRobust_MFOCUSS1{nSNR}(:,1,k,i,j))).^2);
%                 end
%             end
%         end
% %         PRobust_MFOCUSS1{nSNR} = Robust_Precoder_MLOptimal_with_mu ...
% %             (mu_list_beta1{nSNR}, H_freq(:, 1:RxAntNum, 1:RBNum, srs_indx_list, :), Ut, Omega, beta1, EsN0(nSNR), MeanSparisity, PowerConst);
% %         [Sum_Rate_Robust_beta1(nSNR), ~, ~] = CalSumRate ...
% %             (PRobust_MFOCUSS1{nSNR} , H_freq(:, 1:RxAntNum, 1:RBNum, 1:SlotNum * BlockNum, :), EsN0(nSNR), BlockNum);
%     end
%     
%     % -------- Robust (beta = 0) -------- %——为了得到mu_list_beta0和rho_list_beta0
% %     Sum_Rate_Robust_beta0 = zeros(1, length(EsN0));
%     rho_list_beta0 = zeros(length(EsN0), UserNum, RBNum, max_SlotNum - 1);
%     for nSNR = 1:length(EsN0)
%         PRobust_MFOCUSS0{nSNR} = Robust_Precoder_MFOCUSS ...
%             (PRZF{nSNR}, H_freq(:, 1:RxAntNum, 1:RBNum, srs_indx_list, :), Ut, Omega, beta0, EsN0(nSNR), MeanSparisity, IterNum, PowerConst);
%         mu_list_beta0{nSNR} = get_mu_list(PRobust_MFOCUSS0{nSNR}, H_freq(:, 1:RxAntNum, 1:RBNum, srs_indx_list, :), Ut, Omega, beta0, EsN0(nSNR));
%         for i = 1:RBNum
%             for j = 1:SlotNum - 1
%                 for k = 1:UserNum
%                     rho_list_beta0(nSNR,k,i,j) = mean(abs(squeeze(PRobust_MFOCUSS0{nSNR}(:,1,k,i,j))).^2);
%                 end
%             end
%         end
% %         PRobust_MFOCUSS0{nSNR} = Robust_Precoder_MLOptimal_with_mu ...
% %             (mu_list_beta0{nSNR}, H_freq(:, 1:RxAntNum, 1:RBNum, srs_indx_list, :), Ut, Omega, beta0, EsN0(nSNR), MeanSparisity, PowerConst);
% %         [Sum_Rate_Robust_beta0(nSNR), ~, ~] = CalSumRate ...
% %             (PRobust_MFOCUSS0{nSNR}, H_freq(:, 1:RxAntNum, 1:RBNum, 1:SlotNum * BlockNum, :), EsN0(nSNR), BlockNum);
%     end
%     
%     % -------- weighted mu Robust ML Optimal -------- %
% %     Sum_Rate_ML_wmu = zeros(1, length(EsN0));
%     for nSNR = 1:length(EsN0)
%         w_mu_list{nSNR} = zeros(UserNum, RBNum, SlotNum);
%         for k = 1:UserNum
%             w_mu_list{nSNR}(k, :, :) = beta(k)^2 * mu_list_beta1{nSNR}(k, :, :) + (1 - beta(k)^2) * mu_list_beta0{nSNR}(k, :, :);
%         end
%         for nSlot = 2 : SlotNum
%             for nRB = 1 : RBNum
%                 w_mu_list{nSNR}(:, nRB, nSlot) = cut_small_number(w_mu_list{nSNR}(:, nRB, nSlot), 0.1);
%             end
%         end
%         
%         PRobust_ML_weighted{nSNR} = Robust_Precoder_MLOptimal_with_mu ...
%             (w_mu_list{nSNR}, H_freq(:, 1:RxAntNum, 1:RBNum, srs_indx_list, :), Ut, Omega, beta, EsN0(nSNR), MeanSparisity, PowerConst);
% %         [Sum_Rate_ML_wmu(nSNR), ~, ~] = CalSumRate ...
% %             (PRobust_ML_weighted{nSNR}, H_freq(:, 1:RxAntNum, 1:RBNum, 1:SlotNum * BlockNum, :), EsN0(nSNR), BlockNum);
%     end
%     
%     % -------- weighted rho Robust ML Optimal -------- %
% %     Sum_Rate_ML_wrho = zeros(1, length(EsN0));
%     w_rho_list = zeros(length(EsN0), UserNum, RBNum, max_SlotNum - 1);
%     for nSNR = 1:length(EsN0)
%         
%         for i = 1:RBNum
%             for j = 1:SlotNum - 1
%                 for k = 1:UserNum
%                     w_rho_list(nSNR,k,i,j) = beta(k)^2 * rho_list_beta1(nSNR,k,i,j) + (1-beta(k)^2) * rho_list_beta0(nSNR,k,i,j);
%                 end
%             end
%         end
%     end 
% end
% 
%      % -------- 插值法（针对mu,rho） -------- %
%      % -------- 插值法（针对mu,rho） -------- %
%      
% for indx_bn = 1:length(bn_list)
% % %      ——利用差值好的mu和rho，计算预编码矩阵
%         PRobust_ML_wrho{nSNR} = Robust_Precoder_MLOptimal_with_mu ...
%             (mu_list_beta{nSNR}, H_freq(:, 1:RxAntNum, 1:RBNum, srs_indx_list, :), Ut, Omega, beta, EsN0(nSNR), MeanSparisity, PowerConst);
%         for i = 1:RBNum
%             for j = 1:SlotNum - 1
%                 for k = 1:UserNum
%                     pn = mean(abs(squeeze(PRobust_ML_wrho{nSNR}(:,1,k,i,j))).^2);
%                     if pn > 0
%                         PRobust_ML_wrho{nSNR}(:,1,k,i,j) = PRobust_ML_wrho{nSNR}(:,1,k,i,j) ./ sqrt(pn) .*sqrt(w_rho_list(nSNR,k,i,j));
%                     end
%                 end
%             end
%         end
%         
% %         [Sum_Rate_ML_wrho(nSNR), ~, ~] = CalSumRate ...
% %             (PRobust_ML_wrho{nSNR}, H_freq(:, 1:RxAntNum, 1:RBNum, 1:SlotNum * BlockNum, :), EsN0(nSNR), BlockNum);
%     end
% % %     PRobust_ML_wrho即为所求p






