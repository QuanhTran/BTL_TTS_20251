%% 2x2 MIMO-OFDM, 16-QAM, "Turbo" coding (repetition-3), BER/SER
clear; close all; clc;

%% System parameters
N      = 64;          % Number of OFDM subcarriers
CP     = 16;          % Cyclic prefix length
N_sym  = 100;         % Number of OFDM frames per SNR point
snr_db = 0:2:20;      % SNR range (dB)

nTx = 2;              % Transmit antennas
nRx = 2;              % Receive antennas
L   = 10;             % Channel tap length (<= CP)

% 16-QAM
M            = 16;
bits_per_sym = log2(M);      % = 4
sym_per_ofdm = N * nTx;      % 128 QAM symbols per OFDM
bits_per_ofdm = sym_per_ofdm * bits_per_sym;  % 512 bits

% "Turbo" code: dùng repetition-3 cho đơn giản (rate ~ 1/3)
K_info  = 160;                % số bit thông tin / frame
R       = 1/3;                % coding rate xấp xỉ
Ncoded  = 3 * K_info;         % 3 lần lặp
if mod(Ncoded, bits_per_sym) ~= 0
    error('Ncoded must be multiple of bits_per_sym');
end
if Ncoded > bits_per_ofdm
    error('Ncoded > bits_per_ofdm, tăng K_info hoặc giảm N');
end
bits_pad = bits_per_ofdm - Ncoded;   % bit "trống" không dùng
Ns_used  = Ncoded / bits_per_sym;    % số symbol QAM thực sự mang dữ liệu

fprintf('Each OFDM frame: %d info bits -> %d coded bits (rate ~ %.2f)\n', ...
    K_info, Ncoded, K_info/Ncoded);
fprintf('We use %d/%d QAM symbols in each OFDM (rest zero padded).\n\n', ...
    Ns_used, sym_per_ofdm);

%% Results
BER = zeros(length(snr_db),1);
SER = zeros(length(snr_db),1);

%% Main loop over SNR
for iSNR = 1:length(snr_db)
    SNRdB = snr_db(iSNR);

    bit_errors = 0; total_bits = 0;
    sym_errors = 0; total_syms  = 0;

    for frame = 1:N_sym
        %% ============== Transmitter ==============

        % 1) Sinh bit thông tin
        info_bits = randi([0 1], K_info, 1);

        % 2) "Turbo" encode (repetition-3 ở đây)
        coded_bits = turbo_encode_simple(info_bits);   % length = Ncoded

        % 3) Nhét vào 1 OFDM frame, pad thêm bit 0 nếu cần
        bits_frame = [coded_bits; zeros(bits_pad,1)];  % length = bits_per_ofdm

        % 4) Điều chế 16-QAM
        tx_syms_all = qam16_mod(bits_frame);           % length = sym_per_ofdm (=128)

        % 5) Xếp thành N subcarrier x nTx anten
        tx_freq = reshape(tx_syms_all, N, nTx);

        % 6) OFDM IFFT + CP
        tx_time = ifft(tx_freq, N);
        tx_time_cp = [tx_time(end-CP+1:end,:); tx_time];   % (N+CP) x nTx

        %% ============== Channel (Rayleigh + AWGN) ==============
        % Channel impulse response (time domain) L taps
        H_time = (randn(L,nRx,nTx) + 1j*randn(L,nRx,nTx))/sqrt(2*L);

        % Frequency response trên từng subcarrier
        H_freq = fft(H_time, N, 1);   % size: N x nRx x nTx

        % Truyền qua kênh trong miền thời gian
        rx_time = zeros(N+CP, nRx);
        for rx = 1:nRx
            for tx = 1:nTx
                h_rt = squeeze(H_time(:,rx,tx));      % L x 1
                conv_result = conv(tx_time_cp(:,tx), h_rt);
                rx_time(:,rx) = rx_time(:,rx) + conv_result(1:N+CP);
            end
        end

        % Thêm AWGN
        signal_power = mean(abs(tx_time_cp(:)).^2);   % Es (per complex sample)
        noise_power  = signal_power / (10^(SNRdB/10));   % N0 (per complex)
        noise = sqrt(noise_power/2) * ...
                (randn(size(rx_time)) + 1j*randn(size(rx_time)));
        rx_time = rx_time + noise;

        %% ============== Receiver ==============

        % 1) Remove CP + FFT
        rx_time_no_cp = rx_time(CP+1:end,:);
        rx_freq = fft(rx_time_no_cp, N);   % N x nRx

        % 2) ZF detection từng subcarrier
        rx_syms = zeros(N, nTx);
        for k = 1:N
            Hk = squeeze(H_freq(k,:,:));        % nRx x nTx
            rx_syms(k,:) = (pinv(Hk) * rx_freq(k,:).').';
        end

        rx_syms_vec = rx_syms(:);   % length = sym_per_ofdm

        % 3) Chỉ lấy Ns_used symbol mang dữ liệu
        tx_syms_used = tx_syms_all(1:Ns_used);
        rx_syms_used = rx_syms_vec(1:Ns_used);

        % 4) SER: so sánh symbol 16-QAM trước khi giải mã
        rx_syms_hd = qam16_hard(rx_syms_used);
        sym_errors = sym_errors + sum(rx_syms_hd ~= tx_syms_used);
        total_syms = total_syms + Ns_used;

        % 5) Giải điều chế 16-QAM -> bit + LLR cho từng bit
        [rx_bits_used, llr_used] = qam16_demod_llr(rx_syms_used, noise_power);
        % rx_bits_used, llr_used: Ncoded bit đầu tiên

        % 6) Turbo decode (trên Ncoded bit tương ứng K_info info bits)
        decoded_bits = turbo_decode_simple(llr_used, K_info);

        % 7) BER trên bit thông tin
        bit_errors = bit_errors + sum(decoded_bits ~= info_bits);
        total_bits = total_bits + K_info;
    end

    BER(iSNR) = bit_errors / total_bits;
    SER(iSNR) = sym_errors / total_syms;

    fprintf('SNR = %2d dB: BER = %.4e, SER = %.4e\n', ...
        SNRdB, BER(iSNR), SER(iSNR));
end

%% Plot BER/SER
figure;
semilogy(snr_db, BER, 'b-o', 'LineWidth', 2, 'MarkerSize', 7); hold on;
semilogy(snr_db, SER, 'r-s', 'LineWidth', 2, 'MarkerSize', 7);
grid on;
xlabel('SNR (dB)');
ylabel('Error rate');
legend('BER sau giải mã "Turbo"', 'SER trước giải mã', 'Location', 'southwest');
title('2x2 MIMO-OFDM, 16-QAM, Turbo-like (repetition-3) over Rayleigh+AWGN');

%% =================== helper: de2bi tự viết (không toolbox) ===================
function B = my_de2bi(vec, n)
% vec: vector hàng hoặc cột các số nguyên không âm
% n: số bit, output B: [length(vec) x n], MSB bên trái (left-msb)
    vec = vec(:);
    N   = length(vec);
    B   = zeros(N, n);
    for i = 1:n
        B(:,i) = mod(floor(vec / 2^(n-i)), 2);
    end
end

%% =================== 16-QAM MOD/DEMOD (không dùng toolbox) ===================

function syms = qam16_mod(bits)
% bits: vector cột 0/1, length phải chia hết cho 4
    if mod(length(bits),4) ~= 0
        error('qam16_mod: length(bits) must be multiple of 4');
    end
    b = reshape(bits, 4, []).';   % Ns x 4
    b3 = b(:,1); b2 = b(:,2); b1 = b(:,3); b0 = b(:,4);

    Ns = size(b,1);
    I = zeros(Ns,1);
    Q = zeros(Ns,1);

    % Gray mapping trên trục I
    % b3b2: 00->+3, 01->+1, 11->-1, 10->-3
    I(~b3 & ~b2) =  3;
    I(~b3 &  b2) =  1;
    I( b3 &  b2) = -1;
    I( b3 & ~b2) = -3;

    % Gray mapping trên trục Q
    % b1b0: 00->+3, 01->+1, 11->-1, 10->-3
    Q(~b1 & ~b0) =  3;
    Q(~b1 &  b0) =  1;
    Q( b1 &  b0) = -1;
    Q( b1 & ~b0) = -3;

    % Chuẩn hoá Es_avg = 1 (vì E[I^2+Q^2]=10)
    syms = (I + 1j*Q) / sqrt(10);
end

function [bits, llr_vec] = qam16_demod_llr(syms, noise_power)
% Giải điều chế 16-QAM:
% - bits: quyết định cứng 0/1
% - llr_vec: LLR cho từng bit (log P(b=0)/P(b=1)) theo max-log

    Ns = length(syms);

    % Tạo bảng chòm sao + nhãn bit (consistent với qam16_mod)
    bit_labels = my_de2bi(0:15, 4);   % 16 x 4, left-msb
    all_bits   = bit_labels.';
    all_bits   = all_bits(:);         % 64 x 1
    const_points = qam16_mod(all_bits);  % 16 x 1

    % Tính khoảng cách đến từng điểm chòm sao
    numConst = 16;
    d2 = zeros(Ns, numConst);
    for m = 1:numConst
        d2(:,m) = abs(syms - const_points(m)).^2;
    end

    llr = zeros(Ns,4);
    bits_hard = zeros(Ns,4);

    for j = 1:4
        mask0 = (bit_labels(:,j) == 0);
        mask1 = ~mask0;

        d0 = min(d2(:,mask0), [], 2);  % distance nhóm bit=0
        d1 = min(d2(:,mask1), [], 2);  % distance nhóm bit=1

        % Max-log LLR: log(P0/P1) ~ (d1 - d0)/N0
        llr(:,j) = (d1 - d0) / noise_power;

        bits_hard(:,j) = (llr(:,j) < 0);   % LLR<0 -> bit=1
    end

    bits     = reshape(bits_hard.', [], 1);
    llr_vec  = reshape(llr.', [], 1);
end

function syms_hd = qam16_hard(syms)
% Quyết định cứng symbol 16-QAM (chọn điểm gần nhất)
    bit_labels = my_de2bi(0:15, 4);   % 16 x 4
    all_bits   = bit_labels.';
    all_bits   = all_bits(:);
    const_points = qam16_mod(all_bits);  % 16 x 1

    Ns = length(syms);
    numConst = 16;
    d2 = zeros(Ns, numConst);
    for m = 1:numConst
        d2(:,m) = abs(syms - const_points(m)).^2;
    end
    [~, idx_min] = min(d2, [], 2);
    syms_hd = const_points(idx_min);
end

%% =================== "Turbo" encoder / decoder (repetition-3) ===================

function coded_bits = turbo_encode_simple(info_bits)
% Mã "Turbo" cực đơn giản: lặp mỗi bit 3 lần (rate 1/3)
% coded_bits: [u1 u1 u1 u2 u2 u2 ...]'
    coded_bits = repmat(info_bits, 3, 1);
end

function decoded_bits = turbo_decode_simple(llr_coded, K_info)
% Giải mã repetition-3 dùng LLR mềm
% llr_coded: vector LLR chiều dài 3*K_info
% decoded_bits: K_info bit 0/1

    if length(llr_coded) ~= 3*K_info
        error('turbo_decode_simple: length(llr_coded) must be 3*K_info');
    end
    llr_mat = reshape(llr_coded, K_info, 3);
    llr_sum = sum(llr_mat, 2);       % cộng LLR 3 bản sao

    % LLR > 0 -> bit=0, LLR < 0 -> bit=1
    decoded_bits = (llr_sum < 0);
end
