%% 2x2 MIMO-OFDM, Rayleigh+AWGN, 16-QAM, Turbo (rate~1/3)
% BER/SER: before & after Turbo, plus "theory" (uncoded, ZF 2x2 Rayleigh)
clear; close all; clc;

rng(2026);

%% ---------------- System parameters ----------------
N      = 64;          % OFDM subcarriers
CP     = 16;          % cyclic prefix
N_sym  = 200;         % frames per SNR (tăng lên nếu muốn mượt hơn)
snr_db = 0:2:20;      % SNR sweep

nTx = 2; nRx = 2;
L   = 10;             % channel taps (<= CP)

% 16-QAM
M = 16;
bits_per_sym = log2(M);   % 4
sym_per_ofdm = N*nTx;     % 128 symbols / OFDM
bits_per_ofdm = sym_per_ofdm * bits_per_sym;

%% ---------------- Turbo parameters (rate ~ 1/3) ----------------
K_info = 160;                 % info bits
Ncoded = 3*K_info;            % systematic + parity1 + parity2
if mod(Ncoded,bits_per_sym)~=0
    error('Ncoded must be multiple of bits_per_sym');
end
Ns_used = Ncoded/bits_per_sym;  % number of QAM symbols that carry coded bits
if Ncoded > bits_per_ofdm
    error('Ncoded > bits_per_ofdm: giảm K_info hoặc tăng N');
end
bits_pad = bits_per_ofdm - Ncoded;

nIter = 6;                    % turbo iterations

% Interleaver
pi = randperm(K_info).';
invpi = zeros(K_info,1); invpi(pi) = (1:K_info).';

fprintf('OFDM: %d subcarriers, CP=%d, nTx=nRx=2, L=%d\n', N, CP, L);
fprintf('Turbo: K=%d, Ncoded=%d, Ns_used=%d/%d QAM symbols\n\n', ...
    K_info, Ncoded, Ns_used, sym_per_ofdm);

%% ---------------- Results ----------------
BER_pre  = zeros(numel(snr_db),1);   % before Turbo (systematic hard decision)
BER_post = zeros(numel(snr_db),1);   % after Turbo decoding

SER_pre  = zeros(numel(snr_db),1);   % before Turbo (symbol hard-decision)
SER_post = zeros(numel(snr_db),1);   % after Turbo (decode->re-encode->remap compare)

% "Theory" (uncoded) for ZF 2x2 Rayleigh via averaging over H
BER_th = zeros(numel(snr_db),1);
SER_th = zeros(numel(snr_db),1);

%% ---------------- Precompute "theory" channel samples ----------------
Ntheory = 200000;  % tăng nếu muốn đường lý thuyết mượt hơn
diagInv = zeros(Ntheory,nTx); % diag((H^H H)^-1) for each stream
for m = 1:Ntheory
    H = (randn(nRx,nTx)+1j*randn(nRx,nTx))/sqrt(2);
    G = (H'*H)\eye(nTx);
    diagInv(m,:) = real(diag(G)).';
end

%% ---------------- Main loop over SNR ----------------
for iSNR = 1:numel(snr_db)
    SNRdB  = snr_db(iSNR);
    SNRlin = 10^(SNRdB/10);

    bit_err_pre  = 0; bit_tot  = 0;
    bit_err_post = 0;

    sym_err_pre  = 0; sym_tot  = 0;
    sym_err_post = 0;

    for frame = 1:N_sym
        %% ================= Transmitter =================
        info_bits = randi([0 1], K_info, 1);

        coded_bits = turbo_encode_rsc(info_bits, pi);  % length 3K

        % Pad to fill one OFDM frame (keeps average power consistent)
        bits_frame = [coded_bits; zeros(bits_pad,1)];  % length bits_per_ofdm

        % 16-QAM map
        tx_syms_all = qam16_mod(bits_frame);           % length sym_per_ofdm

        % Arrange into subcarriers x Tx antennas
        tx_freq = reshape(tx_syms_all, N, nTx);

        % Normalize total transmit power (important & standard)
        tx_freq = tx_freq / sqrt(nTx);

        % OFDM IFFT + CP
        tx_time    = ifft(tx_freq, N);
        tx_time_cp = [tx_time(end-CP+1:end,:); tx_time];   % (N+CP) x nTx

        %% ================= Channel: Rayleigh (L taps) + AWGN =================
        H_time = (randn(L,nRx,nTx) + 1j*randn(L,nRx,nTx))/sqrt(2*L);
        H_freq = fft(H_time, N, 1);  % N x nRx x nTx

        rx_time = zeros(N+CP, nRx);
        for rx = 1:nRx
            for tx = 1:nTx
                h = squeeze(H_time(:,rx,tx));
                tmp = conv(tx_time_cp(:,tx), h);
                rx_time(:,rx) = rx_time(:,rx) + tmp(1:N+CP);
            end
        end

        % AWGN added in time domain
        signal_power = mean(abs(tx_time_cp(:)).^2);              % average per sample (all Tx)
        noise_power_time = signal_power / SNRlin;                % per complex time sample
        noise = sqrt(noise_power_time/2) * (randn(size(rx_time)) + 1j*randn(size(rx_time)));
        rx_time = rx_time + noise;

        %% ================= Receiver =================
        % Remove CP + FFT
        rx_no_cp = rx_time(CP+1:end,:);
        rx_freq  = fft(rx_no_cp, N);   % N x nRx

        % ZF per subcarrier + compute noise enhancement
        rx_syms = zeros(N,nTx);
        noiseEnh = zeros(N,nTx);  % diag((H^H H)^-1) per subcarrier

        for k = 1:N
            Hk = squeeze(H_freq(k,:,:));        % nRx x nTx
            HH = (Hk'*Hk);
            G  = HH\eye(nTx);                   % inv(H^H H)
            Wk = G*Hk';                         % ZF: (H^H H)^-1 H^H

            rx_syms(k,:)  = (Wk * rx_freq(k,:).').';
            noiseEnh(k,:) = real(diag(G)).';
        end

        rx_syms_vec = rx_syms(:);   % length sym_per_ofdm

        % Only evaluate over first Ns_used symbols that actually carry coded bits
        tx_syms_used = tx_syms_all(1:Ns_used);
        rx_syms_used = rx_syms_vec(1:Ns_used);

        % SER before decoding
        rx_syms_hd = qam16_hard(rx_syms_used);
        sym_err_pre = sym_err_pre + sum(rx_syms_hd ~= tx_syms_used);
        sym_tot     = sym_tot + Ns_used;

        % Compute per-symbol noise variance at ZF output
        % Noise variance per Rx subcarrier after FFT: sigma2_rx = N * noise_power_time
        sigma2_rx = N * noise_power_time;

        % Symbol order in rx_syms_vec is: stream1(all N) then stream2(all N)
        noiseVar_sym = [sigma2_rx*noiseEnh(:,1); sigma2_rx*noiseEnh(:,2)];
        noiseVar_used = noiseVar_sym(1:Ns_used);

        % Soft demod (LLR) for used symbols -> coded bit LLRs
        [~, llr_vec] = qam16_demod_llr(rx_syms_used, noiseVar_used); % length 4*Ns_used = Ncoded

        % Split LLRs into sys, p1, p2 according to turbo encoder order
        L_sys = llr_vec(1:K_info);
        L_p1  = llr_vec(K_info+1:2*K_info);
        L_p2  = llr_vec(2*K_info+1:3*K_info);

        % BER before turbo: hard decision on systematic bits
        hard_sys = (L_sys < 0);
        bit_err_pre = bit_err_pre + sum(hard_sys ~= info_bits);

        % Turbo decode (max-log-MAP)
        dec_bits = turbo_decode_maxlog(L_sys, L_p1, L_p2, pi, invpi, nIter);

        % BER after turbo
        bit_err_post = bit_err_post + sum(dec_bits ~= info_bits);
        bit_tot      = bit_tot + K_info;

        % SER after turbo: decode -> re-encode -> remap -> compare to transmitted symbols
        coded_hat = turbo_encode_rsc(dec_bits, pi);
        bits_hat_frame = [coded_hat; zeros(bits_pad,1)];
        syms_hat_all   = qam16_mod(bits_hat_frame);
        syms_hat_used  = syms_hat_all(1:Ns_used);

        sym_err_post = sym_err_post + sum(syms_hat_used ~= tx_syms_used);
    end

    BER_pre(iSNR)  = bit_err_pre  / bit_tot;
    BER_post(iSNR) = bit_err_post / bit_tot;

    SER_pre(iSNR)  = sym_err_pre  / sym_tot;
    SER_post(iSNR) = sym_err_post / sym_tot;

    %% -------- "Theory" (uncoded) for ZF 2x2 Rayleigh ----------
    % Effective SNR per stream after ZF: gamma = SNRlin / diag((H^H H)^-1)
    gamma = SNRlin ./ diagInv; % Ntheory x 2

    % Conditional (AWGN) approximations for square 16-QAM (Gray)
    Q = qfunc_local(sqrt(gamma/5));
    ber_cond = (3/4) * Q;
    ser_cond = 3*Q - (9/4)*(Q.^2);

    BER_th(iSNR) = mean(ber_cond,'all');
    SER_th(iSNR) = mean(ser_cond,'all');

    fprintf('SNR=%2d dB | BER_pre=%.3e BER_post=%.3e | SER_pre=%.3e SER_post=%.3e\n', ...
        SNRdB, BER_pre(iSNR), BER_post(iSNR), SER_pre(iSNR), SER_post(iSNR));
end

%% ---------------- Plots ----------------
figure;
semilogy(snr_db, BER_pre,  'o-', 'LineWidth', 2); hold on;
semilogy(snr_db, BER_post, 's-', 'LineWidth', 2);
semilogy(snr_db, BER_th,   'k--', 'LineWidth', 2);
grid on; xlabel('SNR (dB)'); ylabel('BER');
legend('BER trước giải mã (systematic hard)', ...
       'BER sau giải mã Turbo (Max-Log-MAP)', ...
       'BER lý thuyết (uncoded, ZF 2x2 Rayleigh)', ...
       'Location','southwest');
title('BER: 2x2 MIMO-OFDM, 16-QAM, Rayleigh+AWGN');

figure;
semilogy(snr_db, SER_pre,  'o-', 'LineWidth', 2); hold on;
semilogy(snr_db, SER_post, 's-', 'LineWidth', 2);
semilogy(snr_db, SER_th,   'k--', 'LineWidth', 2);
grid on; xlabel('SNR (dB)'); ylabel('SER');
legend('SER trước giải mã (hard symbol)', ...
       'SER sau giải mã (decode->re-encode->remap)', ...
       'SER lý thuyết (uncoded, ZF 2x2 Rayleigh)', ...
       'Location','southwest');
title('SER: 2x2 MIMO-OFDM, 16-QAM, Rayleigh+AWGN');

%% ========================= Local functions =========================

function coded = turbo_encode_rsc(u, pi)
% Turbo rate~1/3: [systematic; parity1; parity2]
    u = u(:);
    p1 = rsc_parity_13_15(u);
    u2 = u(pi);
    p2 = rsc_parity_13_15(u2);
    coded = [u; p1; p2];
end

function uhat = turbo_decode_maxlog(Lsys, Lp1, Lp2, pi, invpi, nIter)
% Max-Log-MAP turbo decoder for RSC(13,15), no termination (simplified)
    K = length(Lsys);
    Lsys = Lsys(:); Lp1 = Lp1(:); Lp2 = Lp2(:);

    Lsys_i = Lsys(pi);

    La1 = zeros(K,1);
    Le1 = zeros(K,1);

    for it = 1:nIter
        % Decoder 1
        [~, Le1] = rsc_bcjr_maxlog(Lsys, Lp1, La1);

        % Interleave extrinsic -> a priori for decoder 2
        La2 = Le1(pi);

        % Decoder 2 (on interleaved sequence)
        [~, Le2_i] = rsc_bcjr_maxlog(Lsys_i, Lp2, La2);

        % Deinterleave extrinsic back
        Le2 = Le2_i(invpi);

        % Feedback as new a priori
        La1 = Le2;
    end

    % Final LLR (common practice)
    L_final = Lsys + La1 + Le1;
    uhat = (L_final < 0);
end

function p = rsc_parity_13_15(u)
% RSC(13,15) octal, memory=3 (8 states), no tail bits
% feedback poly 13(oct)=1011 => taps r1,r3
% feedforward 15(oct)=1101 => taps r2,r3
    u = u(:);
    K = length(u);
    p = zeros(K,1);

    r1=0; r2=0; r3=0; % shift register
    for k = 1:K
        d = xor(u(k), xor(r1, r3));        % feedback
        p(k) = xor(d, xor(r2, r3));        % parity
        % shift
        r3 = r2; r2 = r1; r1 = d;
    end
end

function [Lpost, Lext] = rsc_bcjr_maxlog(Lsys, Lpar, Lapri)
% Max-Log-MAP BCJR for RSC(13,15), 8-state, no termination
    Lsys = Lsys(:); Lpar = Lpar(:); Lapri = Lapri(:);
    T = length(Lsys);
    nStates = 8;

    [nextState, parityBit] = rsc_trellis_13_15();

    % alpha/beta in log domain
    alpha = -inf(T+1, nStates);
    beta  = zeros(T+1, nStates);  % no termination: beta(T+1,:) = 0
    alpha(1,1) = 0;               % start from state 0

    % branch metrics gamma(t,s,u)
    gamma = -inf(T, nStates, 2);
    for t = 1:T
        for s = 1:nStates
            for u = 0:1
                p = parityBit(s,u+1);
                xu = 1 - 2*u;     % 0->+1, 1->-1
                xp = 1 - 2*p;
                gamma(t,s,u+1) = 0.5*(xu*(Lsys(t)+Lapri(t)) + xp*Lpar(t));
            end
        end
    end

    % forward
    for t = 1:T
        for s = 1:nStates
            a = alpha(t,s);
            if isinf(a), continue; end
            for u = 0:1
                sp = nextState(s,u+1) + 1;
                alpha(t+1,sp) = max(alpha(t+1,sp), a + gamma(t,s,u+1));
            end
        end
    end

    % backward
    for t = T:-1:1
        for s = 1:nStates
            sp0 = nextState(s,1)+1;
            sp1 = nextState(s,2)+1;
            b0 = beta(t+1,sp0) + gamma(t,s,1);
            b1 = beta(t+1,sp1) + gamma(t,s,2);
            beta(t,s) = max(b0,b1);
        end
    end

    % LLR computation
    Lpost = zeros(T,1);
    for t = 1:T
        m0 = -inf; m1 = -inf;
        for s = 1:nStates
            for u = 0:1
                sp = nextState(s,u+1)+1;
                metric = alpha(t,s) + gamma(t,s,u+1) + beta(t+1,sp);
                if u==0
                    m0 = max(m0, metric);
                else
                    m1 = max(m1, metric);
                end
            end
        end
        Lpost(t) = m0 - m1; % log P(u=0)/P(u=1)
    end

    Lext = Lpost - Lsys - Lapri;
end

function [nextState, parityBit] = rsc_trellis_13_15()
% state s is 0..7 with bits [r1 r2 r3] (r1=1-delay, r3=3-delay)
    nStates = 8;
    nextState = zeros(nStates,2);
    parityBit = zeros(nStates,2);

    for s = 0:nStates-1
        r1 = bitget(s,3);
        r2 = bitget(s,2);
        r3 = bitget(s,1);

        for u = 0:1
            d = xor(u, xor(r1, r3));      % feedback 13
            p = xor(d, xor(r2, r3));      % feedforward 15

            nr1 = d; nr2 = r1; nr3 = r2;  % shift
            sp = nr1*4 + nr2*2 + nr3;

            nextState(s+1,u+1) = sp;
            parityBit(s+1,u+1) = p;
        end
    end
end

%% ---------------- 16-QAM MOD/DEMOD (no toolbox) ----------------
function syms = qam16_mod(bits)
% bits: column vector length multiple of 4
    bits = bits(:);
    if mod(length(bits),4)~=0
        error('qam16_mod: bits length must be multiple of 4');
    end
    b = reshape(bits,4,[]).'; % Ns x 4
    b3=b(:,1); b2=b(:,2); b1=b(:,3); b0=b(:,4);

    Ns = size(b,1);
    I = zeros(Ns,1); Q = zeros(Ns,1);

    I(~b3 & ~b2) =  3;
    I(~b3 &  b2) =  1;
    I( b3 &  b2) = -1;
    I( b3 & ~b2) = -3;

    Q(~b1 & ~b0) =  3;
    Q(~b1 &  b0) =  1;
    Q( b1 &  b0) = -1;
    Q( b1 & ~b0) = -3;

    syms = (I + 1j*Q)/sqrt(10); % unit avg power
end

function [bits, llr_vec] = qam16_demod_llr(syms, noiseVarSym)
% Max-log LLR per bit for Gray 16-QAM
% noiseVarSym: scalar or vector (Ns x 1) variance at symbol (complex)
    persistent const_points bit_labels;
    if isempty(const_points)
        bit_labels = my_de2bi(0:15,4);        % 16 x 4
        all_bits = bit_labels.'; all_bits = all_bits(:);
        const_points = qam16_mod(all_bits);   % 16 x 1
    end

    syms = syms(:);
    Ns = length(syms);

    if isscalar(noiseVarSym)
        noiseVarSym = noiseVarSym*ones(Ns,1);
    else
        noiseVarSym = noiseVarSym(:);
        if length(noiseVarSym)~=Ns
            error('qam16_demod_llr: noiseVarSym must be scalar or Ns-length vector');
        end
    end

    % distances
    d2 = zeros(Ns,16);
    for m = 1:16
        d2(:,m) = abs(syms - const_points(m)).^2;
    end

    llr = zeros(Ns,4);
    bits_hard = zeros(Ns,4);

    for j = 1:4
        mask0 = (bit_labels(:,j)==0);
        mask1 = ~mask0;

        d0 = min(d2(:,mask0),[],2);
        d1 = min(d2(:,mask1),[],2);

        llr(:,j) = (d1 - d0) ./ noiseVarSym;  % log(P0/P1) up to const
        bits_hard(:,j) = (llr(:,j) < 0);
    end

    bits    = reshape(bits_hard.',[],1);
    llr_vec = reshape(llr.',[],1);
end

function syms_hd = qam16_hard(syms)
    persistent const_points;
    if isempty(const_points)
        bit_labels = my_de2bi(0:15,4);
        all_bits = bit_labels.'; all_bits = all_bits(:);
        const_points = qam16_mod(all_bits);
    end
    syms = syms(:);
    Ns = length(syms);
    d2 = zeros(Ns,16);
    for m = 1:16
        d2(:,m) = abs(syms - const_points(m)).^2;
    end
    [~,idx] = min(d2,[],2);
    syms_hd = const_points(idx);
end

function B = my_de2bi(vec, n)
% left-msb
    vec = vec(:);
    N = length(vec);
    B = zeros(N,n);
    for i = 1:n
        B(:,i) = mod(floor(vec / 2^(n-i)), 2);
    end
end

function y = qfunc_local(x)
% Q(x)=0.5*erfc(x/sqrt(2))
    y = 0.5*erfc(x/sqrt(2));
end
