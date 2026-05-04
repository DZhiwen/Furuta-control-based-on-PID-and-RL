% train_sim2real_ppo.m
% =========================================================
%  Sim-to-Real 微调训练脚本（PPO版）
%
%  设计目标：
%    1. 从基础 PPO agent 继续微调
%    2. 更适合 PPO 的 on-policy 更新方式
%    3. 使用课程式随机化（curriculum）
%    4. th_ref0 在后续阶段逐步变为变量
%    5. 初始状态逐步从小范围扩展到大范围
%    6. 参数随机化逐步增强，最终适应完整 s2r 偏差
%
%  阶段设计：
%    Stage 0: 轻度 s2r，th_ref0固定为0，小范围初值
%    Stage 1: 轻度 s2r，小范围 th_ref0 随机化
%    Stage 2: 中度 s2r，中范围 th_ref0，混合中/大范围初值
%    Stage 3: 完整 s2r，大范围 th_ref0，混合大范围初值
%
%  训练集数：
%    Stage 0: 600
%    Stage 1: 1000
%    Stage 2: 1500
%    Stage 3: 2500
%
%  说明：
%    - 本脚本假设 furuta_ppo_agent_small.mat 中变量名为 agent
%    - 若 mat 文件中变量名不同，请修改 load 后的字段名
%    - 所有阶段均按 EpisodeCount 跑满，不提前停止
% =========================================================
clear;
clc;
close all;

%% =========================
%  基本设置
% ==========================
mdl      = "furuta_rl_balance_s2r";
agentBlk = mdl + "/Subsystem2/RL Agent";
Ts       = 0.01;
Tf       = 5;
maxSteps = ceil(Tf / Ts);

% 标称参数写入 base workspace
p = getNominalPlantParams();
assignin('base', "p", p);

% Bus 信息
busInfo   = Simulink.Bus.createObject(p);
PlantPBus = evalin('base', busInfo.busName);
assignin('base', "PlantPBus", PlantPBus);

open_system(mdl);

%% =========================
%  Observation / Action
% ==========================
obsInfo = rlNumericSpec([5 1], ...
    LowerLimit=-inf*ones(5,1), ...
    UpperLimit= inf*ones(5,1));
obsInfo.Name = "observations";
obsInfo.Description = "e_th, th_d, cos(al), sin(al), al_d";

actInfo = rlNumericSpec([1 1], ...
    LowerLimit=-1, ...
    UpperLimit=1);
actInfo.Name = "duty";

env = rlSimulinkEnv(mdl, agentBlk, obsInfo, actInfo);

%% =========================
%  加载基础 PPO Agent
% ==========================
S = load("furuta_ppo_agent_small_64_32.mat");
agent = S.agent;   % 如果你的变量名不是 agent，请修改这里

fprintf("✅ 基础 PPO Agent 加载完成\n");

%% =========================
%  PPO 微调参数重设
% ==========================
% 说明：
% - 微调时比基础训练更保守一点，防止破坏已有策略
% - 但不需要像 SAC 那样极端保守
% - 这些字段基于你给出的 PPO 训练代码接口风格
agent.AgentOptions.ExperienceHorizon       = 1000;
agent.AgentOptions.ClipFactor              = 0.15;
agent.AgentOptions.EntropyLossWeight       = 0.001;
agent.AgentOptions.MiniBatchSize           = 128;
agent.AgentOptions.NumEpoch                = 6;
agent.AgentOptions.AdvantageEstimateMethod = "gae";
agent.AgentOptions.GAEFactor               = 0.97;
agent.AgentOptions.DiscountFactor          = 0.995;
agent.AgentOptions.SampleTime              = Ts;

agent.AgentOptions.ActorOptimizerOptions.LearnRate         = 5e-5;
agent.AgentOptions.ActorOptimizerOptions.GradientThreshold = 1.0;

agent.AgentOptions.CriticOptimizerOptions.LearnRate         = 1e-4;
agent.AgentOptions.CriticOptimizerOptions.GradientThreshold = 1.0;

fprintf("✅ PPO 微调参数已设置\n");
fprintf("   ExperienceHorizon = %d\n", agent.AgentOptions.ExperienceHorizon);
fprintf("   ClipFactor        = %.3f\n", agent.AgentOptions.ClipFactor);
fprintf("   EntropyWeight     = %.4f\n", agent.AgentOptions.EntropyLossWeight);
fprintf("   NumEpoch          = %d\n", agent.AgentOptions.NumEpoch);
fprintf("   Actor LR          = %.1e\n", agent.AgentOptions.ActorOptimizerOptions.LearnRate);
fprintf("   Critic LR         = %.1e\n\n", agent.AgentOptions.CriticOptimizerOptions.LearnRate);

%% =========================================================
%  Stage 0：轻度 s2r 适配
%  - th_ref0 = 0
%  - 小范围初值
%  - 很轻的摩擦/死区/延迟随机化
%  目的：先适应 sim2real 偏差，不引入 tracking 难度
% ==========================================================
fprintf("=== Stage 0：轻度 s2r 适配（600集，跑满）===\n");

env.ResetFcn = @(in) resetS2R_PPO(in, 0);

trainS0 = rlTrainingOptions(...
    MaxEpisodes                 = 600, ...
    MaxStepsPerEpisode          = maxSteps, ...
    ScoreAveragingWindowLength  = 30, ...
    Verbose                     = true, ...
    Plots                       = "training-progress", ...
    StopOnError                 = "on", ...
    StopTrainingCriteria        = "EpisodeCount", ...
    StopTrainingValue           = 600, ...
    SaveAgentCriteria           = "AverageReward", ...
    SaveAgentValue              = 650, ...
    SaveAgentDirectory          = "savedAgents_ppo_s2r_s0");

statsS0 = train(agent, env, trainS0);
save("agent_ppo_s2r_stage0.mat", "agent", "statsS0");

avgS0 = mean(statsS0.EpisodeReward(max(1,end-29):end));
fprintf("Stage 0 完成，最近30集平均 reward: %.1f\n", avgS0);

if avgS0 < 500
    warning("⚠️ Stage 0 最近30集平均 reward 低于 500，建议后续单独补训 Stage 0 或减小随机化幅度！");
elseif avgS0 < 650
    fprintf("⚠️ Stage 0 已基本稳定，但收敛一般。\n\n");
else
    fprintf("✅ Stage 0 表现良好，进入 Stage 1\n\n");
end

%% =========================================================
%  Stage 1：轻度 s2r + 小范围目标随机化
%  - th_ref0 ∈ [-0.15, 0.15]
%  - 初值仍为小范围
%  - 随机化略增强
% ==========================================================
fprintf("=== Stage 1：轻度 s2r + 小范围目标随机化（1000集，跑满）===\n");

env.ResetFcn = @(in) resetS2R_PPO(in, 1);

trainS1 = rlTrainingOptions(...
    MaxEpisodes                 = 1000, ...
    MaxStepsPerEpisode          = maxSteps, ...
    ScoreAveragingWindowLength  = 40, ...
    Verbose                     = true, ...
    Plots                       = "training-progress", ...
    StopOnError                 = "on", ...
    StopTrainingCriteria        = "EpisodeCount", ...
    StopTrainingValue           = 1000, ...
    SaveAgentCriteria           = "AverageReward", ...
    SaveAgentValue              = 600, ...
    SaveAgentDirectory          = "savedAgents_ppo_s2r_s1");

statsS1 = train(agent, env, trainS1);
save("agent_ppo_s2r_stage1.mat", "agent", "statsS1");

avgS1 = mean(statsS1.EpisodeReward(max(1,end-39):end));
fprintf("Stage 1 完成，最近40集平均 reward: %.1f\n", avgS1);

if avgS1 < 450
    warning("⚠️ Stage 1 最近40集平均 reward 偏低，建议后续单独补训 Stage 1！");
elseif avgS1 < 600
    fprintf("⚠️ Stage 1 已初步适应 tracking，但鲁棒性仍可提升。\n\n");
else
    fprintf("✅ Stage 1 表现良好，进入 Stage 2\n\n");
end

%% =========================================================
%  Stage 2：中度 s2r + 中等目标随机化 + 混合中/大范围初值
%  - th_ref0 主要在 [-0.30, 0.30]
%  - 初始状态采用混合采样：大部分中范围，小部分大范围
%  - 加入电机参数随机化
% ==========================================================
fprintf("=== Stage 2：中度 s2r + 中等目标随机化（1500集，跑满）===\n");

env.ResetFcn = @(in) resetS2R_PPO(in, 2);

trainS2 = rlTrainingOptions(...
    MaxEpisodes                 = 1500, ...
    MaxStepsPerEpisode          = maxSteps, ...
    ScoreAveragingWindowLength  = 50, ...
    Verbose                     = true, ...
    Plots                       = "training-progress", ...
    StopOnError                 = "on", ...
    StopTrainingCriteria        = "EpisodeCount", ...
    StopTrainingValue           = 1500, ...
    SaveAgentCriteria           = "AverageReward", ...
    SaveAgentValue              = 520, ...
    SaveAgentDirectory          = "savedAgents_ppo_s2r_s2");

statsS2 = train(agent, env, trainS2);
save("agent_ppo_s2r_stage2.mat", "agent", "statsS2");

avgS2 = mean(statsS2.EpisodeReward(max(1,end-49):end));
fprintf("Stage 2 完成，最近50集平均 reward: %.1f\n", avgS2);

if avgS2 < 400
    warning("⚠️ Stage 2 最近50集平均 reward 偏低，建议后续单独补训 Stage 2 或适当减小随机化幅度！");
elseif avgS2 < 520
    fprintf("⚠️ Stage 2 基本通过，但面对更强随机化时可能仍有波动。\n\n");
else
    fprintf("✅ Stage 2 表现良好，进入 Stage 3\n\n");
end

%% =========================================================
%  Stage 3：完整 s2r + 大范围目标随机化 + 大范围初值
%  - th_ref0 扩展到 [-0.50, 0.50]
%  - 初始状态逐步偏向大范围
%  - 完整参数随机化
% ==========================================================
fprintf("=== Stage 3：完整 s2r + 大范围目标随机化（2500集，跑满）===\n");

env.ResetFcn = @(in) resetS2R_PPO(in, 3);

trainS3 = rlTrainingOptions(...
    MaxEpisodes                 = 2500, ...
    MaxStepsPerEpisode          = maxSteps, ...
    ScoreAveragingWindowLength  = 50, ...
    Verbose                     = true, ...
    Plots                       = "training-progress", ...
    StopOnError                 = "on", ...
    StopTrainingCriteria        = "EpisodeCount", ...
    StopTrainingValue           = 2500, ...
    SaveAgentCriteria           = "AverageReward", ...
    SaveAgentValue              = 450, ...
    SaveAgentDirectory          = "savedAgents_ppo_s2r_s3");

statsS3 = train(agent, env, trainS3);
save("agent_ppo_s2r_final.mat", "agent", "statsS3");

avgS3 = mean(statsS3.EpisodeReward(max(1,end-49):end));
fprintf("\n========================================\n");
fprintf("✅ PPO Sim-to-Real 全部训练完成！\n");
fprintf("   最终平均 reward（最近50集）: %.1f\n", avgS3);
fprintf("   最终 Agent 已保存至 agent_ppo_s2r_final.mat\n");
fprintf("========================================\n");

%% =========================================================
%  Reset 函数（PPO版）
% ==========================================================
function in = resetS2R_PPO(in, stage)

    % -----------------------------
    % 根据阶段决定初值范围与目标范围
    % -----------------------------
    switch stage
        case 0
            % 小范围，目标固定 0
            th0_rng   = 0.05;
            thd0_rng  = 0.08;
            al0_rng   = 0.08;
            ald0_rng  = 0.12;
            thref_rng = 0.00;

        case 1
            % 小范围 tracking
            th0_rng   = 0.06;
            thd0_rng  = 0.10;
            al0_rng   = 0.10;
            ald0_rng  = 0.14;
            thref_rng = 0.15;

        case 2
            % 混合采样：70% 中范围，30% 大范围
            if rand < 0.7
                th0_rng  = 0.10;
                thd0_rng = 0.14;
                al0_rng  = 0.16;
                ald0_rng = 0.20;
            else
                th0_rng  = 0.15;
                thd0_rng = 0.20;
                al0_rng  = 0.25;
                ald0_rng = 0.30;
            end
            thref_rng = 0.30;

        otherwise
            % 混合采样：40% 中范围，60% 大范围
            if rand < 0.4
                th0_rng  = 0.10;
                thd0_rng = 0.14;
                al0_rng  = 0.16;
                ald0_rng = 0.20;
            else
                th0_rng  = 0.15;
                thd0_rng = 0.20;
                al0_rng  = 0.25;
                ald0_rng = 0.30;
            end
            thref_rng = 0.50;
    end

    % -----------------------------
    % 随机初始化
    % -----------------------------
    th0     = th0_rng   * (2*rand - 1);
    thd0    = thd0_rng  * (2*rand - 1);
    al0     = al0_rng   * (2*rand - 1);
    ald0    = ald0_rng  * (2*rand - 1);
    th_ref0 = thref_rng * (2*rand - 1);

    % -----------------------------
    % 参数随机化
    % -----------------------------
    p = samplePlantParamsPPO(stage);

    % -----------------------------
    % 写入模型变量
    % -----------------------------
    in = setVariable(in, "th0",     th0);
    in = setVariable(in, "thd0",    thd0);
    in = setVariable(in, "al0",     al0);
    in = setVariable(in, "ald0",    ald0);
    in = setVariable(in, "th_ref0", th_ref0);
    in = setVariable(in, "p",       p);
end

%% =========================================================
%  Plant 参数随机化（PPO版）
% ==========================================================
function p = samplePlantParamsPPO(stage)
    p = getNominalPlantParams();

    if stage == 0
        % ---------------------------------------------------
        % Stage 0：非常轻的 sim2real 扰动
        % ---------------------------------------------------
        p.B_r        = p.B_r  * (1 + 0.08*(2*rand-1));
        p.B_p        = p.B_p  * (1 + 0.08*(2*rand-1));
        p.Fc_r       = p.Fc_r * (1 + 0.08*(2*rand-1));
        p.Fc_p       = p.Fc_p * (1 + 0.08*(2*rand-1));

        p.deadzone   = max(0.02, p.deadzone * (1 + 0.08*(2*rand-1)));
        p.tau_offset = 0.001 * (2*rand - 1);

        % 极小延迟
        p.alpha_act  = 0.97 + 0.03*rand;

        p.dist_th    = 0.0;
        p.dist_al    = 0.0;

    elseif stage == 1
        % ---------------------------------------------------
        % Stage 1：轻度随机化
        % ---------------------------------------------------
        p.B_r        = p.B_r  * (1 + 0.15*(2*rand-1));
        p.B_p        = p.B_p  * (1 + 0.15*(2*rand-1));
        p.Fc_r       = p.Fc_r * (1 + 0.15*(2*rand-1));
        p.Fc_p       = p.Fc_p * (1 + 0.15*(2*rand-1));

        p.deadzone   = max(0.02, p.deadzone * (1 + 0.15*(2*rand-1)));
        p.tau_offset = 0.002 * (2*rand - 1);

        p.alpha_act  = 0.93 + 0.07*rand;

        p.dist_th    = 0.0005 * (2*rand - 1);
        p.dist_al    = 0.0005 * (2*rand - 1);

    elseif stage == 2
        % ---------------------------------------------------
        % Stage 2：中度随机化 + 电机参数
        % ---------------------------------------------------
        p.B_r        = p.B_r  * (1 + 0.25*(2*rand-1));
        p.B_p        = p.B_p  * (1 + 0.25*(2*rand-1));
        p.Fc_r       = p.Fc_r * (1 + 0.30*(2*rand-1));
        p.Fc_p       = p.Fc_p * (1 + 0.30*(2*rand-1));

        p.V_s        = p.V_s  * (1 + 0.10*(2*rand-1));
        p.R_m        = p.R_m  * (1 + 0.10*(2*rand-1));
        p.k_t        = p.k_t  * (1 + 0.08*(2*rand-1));
        p.k_e        = p.k_e  * (1 + 0.08*(2*rand-1));

        p.deadzone   = max(0.02, p.deadzone * (1 + 0.25*(2*rand-1)));
        p.tau_offset = 0.003 * (2*rand - 1);

        % 中等延迟
        p.alpha_act  = 0.88 + 0.10*rand;

        p.dist_th    = 0.001 * (2*rand - 1);
        p.dist_al    = 0.001 * (2*rand - 1);

    else
        % ---------------------------------------------------
        % Stage 3：完整随机化
        % ---------------------------------------------------
        p.J_theta    = p.J_theta * (1 + 0.15*(2*rand-1));
        p.J_alpha    = p.J_alpha * (1 + 0.15*(2*rand-1));
        p.m_p        = p.m_p    * (1 + 0.10*(2*rand-1));

        p.B_r        = p.B_r    * (1 + 0.35*(2*rand-1));
        p.B_p        = p.B_p    * (1 + 0.35*(2*rand-1));
        p.Fc_r       = p.Fc_r   * (1 + 0.40*(2*rand-1));
        p.Fc_p       = p.Fc_p   * (1 + 0.40*(2*rand-1));

        p.V_s        = p.V_s    * (1 + 0.15*(2*rand-1));
        p.R_m        = p.R_m    * (1 + 0.15*(2*rand-1));
        p.k_t        = p.k_t    * (1 + 0.10*(2*rand-1));
        p.k_e        = p.k_e    * (1 + 0.10*(2*rand-1));
        p.eta_m      = p.eta_m  * (1 + 0.10*(2*rand-1));
        p.eta_g      = p.eta_g  * (1 + 0.10*(2*rand-1));

        p.deadzone   = max(0.02, p.deadzone * (1 + 0.30*(2*rand-1)));
        p.tau_offset = 0.004 * (2*rand - 1);

        % 完整延迟范围，但仍避免过于激进
        p.alpha_act  = 0.80 + 0.20*rand;

        p.dist_th    = 0.002 * (2*rand - 1);
        p.dist_al    = 0.002 * (2*rand - 1);
    end

    % 先不加传感器噪声，避免训练目标过多耦合
    p.noise_al_std  = 0.0;
    p.noise_th_std  = 0.0;
    p.noise_ald_std = 0.0;
    p.noise_thd_std = 0.0;
end
