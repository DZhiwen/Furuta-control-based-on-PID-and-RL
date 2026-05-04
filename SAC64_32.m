clear;
clc;
close all;

%% =========================
%  基本设置
% ==========================
mdl = "furuta_rl_balance";                 % Simulink模型名
agentBlk = mdl + "/Subsystem2/RL Agent";  % RL Agent模块路径
Ts = 0.01;                                % 采样时间
Tf = 5;                                   % 每回合仿真时长（秒）
maxSteps = ceil(Tf / Ts);

%% 打开模型
open_system(mdl);

%% =========================
%  Observation / Action 规格
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

%% =========================
%  创建Simulink环境
% ==========================
env = rlSimulinkEnv(mdl, agentBlk, obsInfo, actInfo);
env.ResetFcn = @(in)localResetFcn(in);

%% =========================================================
%  创建 SAC Actor
%  结构: 5 -> 64 -> 32 -> mean/std
% ==========================================================
statePath = [
    featureInputLayer(5, Normalization="none", Name="obs")
    fullyConnectedLayer(64, Name="actor_fc1")
    reluLayer(Name="actor_relu1")
    fullyConnectedLayer(32, Name="actor_fc2")
    reluLayer(Name="actor_relu2")
    ];

meanPath = [
    fullyConnectedLayer(1, Name="mean_fc")
    tanhLayer(Name="mean_tanh")
    ];

stdPath = [
    fullyConnectedLayer(1, Name="std_fc")
    softplusLayer(Name="std_softplus")
    ];

actorLG = layerGraph(statePath);
actorLG = addLayers(actorLG, meanPath);
actorLG = addLayers(actorLG, stdPath);

actorLG = connectLayers(actorLG, "actor_relu2", "mean_fc");
actorLG = connectLayers(actorLG, "actor_relu2", "std_fc");

actorNet = dlnetwork(actorLG);

actor = rlContinuousGaussianActor( ...
    actorNet, ...
    obsInfo, ...
    actInfo, ...
    ObservationInputNames="obs", ...
    ActionMeanOutputNames="mean_tanh", ...
    ActionStandardDeviationOutputNames="std_softplus");

%% =========================================================
%  创建 Q Critic 1
%  结构: [obs, act] -> 64 -> 32 -> 1
% ==========================================================
criticStatePath1 = [
    featureInputLayer(5, Normalization="none", Name="obs")
    fullyConnectedLayer(64, Name="c1_state_fc1")
    reluLayer(Name="c1_state_relu1")
    ];

criticActionPath1 = [
    featureInputLayer(1, Normalization="none", Name="act")
    fullyConnectedLayer(64, Name="c1_action_fc1")
    ];

criticCommonPath1 = [
    additionLayer(2, Name="c1_add")
    reluLayer(Name="c1_relu1")
    fullyConnectedLayer(32, Name="c1_fc2")
    reluLayer(Name="c1_relu2")
    fullyConnectedLayer(1, Name="q1")
    ];

criticLG1 = layerGraph();
criticLG1 = addLayers(criticLG1, criticStatePath1);
criticLG1 = addLayers(criticLG1, criticActionPath1);
criticLG1 = addLayers(criticLG1, criticCommonPath1);

criticLG1 = connectLayers(criticLG1, "c1_state_relu1", "c1_add/in1");
criticLG1 = connectLayers(criticLG1, "c1_action_fc1", "c1_add/in2");

criticNet1 = dlnetwork(criticLG1);

critic1 = rlQValueFunction( ...
    criticNet1, ...
    obsInfo, ...
    actInfo, ...
    ObservationInputNames="obs", ...
    ActionInputNames="act");

%% =========================================================
%  创建 Q Critic 2
%  结构: [obs, act] -> 64 -> 32 -> 1
% ==========================================================
criticStatePath2 = [
    featureInputLayer(5, Normalization="none", Name="obs")
    fullyConnectedLayer(64, Name="c2_state_fc1")
    reluLayer(Name="c2_state_relu1")
    ];

criticActionPath2 = [
    featureInputLayer(1, Normalization="none", Name="act")
    fullyConnectedLayer(64, Name="c2_action_fc1")
    ];

criticCommonPath2 = [
    additionLayer(2, Name="c2_add")
    reluLayer(Name="c2_relu1")
    fullyConnectedLayer(32, Name="c2_fc2")
    reluLayer(Name="c2_relu2")
    fullyConnectedLayer(1, Name="q2")
    ];

criticLG2 = layerGraph();
criticLG2 = addLayers(criticLG2, criticStatePath2);
criticLG2 = addLayers(criticLG2, criticActionPath2);
criticLG2 = addLayers(criticLG2, criticCommonPath2);

criticLG2 = connectLayers(criticLG2, "c2_state_relu1", "c2_add/in1");
criticLG2 = connectLayers(criticLG2, "c2_action_fc1", "c2_add/in2");

criticNet2 = dlnetwork(criticLG2);

critic2 = rlQValueFunction( ...
    criticNet2, ...
    obsInfo, ...
    actInfo, ...
    ObservationInputNames="obs", ...
    ActionInputNames="act");

%% =========================
%  SAC Agent 参数
% ==========================
agentOpts = rlSACAgentOptions(...
    SampleTime=Ts, ...
    DiscountFactor=0.995, ...
    ExperienceBufferLength=1e6, ...
    MiniBatchSize=256, ...
    TargetSmoothFactor=5e-3);

agentOpts.ActorOptimizerOptions.LearnRate = 1e-4;
agentOpts.ActorOptimizerOptions.GradientThreshold = 1.0;

agentOpts.CriticOptimizerOptions(1).LearnRate = 1e-3;
agentOpts.CriticOptimizerOptions(1).GradientThreshold = 1.0;

agentOpts.CriticOptimizerOptions(2).LearnRate = 1e-3;
agentOpts.CriticOptimizerOptions(2).GradientThreshold = 1.0;

% 若你的MATLAB版本支持自动熵调节，保留下面这一句
agentOpts.EntropyWeightOptions.LearnRate = 1e-4;

%% =========================
%  创建 SAC Agent
% ==========================
agent = rlSACAgent(actor, [critic1 critic2], agentOpts);

%% =========================
%  训练参数
% ==========================
trainOpts = rlTrainingOptions( ...
    MaxEpisodes=5000, ...
    MaxStepsPerEpisode=maxSteps, ...
    ScoreAveragingWindowLength=50, ...
    Verbose=true, ...
    Plots="training-progress", ...
    StopOnError="on", ...
    SaveAgentCriteria="AverageReward", ...
    SaveAgentValue=200, ...
    SaveAgentDirectory="savedAgents_sac_64_32");

%% =========================
%  开始训练
% ==========================
trainingStats = train(agent, env, trainOpts);

%% =========================
%  保存结果
% ==========================
save("furuta_sac_agent_64_32.mat", "agent", "trainingStats");

disp("SAC(64-32)训练完成，agent 已保存到 furuta_sac_agent_64_32.mat");
