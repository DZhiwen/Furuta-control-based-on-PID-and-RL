clear;
clc;
close all;

%% =========================
%  基本设置
% ==========================
mdl = "furuta_rl_balance";
agentBlk = mdl + "/Subsystem2/RL Agent";
Ts = 0.01;
Tf = 5;
maxSteps = ceil(Tf / Ts);

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
env.ResetFcn = @(in)localResetFcn(in);

%% =========================
%  Actor: 5 -> 64 -> 32 -> 1
% ==========================
actorLG = layerGraph();

obsInput = featureInputLayer(5, Normalization="none", Name="obs");

commonLayers = [
    fullyConnectedLayer(64, Name="actor_fc1")
    reluLayer(Name="actor_relu1")
    fullyConnectedLayer(32, Name="actor_fc2")
    reluLayer(Name="actor_relu2")
    ];

meanLayers = [
    fullyConnectedLayer(1, Name="mean_fc")
    tanhLayer(Name="mean_tanh")
    ];

stdLayers = [
    fullyConnectedLayer(1, Name="std_fc")
    softplusLayer(Name="std_softplus")
    ];

actorLG = addLayers(actorLG, obsInput);
actorLG = addLayers(actorLG, commonLayers);
actorLG = addLayers(actorLG, meanLayers);
actorLG = addLayers(actorLG, stdLayers);

actorLG = connectLayers(actorLG, "obs", "actor_fc1");
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

%% =========================
%  Critic: 5 -> 64 -> 32 -> 1
% ==========================
criticLG = layerGraph([
    featureInputLayer(5, Normalization="none", Name="obs")
    fullyConnectedLayer(64, Name="critic_fc1")
    reluLayer(Name="critic_relu1")
    fullyConnectedLayer(32, Name="critic_fc2")
    reluLayer(Name="critic_relu2")
    fullyConnectedLayer(1, Name="value")
    ]);

criticNet = dlnetwork(criticLG);

critic = rlValueFunction(criticNet, obsInfo, ...
    ObservationInputNames="obs");

%% =========================
%  PPO参数
% ==========================
agentOpts = rlPPOAgentOptions( ...
    ExperienceHorizon=512, ...
    ClipFactor=0.2, ...
    EntropyLossWeight=0.005, ...
    MiniBatchSize=128, ...
    NumEpoch=4, ...
    AdvantageEstimateMethod="gae", ...
    GAEFactor=0.97, ...
    SampleTime=Ts, ...
    DiscountFactor=0.995);

agentOpts.ActorOptimizerOptions.LearnRate = 1e-4;
agentOpts.ActorOptimizerOptions.GradientThreshold = 1.0;

agentOpts.CriticOptimizerOptions.LearnRate = 3e-4;
agentOpts.CriticOptimizerOptions.GradientThreshold = 1.0;

agent = rlPPOAgent(actor, critic, agentOpts);

%% =========================
% 训练参数
% ==========================
trainOpts = rlTrainingOptions( ...
    MaxEpisodes=8000, ...
    MaxStepsPerEpisode=maxSteps, ...
    ScoreAveragingWindowLength=50, ...
    Verbose=true, ...
    Plots="training-progress", ...
    StopOnError="on", ...
    SaveAgentCriteria="AverageReward", ...
    SaveAgentValue=200, ...
    SaveAgentDirectory="savedAgents_small_64_32");

trainingStats = train(agent, env, trainOpts);

save("furuta_ppo_agent_small_64_32.mat", "agent", "trainingStats");

disp("训练完成，agent 已保存到 furuta_ppo_agent_small_64_32.mat");
