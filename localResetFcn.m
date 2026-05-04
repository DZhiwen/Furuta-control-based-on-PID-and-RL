function in = localResetFcn(in)
% 每个episode开始时随机重置初始状态和目标位置
%
% 需要你的Simulink模型中存在这些变量：
%   th0, thd0, al0, ald0, th_ref0

% -----------------------------
% 初始状态随机化（倒立附近）
% -----------------------------
% 旋转臂角度
 th0 = 0.15 * (2*rand - 1);      % [-0.15, 0.15] rad
%th0  = 0.05 * (2*rand - 1);
% 旋转臂角速度
 thd0 = 0.20 * (2*rand - 1);     % [-0.20, 0.20] rad/s
%thd0 = 0.08 * (2*rand - 1);
% 摆杆角度（0为倒立）
al0 = 0.25 * (2*rand - 1);      % [-0.25, 0.25] rad
%al0  = 0.08 * (2*rand - 1);
% 摆杆角速度
ald0 = 0.30 * (2*rand - 1);     % [-0.30, 0.30] rad/s
%ald0 = 0.12 * (2*rand - 1);

% -----------------------------
% 目标位置随机化
% -----------------------------

%th_ref0 = 0.50 * (2*rand - 1);  % [-0.50, 0.50] rad
th_ref0 = 0;

% -----------------------------
% 写入模型变量
% -----------------------------
in = setVariable(in, "th0", th0);
in = setVariable(in, "thd0", thd0);
in = setVariable(in, "al0", al0);
in = setVariable(in, "ald0", ald0);
in = setVariable(in, "th_ref0", th_ref0);


end
