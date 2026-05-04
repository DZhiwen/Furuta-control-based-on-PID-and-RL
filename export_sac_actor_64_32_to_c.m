clear; clc;

%% =========================
% 加载训练好的 SAC agent
% ==========================
load("agent_ppo_s2r_final.mat", "agent");

actor = getActor(agent);
L = actor.Learnables;

disp("===== Learnables size =====");
for i = 1:numel(L)
    x = extractdata(L{i});
    fprintf("L{%d}: ", i);
    disp(size(x));
end

%% =========================
% 按 64-32 Actor 提取 mean 分支参数
% 预计：
% L{1} = W1  [64x5]
% L{2} = b1  [64x1]
% L{3} = W2  [32x64]
% L{4} = b2  [32x1]
% L{5} = Wm  [1x32]
% L{6} = bm  [1x1]
% L{7} = Ws  [1x32]
% L{8} = bs  [1x1]
% ==========================
W1 = extractdata(L{1});
b1 = extractdata(L{2});
W2 = extractdata(L{3});
b2 = extractdata(L{4});
Wm = extractdata(L{5});
bm = extractdata(L{6});

%% =========================
% 写入头文件
% ==========================
fid = fopen("sac_actor_weights_64_32.h", "w");

fprintf(fid, "#ifndef __SAC_ACTOR_WEIGHTS_64_32_H__\n");
fprintf(fid, "#define __SAC_ACTOR_WEIGHTS_64_32_H__\n\n");

fprintf(fid, "#define ACTOR_IN_DIM   5\n");
fprintf(fid, "#define ACTOR_H1_DIM   64\n");
fprintf(fid, "#define ACTOR_H2_DIM   32\n");
fprintf(fid, "#define ACTOR_OUT_DIM  1\n\n");

write_matrix(fid, "actor_W1", W1);
write_vector(fid, "actor_b1", b1);

write_matrix(fid, "actor_W2", W2);
write_vector(fid, "actor_b2", b2);

write_matrix(fid, "actor_Wm", Wm);
write_vector(fid, "actor_bm", bm);

fprintf(fid, "\n#endif\n");
fclose(fid);

disp("导出完成：sac_actor_weights_64_32.h");

%% =========================
% 本地校验：给一个测试输入做前向
% 注意：SAC actor 在 MATLAB 中 getAction 可能带随机性
% 部署时我们只用 mean 分支，因此这里更建议只看权重导出，不直接拿 getAction 做严格对比
% ==========================
obs = [0;0;1;0;0];
disp("SAC actor weight export done. Example obs:");
disp(obs);

%% =========================
% 辅助函数
% ==========================
function write_matrix(fid, name, M)
    [r,c] = size(M);
    fprintf(fid, "static const float %s[%d][%d] = {\n", name, r, c);
    for i = 1:r
        fprintf(fid, "    {");
        for j = 1:c
            if j < c
                fprintf(fid, "%.9ff, ", M(i,j));
            else
                fprintf(fid, "%.9ff", M(i,j));
            end
        end
        if i < r
            fprintf(fid, "},\n");
        else
            fprintf(fid, "}\n");
        end
    end
    fprintf(fid, "};\n\n");
end

function write_vector(fid, name, v)
    v = v(:);
    n = numel(v);
    fprintf(fid, "static const float %s[%d] = {", name, n);
    for i = 1:n
        if i < n
            fprintf(fid, "%.9ff, ", v(i));
        else
            fprintf(fid, "%.9ff", v(i));
        end
    end
    fprintf(fid, "};\n\n");
end
