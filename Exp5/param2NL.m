% param2NL.m 
% Parâmetros do Processo de nível de líquido de 2a ordem
% LARA-UnB

% Autor: Adolfo Bauchspiess @ Brasília, 10 de novembro de 2010

% Parametros do Processo 
Ar=6*25.4;  % cm2 - Área da seção transversal dos tanques  

k12=38;     % válvula entre tanques no 3o engaste
k2=11.5;    % parâmetro do furo do tanque de saída

hmax=16;    % altura máxima antes de sair água pelo ladrão
qmax=90;    % vazão máxima da bomba +
qmin=0;     % vazão máxima da bomba - % qmin=-85; 

narmaL2
sim('narmaL2');     % Controle Neural Narma-L2
