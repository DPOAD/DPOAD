%%%DPOD

clc
filename = 'Cherry.csv';
A = readtable(filename);
% % load densitiesToPlot
B=A(:,2:28);
%dataset2=dataset4(:,1:3);
B=table2array(B);
%load 4.mat
% % load densitiesToPlot
% load S
sizeofD=200;
%dataset2=dataset4(:,1:3);
% T=table2array(dataset2);

%Candidates=data;
%Candidates=data;
%% Execute the Autonomous Anomaly Detection software
%Input.Data=tabulatedLoopData(:,3); % input
% [Output]=AutonomusAnomalyDetection(Input);
eps=linspace(0.1,2,10);
% B=tabulatedLoopData(:,3);


eps=1;
gamma=0.2;
rho = exp(lambertw(-1,-gamma / (2 * exp(0.5))) + 0.5);
m = log(1/rho)/(2*(gamma-rho)^2);
k = (m * (1 - gamma + rho + sqrt(log(1/rho)/(2 * m))));
%m=ceil(m);
m=100;
%k=ceil(k); 
k=80;
sizeofDD=500;
Divlap=[];
Divdpoad=[];
Divpainfree=[];
%S=histogram(dataset2,sizeofDD);
%S=S.Values;
%B=S';
Lapsens=max(B);
EVAL=zeros(10,4);
EVAL1=zeros(10,4);
EVAL2=zeros(10,4);
for j=1:1
for rr=1:1
%S=histogram(dataset2(1:rr*sizeofD,:),sizeofDD);
%S=S.Values;
%B=S';
sizeofDD=rr*sizeofD;
px=[1:1:sizeofDD];
C=B(1:rr*sizeofD);
%C=B;
[Output]=kse_test(C);
pp=1./(Output.^4);
pp=pp';
warning('off','all')
%Buiolding CDF for sensitivity sampler %%%%%%%%%%%%%%%%%%%%
%%%first round is uniform
GS=datasample(C,m);
GS=sort(GS);
Sensunif=GS(k);
DD=randpdf(Output,px,[m,1]);
DD=Output(floor(DD));
DD=sort(DD);
Sensorig=DD(k);
Senspainfree=Sensunif;
if rr==1
unif=laprnd(sizeofDD, 1, 0,Sensunif/(eps))+C;
[noiseunif]=kse_test(unif);
[pol,DS,mu]=polyfit(C,noiseunif,7);

Divpainfree=[Divpainfree kldiv(Output',noiseunif')];

Lap=laprnd(sizeofDD, 1, 0,Lapsens/(eps))+C;

Lap=kse_test(Lap);

Divlap=[Divlap kldiv(Output',Lap')];

Divdpoad=[Divdpoad kldiv(Output',noiseunif')];

dpoddist=laprnd(sizeofDD, 1, 0,Senspainfree/(eps))+C;
dpoddist=kse_test(dpoddist);

painfree=dpoddist;

noiseunif=1./(noiseunif.^4);
noiseunif=noiseunif';
end



%%%next round rounds are anomaly exclusion


    rr

  
if rr>1

SS=randpdf(noiseunif',pxx,[m,1]);

X=polyval(pol,C,[],mu);
mu
X=normalize(X,'range');

SS=X(floor(SS));
SS=sort(SS);nm                           
Sensanom=SS(k);

FF=datasample(C,m);
FF=sort(FF);
Senspainfree=FF(k);



%SensDPOAD=[SensDPOAD Sensanom];
%Sensorig=[Sensorig Sensorig]; 
%Senspainfree=[Senspainfree Senspainfree];
painfree=laprnd(sizeofDD, 1, 0,Senspainfree/(eps))+C;
painfree=kse_test(painfree);
Divpainfree=[Divpainfree kldiv(Output',painfree')];

Lap=laprnd(sizeofDD, 1, 0,Lapsens/(eps))+C;

Lap=kse_test(Lap);

Divlap=[Divlap kldiv(Output',Lap')];

noiseunif=laprnd(sizeofDD, 1, 0,Sensanom/(eps))+X;
noiseanom=kse_test(noiseunif);
Divdpoad=[Divdpoad kldiv(Output',noiseanom')];
[pol,DS,mu]= polyfit(C,noiseanom,6);
dpoddist=noiseanom;
noiseanom=1./(noiseanom.^4);
noiseunif=noiseanom;
end
pxx=[1:1:sizeofDD];
end
% %Testing 
    %original results
Index=[1:1:sizeofDD];
Index=[Index',zeros(sizeofDD,1)];
Output=normalize(Output,'range');
for i=1:sizeofDD
    if Output(i)>0.8
       Index(i,2)=1;
    end
end
ACTUAL=Index;
Origindex=find(ACTUAL(:,2)==1);
Lap=normalize(Lap,'range');
%Laplace results
Index=[1:1:sizeofDD];
Index=[Index',zeros(sizeofDD,1)];
for i=1:sizeofDD
    if Lap(i)>0.8
       Index(i,2)=1;
    end
end
LAP=Index;
Pertindex2=find(LAP(:,2)==1);
%Painfree results
Index=[1:1:sizeofDD];
Index=[Index',zeros(sizeofDD,1)];
painfree=normalize(painfree,'range');
for i=1:sizeofDD
    if painfree(i)>0.8
       Index(i,2)=1;
    end
end
Uniformm=Index;
Pertindex=find(Uniformm(:,2)==1);

%DPOD results
Index=[1:1:sizeofDD];
Index=[Index',zeros(sizeofDD,1)];
dpoddist=normalize(dpoddist,'range');
for i=1:sizeofDD
    if dpoddist(i)>0.8
       Index(i,2)=1;
    end
end
Anomm=Index;
Pertindex1=find(Anomm(:,2)==1);


IND2=LAP;
IND1=Anomm;
IND=Uniformm;
% Evaluate(ACTUAL,LAP)
C=union(Origindex,Pertindex2);
C=size(C,1);
p = size(Origindex,1);
n = 5*sizeofD-p;
N = p+n;
tp = sum(ACTUAL(Origindex,2)==IND2(Origindex,2));
tn = N-C;
fp = n-tn;
fn = p-tp;
tp_rate = tp/p;
tn_rate = tn/n;
accuracy = (tp+tn)/N;
sensitivity = tp_rate;
specificity = tn_rate;
precision = tp/(tp+fp);
recall = sensitivity;
f_measure = 2*((precision*recall)/(precision + recall));
gmean = sqrt(tp_rate*tn_rate);
EVAL(j,:)= [tp precision recall f_measure];

% Evaluate(ACTUAL,PAinfree)
C=union(Origindex,Pertindex);
C=size(C,1);
p = size(Origindex,1);
n = 5*sizeofD-p;
N = p+n;
tp = sum(ACTUAL(Origindex,2)==IND(Origindex,2));
tn = N-C;
fp = n-tn;
fn = p-tp;
tp_rate = tp/p;
tn_rate = tn/n;
accuracy = (tp+tn)/N;
sensitivity = tp_rate;
specificity = tn_rate;
precision = tp/(tp+fp);
recall = sensitivity;
f_measure = 2*((precision*recall)/(precision + recall));
gmean = sqrt(tp_rate*tn_rate);
EVAL1(j,:)= [tp precision recall f_measure];

% Evaluate(ACTUAL,DPOD)
C=union(Origindex,Pertindex1);
C=size(C,1);
p = size(Origindex,1);
n = 5*sizeofD-p;
N = p+n;
tp = sum(ACTUAL(Origindex,2)==IND1(Origindex,2));
tn = N-C;
fp = n-tn;
fn = p-tp;
tp_rate = tp/p;
tn_rate = tn/n;
accuracy = (tp+tn)/N;
sensitivity = tp_rate;
specificity = tn_rate;
precision = tp/(tp+fp);
recall = sensitivity;
f_measure = 2*((precision*recall)/(precision + recall));
gmean = sqrt(tp_rate*tn_rate);
EVAL2(j,:)= [tp precision recall f_measure];
end

