uscsa_dl = [0.083004, 0.057772, 0.056990];
uscsa_ul = [0.45228, 0.84057, 4.7486];

gscsa_dl = [0.087747, 0.067380, 0.061682];
gscsa_ul = [0.45059, 0.52565, 0.71185];

new_uscsa_dl = [0.15060, 0.10051, 0.11030];
new_uscsa_ul = [0.51216, 0.70818 , 7.8406];

new_gscsa_dl = [0.16184, 0.10251, 0.12541];
new_gscsa_ul = [0.47769, 0.52616, 0.76654];


fig = figure()
subplot(3,2,1)
plot(gscsa_ul, gscsa_dl, 'Color', 'b')
hold on;
plot(uscsa_ul, uscsa_dl, 'Color', 'y')
legend({'gscsa','uscsa'});
hold off;
subplot(3,2,2);
  
plot(new_gscsa_ul, new_gscsa_dl, 'Color', 'b')
hold on;
plot(new_uscsa_ul, new_uscsa_dl, 'Color', 'y')
legend({'gscsa','uscsa'});
hold off;

