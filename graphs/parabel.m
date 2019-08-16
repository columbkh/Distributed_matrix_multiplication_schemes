gasp_dl = [0.012890,   0.016032,   0.025241,   0.044920,   0.059240,   0.090297,   0.142696,   0.233566]

ass_dl =  [0.012286,   0.010634,   0.031963,   0.038779,   0.060943,   0.084719,   0.156178,   0.247696]

scs_dl =  [0.011261,   0.016352,   0.011740,   0.025683,   0.048896,   0.062536,   0.102179,   0.147719]

gasp_ul = [0.019968,   0.035512,   0.054409,   0.087941,   0.154048,   0.257189,   0.436932,   0.718281]

ass_ul =  [0.026491,   0.042307,   0.077121,   0.123685,   0.216905,   0.378109,   0.631557,   1.062925]

scs_ul =  [0.091265,    0.137291,    0.251251,    0.429027,    0.736594,    1.569412,    4.835482,   10.210443]

gasp_dec = [0.13069,   0.16560,   0.25361,   0.39763,   0.61636,   1.02546,   1.72735,   2.89857]

ass_dec = [0.12255,   0.15902,   0.25202,   0.38022,   0.60927,   0.97828,   1.69046,   2.74486]

scs_dec = [0.14189,   0.16123,   0.22080,   0.29207,   0.41945,   0.61245,   0.99526,   1.61447]

gasp_comp = [ 6.4062e-04,   1.4178e-03,   3.1965e-03,   5.3648e-03,   1.1885e-02,   2.7140e-02,   6.3369e-02,   1.3947e-01]

ass_comp = [8.4122e-04  , 1.4606e-03 ,  3.3936e-03  , 5.6287e-03 ,  1.6385e-02,   3.2268e-02,   7.1321e-02,   1.5396e-01]

scs_comp = [0.0056650 ,  0.0109217 ,  0.0283332 ,  0.0593987 ,  0.1308527  , 0.2642614  , 0.5812315 ,  1.2390475]

res_gasp = [0.16419 ,  0.21856 ,  0.33646 ,  0.53585 ,  0.84153 ,  1.40008 ,  2.37035,   3.98989]

res_ass = [0.16217 ,  0.21342  , 0.36450 ,  0.54831 ,  0.90351 ,  1.47338  , 2.54951  , 4.20943]

res_scs = [0.25008  ,  0.32579  ,  0.51213 ,   0.80618  ,  1.33579  ,  2.50866  ,  6.51415 ,  13.21168]

x = [1000, 1300, 1690, 2197, 2857, 3715, 4830, 6279]


 clear title
  fig = figure();
  subplot(3,2,1)
  gasp_dl
  plot(x, gasp_dl, 'Color', 'b')
  hold on;
  ass_dl
  scs_dl
  plot(x, ass_dl, 'Color', 'y')
  plot(x, scs_dl, 'Color', 'r')
  xlabel ("Matrix Size");
  ylabel ("Time");
  title("Download");
  
  hold off;
  subplot(3,2,2);
  gasp_ul
  ass_ul
  scs_ul
  plot(x, gasp_ul, 'Color', 'b');
  hold on;
  plot(x, ass_ul, 'Color', 'y');
  plot(x, scs_ul, 'Color', 'r');
 # legend({'gasp','ass', 'scs'});
  xlabel ("Matrix Size");
  ylabel ("Time");
  title("Upload");
  hold off;
  subplot(3,2,3);
  gasp_dec
  ass_dec
  scs_dec
  plot(x, gasp_dec, 'Color', 'b');
  hold on;
  plot(x, ass_dec, 'Color', 'y');
  plot(x, scs_dec, 'Color', 'r');
 # legend({'gasp','ass', 'scs'});
  xlabel ("Matrix Size");
  ylabel ("Time");
  title("Decoding");
  hold off;
  subplot(3,2,4);
  gasp_comp
  ass_comp
  scs_comp
  y1 = plot(x, gasp_comp, 'Color', 'b');
  hold on;
  y2 = plot(x, ass_comp, 'Color', 'y');
  y3 = plot(x, scs_comp, 'Color', 'r');
  #legend({'gasp','ass', 'scs'});
  xlabel ("Matrix Size");
  ylabel ("Time");
  title("Slaves");
  hold off;
  subplot(3,2,5);
  res_gasp = gasp_dl + gasp_ul + gasp_dec + gasp_comp;
  res_ass = ass_dl + ass_ul + ass_dec + ass_comp;
  res_scs = scs_dl + scs_ul + scs_dec + scs_comp;
  res_gasp
  res_ass
  res_scs
  plot(x, res_gasp, 'Color', 'b');
  hold on;
  plot(x, res_ass, 'Color', 'y');
  plot(x, res_scs, 'Color', 'r');
 # legend({'gasp','ass', 'scs'});
  xlabel ("Matrix Size");
  ylabel ("Time");
  title("Total");
  hold off;
  #saveas(fig, res);
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  