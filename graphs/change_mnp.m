
a = "Hi there!"


function [mu, sigma] = make_pdf(list, i, name)
  subplot(2,3,i);
  n = numel(list);
  k = kk(n);
  [f, x] = hist(list, k);
  g = f / trapz(x, f);
  bar(x, g);
  hold on;
  [cx, nrm, mu, sigma] = make_norm_pdf(list);
  plot(cx, nrm);
  hold off;
  xlabel ("Time");
  ylabel ("PDF Value");
  title(name);
endfunction

function [cx, nrm, mu, sigma] = make_norm_pdf(list)
  mu = mean(list);
  sigma = std(list);
  cx = unique(list);
  nrm = normpdf(cx, mu, sigma);
endfunction


function k = kk(x)
k = 1 + round(3.322*log10(x));
endfunction

function [fig, dlv, ulv, decv, compv, encv] = make_pdfs(dl, ul, dec, comp, enc)
  fig = figure('visible','off');
  [mu, sigma] = make_pdf(dl, 1, "Download");
  dlv = [mu, sigma];
  [mu, sigma] = make_pdf(ul, 2, "Upload");
  ulv = [mu, sigma];
  [mu, sigma] = make_pdf(dec, 3, "Decoding");
  decv = [mu, sigma];
  [mu, sigma] = make_pdf(comp, 4, "Slaves");
  compv = [mu, sigma];
  [mu, sigma] = make_pdf(enc, 5, "Encoding");
  encv = [mu, sigma];
endfunction

function [dl, ul, dec, comp, enc] = load_data(name)
    load(name);
    dl = FrameStack{1};
    ul = FrameStack{2};
    dec = FrameStack{3};
    comp = FrameStack{4};
    enc = FrameStack{5};
endfunction

function save_pdf(fig, name, q, Field, m, n, p, ordner)
  q_str = num2str(q);
  q_str = strcat("/Q_", q_str);
  field_str = num2str(Field);
  field_str = strcat("Field_", field_str);
  ms_str = num2str(m);
  ms_str = strcat("m_", ms_str);
  ms_str = strcat(num2str(n), ms_str);
  ms_str = strcat("n_", ms_str);
  ms_str = strcat(num2str(p), ms_str);
  ms_str = strcat("p_", ms_str);
  res = strcat(q_str, field_str);
  res = strcat(res, ms_str);
  res = strcat(ordner, res);
  mkdir(strcat("./oct_results/", ordner));
  res = strcat("./oct_results/", res);
  mkdir(sprintf(res));
  res_file = strcat(res, name);
  saveas(fig, res_file);
endfunction

function [dlv, ulv, decv, compv, encv] = work_with_scheme(name, q, Field, m, n, p, ordner)
  [dl, ul, dec, comp, enc] = load_data(strcat(ordner, name));
  [fig, dlv, ulv, decv, compv, encv] = make_pdfs(dl, ul, dec, comp, enc);
  #save_pdf(fig, name, q, Field, m, n, p, ordner);
endfunction


function [gaspv, assv, scsv, uscsav, gscsav] = work_with_schemes(q, Field, m, n, p, ordner)
  q_str = strcat("_Q_", num2str(q));
  m_str = strcat("_m_", num2str(m));
  n_str = strcat("_n_", num2str(n));
  p_str = strcat("_p_", num2str(p));
  name = strcat(q_str, m_str);
  name = strcat(name, n_str);
  name = strcat(name, p_str);
 # name = strcat(name, ".mat");
  gasp_addr = strcat("/gasp", name);
  ass_addr = strcat("/ass", name);
  scs_addr = strcat("/scsa", name);
  uscsa_addr = strcat("/uscsa", name);
  gscsa_addr = strcat("/gscsa", name);
  [dlv, ulv, decv, compv, encv] = work_with_scheme(gasp_addr, q, Field, m, n,  p, ordner);
  gaspv = [dlv, ulv, decv, compv, encv];
  [dlv, ulv, decv, compv, encv] = work_with_scheme(ass_addr, q, Field, m, n, p, ordner);
  assv = [dlv, ulv, decv, compv, encv];
  [dlv, ulv, decv, compv, encv] = work_with_scheme(scs_addr, q, Field, m, n, p, ordner);
  scsv = [dlv, ulv, decv, compv, encv];

  [dlv, ulv, decv, compv, encv] = work_with_scheme(uscsa_addr, q, Field, m, n, p, ordner);
  uscsav = [dlv, ulv, decv, compv, encv];

  [dlv, ulv, decv, compv, encv] = work_with_scheme(gscsa_addr, q, Field, m, n, p, ordner);
  gscsav = [dlv, ulv, decv, compv, encv];

endfunction

function work_with_experiment(q, Field, m, n, p, ordner, nomer, coeff, title)
  gasp_dl = [];
  gasp_ul = [];
  gasp_dec = [];
  gasp_comp = [];
  gasp_enc = [];
  ass_dl = [];
  ass_ul = [];
  ass_dec = [];
  ass_comp = [];
  ass_enc = [];
  scs_dl = [];
  scs_ul = [];
  scs_dec = [];
  scs_comp = [];
  scs_enc = [];
  uscsa_dl = [];
  uscsa_ul = [];
  uscsa_dec = [];
  uscsa_comp = [];
  uscsa_enc = [];
  gscsa_dl = [];
  gscsa_ul = [];
  gscsa_dec = [];
  gscsa_comp = [];
  gscsa_enc = [];
  matr = [];

  for i = 1:nomer
    [gaspv, assv, scsv, uscsav, gscsav] = work_with_schemes(q, Field, m, n, p, ordner);
    matr = [matr, p];
    m
    n
    p
    gasp_enc
    ass_enc
    scs_enc
    uscsa_enc
    gscsa_enc
    gasp_dl = [gasp_dl, gaspv(1)];
    gasp_ul = [gasp_ul, gaspv(3)];
    gasp_dec = [gasp_dec, gaspv(5)];
    gasp_comp = [gasp_comp, gaspv(7)];
    gasp_enc = [gasp_enc, gaspv(9)];
    ass_dl = [ass_dl, assv(1)];
    ass_ul = [ass_ul, assv(3)];
    ass_dec = [ass_dec, assv(5)];
    ass_comp = [ass_comp, assv(7)];
    ass_enc = [ass_enc, assv(9)];
    scs_dl = [scs_dl, scsv(1)];
    scs_ul = [scs_ul, scsv(3)];
    scs_dec = [scs_dec, scsv(5)];
    scs_comp = [scs_comp, scsv(7)];
    scs_enc = [scs_enc, scsv(9)];
    uscsa_dl = [uscsa_dl, uscsav(1)];
    uscsa_ul = [uscsa_ul, uscsav(3)];
    uscsa_dec = [uscsa_dec, uscsav(5)];
    uscsa_comp = [uscsa_comp, uscsav(7)];
    uscsa_enc = [uscsa_enc, uscsav(9)];
    gscsa_dl = [gscsa_dl, gscsav(1)];
    gscsa_ul = [gscsa_ul, gscsav(3)];
    gscsa_dec = [gscsa_dec, gscsav(5)];
    gscsa_comp = [gscsa_comp, gscsav(7)];
    gscsa_enc = [gscsa_enc, gscsav(9)];

    if title == 1
       m = m + coeff;
       n = n + coeff;
       p = p + coeff;
    else
       m = cast(m, "double");
       m = rround(m * coeff);
       n = cast(n, "double");
       n = rround(n * coeff);
       p = cast(p, "double");
       p = rround(p * coeff);
    end
  end

  x = matr
  clear title
  fig = figure();

  subplot(3,3,1)
  plot(x, gasp_enc, 'Color', 'b');
  hold on;
  plot(x, ass_enc, 'Color', 'y');
  plot(x, scs_enc, 'Color', 'r');
  plot(x, uscsa_enc, 'Color', 'g');
  plot(x, gscsa_enc, 'Color', 'magenta')
 # legend({'gasp','ass', 'scs'});
  xlabel ("Size of p");
  ylabel ("Time [s]");
  title("Encoding");
  hold off;

  subplot(3,3,2);
  plot(x, gasp_ul, 'Color', 'b');
  hold on;
  plot(x, ass_ul, 'Color', 'y');
  plot(x, scs_ul, 'Color', 'r');
  plot(x, uscsa_ul, 'Color', 'g');
  plot(x, gscsa_ul, 'Color', 'magenta')
 # legend({'gasp','ass', 'scs'});
  xlabel ("Size of p");
  ylabel ("Time [s]");
  title("Upload");
  hold off;

  subplot(3,3,3);
  plot(x, gasp_comp, 'Color', 'b');
  hold on;
  plot(x, ass_comp, 'Color', 'y');
  plot(x, scs_comp, 'Color', 'r');
  plot(x, uscsa_comp, 'Color', 'g');
  plot(x, gscsa_comp, 'Color', 'magenta')
  #legend({'gasp','ass', 'scs'});
  xlabel ("Size of p");
  ylabel ("Time [s]");
  title("Computation");
  hold off;


  subplot(3,3,4)
  plot(x, gasp_dl, 'Color', 'b')
  hold on;
  plot(x, ass_dl, 'Color', 'y')
  plot(x, scs_dl, 'Color', 'r')
  plot(x, uscsa_dl, 'Color', 'g')
  plot(x, gscsa_dl, 'Color', 'magenta')
  xlabel ("Size of p");
  ylabel ("Time [s]");
  title("Download");
  hold off;


  subplot(3,3,5);
  plot(x, gasp_dec, 'Color', 'b');
  hold on;
  plot(x, ass_dec, 'Color', 'y');
  plot(x, scs_dec, 'Color', 'r');
  plot(x, uscsa_dec, 'Color', 'g');
  plot(x, gscsa_dec, 'Color', 'magenta')
 # legend({'gasp','ass', 'scs'});
  xlabel ("Size of p");
  ylabel ("Time [s]");
  title("Decoding");
  hold off;

  res_gasp = gasp_dl + gasp_ul + gasp_dec + gasp_comp;
  res_ass = ass_dl + ass_ul + ass_dec + ass_comp;
  res_scs = scs_dl + scs_ul + scs_dec + scs_comp;
  res_uscsa = uscsa_dl + uscsa_ul + uscsa_dec + uscsa_comp;
  res_gscsa = gscsa_dl + gscsa_ul + gscsa_dec + gscsa_comp;


  subplot(3,3,6);
  plot(x, res_gasp, 'Color', 'b');
  hold on;
  plot(x, res_ass, 'Color', 'y');
  plot(x, res_scs, 'Color', 'r');
  plot(x, res_uscsa, 'Color', 'g');
  plot(x, res_gscsa, 'Color', 'magenta')
 # legend({'gasp','ass', 'scs'});
  xlabel ("Size [s]");
  ylabel ("Time of p");
  title("Total");
  hold off;
  #saveas(fig, res);

  subplot(3,3,7);
  p1 = plot(x, x, 'Color', 'b');
  hold on;
  p2 = plot(x, ass_comp, 'Color', 'y');
  p3 = plot(x, scs_comp, 'Color', 'r');
  p4 = plot(x, uscsa_comp, 'Color', 'g');
  p5 = plot(x, gscsa_comp, 'Color', 'magenta')
  legend({'gasp','ass', 'scsa', 'uscsa', 'gscsa'});
  set(p1, 'visible', 'off');
  set(p2, 'visible', 'off');
  set(p3, 'visible', 'off');
  set(p4, 'visible', 'off');
  set(p5, 'visible', 'off');
  set(gca, 'visible', 'off');
 # xlabel ("Size of p");
 # ylabel ("Time [s]");
 # title("Slaves");
  hold off;
endfunction

function [title, q, Field, m, n, p, nomer, coeff] = load_title(ordner)
  name = strcat(ordner, "/");
  name = strcat(name, "title.mat");
  load(name);
  title = FrameStack{1};
  q = FrameStack{2};
  Field = FrameStack{3};
  m = FrameStack{4};
  n = FrameStack{5};
  p = FrameStack{6};
  nomer = FrameStack{7};
  coeff = FrameStack{8};
endfunction

function res = rround(val)
 # down = floor(val);
    up = ceil(val);
 # if down == up
    res = up + 1;
 # else
    res = up;
 # end
endfunction

#ordner = argv(){1};
ordner = "test_enc";
[title, q, Field, m, n, p, nomer, coeff] = load_title(ordner)
#q = str2num(argv(){1});
#Field = str2num(argv(){2});
#start_matr_size = str2num(argv(){3});
#ordner = argv(){4};
#nomer = str2num(argv(){5});
#coeff = str2double(argv(){6});
title
work_with_experiment(q, Field, m, n, p, ordner, nomer, coeff, title)





