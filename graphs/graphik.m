
a = "Hi there!"


function [mu, sigma] = make_pdf(list, i, name)
  subplot(2,2,i);
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

function [fig, dlv, ulv, decv, compv] = make_pdfs(dl, ul, dec, comp)
  fig = figure('visible','off');
  [mu, sigma] = make_pdf(dl, 1, "Download");
  dlv = [mu, sigma];
  [mu, sigma] = make_pdf(ul, 2, "Upload");
  ulv = [mu, sigma];
  [mu, sigma] = make_pdf(dec, 3, "Decoding");
  decv = [mu, sigma];
  [mu, sigma] = make_pdf(comp, 4, "Slaves");
  compv = [mu, sigma];
endfunction

function [dl, ul, dec, comp] = load_data(name)
    load(name);
    dl = FrameStack{1};
    ul = FrameStack{2};
    dec = FrameStack{3};
    comp = FrameStack{4};
endfunction

function save_pdf(fig, name, q, Field, matr_size, ordner)
  q_str = num2str(q);
  q_str = strcat("/Q_", q_str);
  field_str = num2str(Field);
  field_str = strcat("Field_", field_str);
  ms_str = num2str(matr_size);
  ms_str = strcat("matrSize_", ms_str);
  res = strcat(q_str, field_str);
  res = strcat(res, ms_str);
  res = strcat(ordner, res);
  mkdir(strcat("./oct_results/", ordner));
  res = strcat("./oct_results/", res);
  mkdir(sprintf(res));
  res_file = strcat(res, name);
  saveas(fig, res_file);
endfunction

function [dlv, ulv, decv, compv] = work_with_scheme(name, q, Field, matr_size, ordner)
  [dl, ul, dec, comp] = load_data(strcat(ordner, name));
  [fig, dlv, ulv, decv, compv] = make_pdfs(dl, ul, dec, comp);
  save_pdf(fig, name, q, Field, matr_size, ordner);
endfunction


function [gaspv, assv, scsv] = work_with_schemes(q, Field, matr_size, ordner)
  q_str = strcat("_Q_", num2str(q));
  m_str = strcat("_m_", num2str(matr_size));
  n_str = strcat("_n_", num2str(matr_size));
  p_str = strcat("_p_", num2str(matr_size));
  name = strcat(q_str, m_str);
  name = strcat(name, n_str);
  name = strcat(name, p_str);
 # name = strcat(name, ".mat");
  gasp_addr = strcat("/gasp", name);
  ass_addr = strcat("/ass", name);
  scs_addr = strcat("/scs", name);
  [dlv, ulv, decv, compv] = work_with_scheme(gasp_addr, q, Field, matr_size, ordner);
  gaspv = [dlv, ulv, decv, compv];
  [dlv, ulv, decv, compv] = work_with_scheme(ass_addr, q, Field, matr_size, ordner);
  assv = [dlv, ulv, decv, compv];
  [dlv, ulv, decv, compv] = work_with_scheme(scs_addr, q, Field, matr_size, ordner);
  scsv = [dlv, ulv, decv, compv];
endfunction

#work_with_schemes(q, Field, matr_size);


function work_with_experiment(q, Field, start_matr_size, ordner, nomer, coeff, title)
  matr_size = start_matr_size;
  gasp_dl = [];
  gasp_ul = [];
  gasp_dec = [];
  gasp_comp = [];
  ass_dl = [];
  ass_ul = [];
  ass_dec = [];
  ass_comp = [];
  scs_dl = [];
  scs_ul = [];
  scs_dec = [];
  scs_comp = [];
  matr = [];

  for i = 1:nomer
    matr_size
    [gaspv, assv, scsv] = work_with_schemes(q, Field, matr_size, ordner);
    matr = [matr, matr_size];
    gasp_dl = [gasp_dl, gaspv(1)];
    gasp_ul = [gasp_ul, gaspv(3)];
    gasp_dec = [gasp_dec, gaspv(5)];
    gasp_comp = [gasp_comp, gaspv(7)];
    ass_dl = [ass_dl, assv(1)];
    ass_ul = [ass_ul, assv(3)];
    ass_dec = [ass_dec, assv(5)];
    ass_comp = [ass_comp, assv(7)];
    scs_dl = [scs_dl, scsv(1)];
    scs_ul = [scs_ul, scsv(3)];
    scs_dec = [scs_dec, scsv(5)];
    scs_comp = [scs_comp, scsv(7)];
    if title == 1
       matr_size = matr_size + coeff;
    else
       matr_size = cast(matr_size, "double");
       matr_size = rround(matr_size * coeff);
    end
  end

  x = matr;
  clear title
  fig = figure('visible','off');
  subplot(3,2,1);
  plot(x, gasp_dl, 'Color', 'b');
  hold on;
  plot(x, ass_dl, 'Color', 'y');
  plot(x, scs_dl, 'Color', 'r');
 # legend({'gasp','ass', 'scs'}, 'east');
  xlabel ("Matrix Size");
  ylabel ("Time");
  title("Download");
  hold off;
  subplot(3,2,2);
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
  y1 = plot(x, gasp_comp, 'Color', 'b');
  hold on;
  y2 = plot(x, ass_comp, 'Color', 'y');
  y3 = plot(x, scs_comp, 'Color', 'r');
  #legend({'gasp','ass', 'scs'});
  xlabel ("Matrix Size");
  ylabel ("Time");
  title("Slaves");
  hold off;
  subplot(3,1,3);
  axis off
  legend([y1, y2, y3], {'gasp','ass', 'scs'}, "Location", "best")



  res = strcat("./oct_results/", ordner);
  res = strcat(res, "/grafik.png");
  saveas(fig, res);
endfunction

function [title, q, Field, start_matr_size, nomer, coeff] = load_title(ordner)
  name = strcat(ordner, "/");
  name = strcat(name, "title.mat");
  load(name);
  title = FrameStack{1};
  q = FrameStack{2};
  Field = FrameStack{3};
  start_matr_size = FrameStack{4};
  nomer = FrameStack{5};
  coeff = FrameStack{6};
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

ordner = argv(){1};

[title, q, Field, start_matr_size, nomer, coeff] = load_title(ordner)
#q = str2num(argv(){1});
#Field = str2num(argv(){2});
#start_matr_size = str2num(argv(){3});
#ordner = argv(){4};
#nomer = str2num(argv(){5});
#coeff = str2double(argv(){6});
title
work_with_experiment(q, Field, start_matr_size, ordner, nomer, coeff, title)





