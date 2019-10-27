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

function [dlv, ulv, decv, compv] = work_with_scheme(name, q, Field, m, n, p, ordner)
  [dl, ul, dec, comp] = load_data(strcat(ordner, name));
  [fig, dlv, ulv, decv, compv] = make_pdfs(dl, ul, dec, comp);
  #save_pdf(fig, name, q, Field, m, n, p, ordner);
endfunction


function [scsv, uscsav] = work_with_schemes(q, Field, m, n, p, ordner)
  q_str = strcat("_Q_", num2str(q));
  m_str = strcat("_m_", num2str(m));
  n_str = strcat("_n_", num2str(n));
  p_str = strcat("_p_", num2str(p));
  name = strcat(q_str, m_str);
  name = strcat(name, n_str);
  name = strcat(name, p_str);
 # name = strcat(name, ".mat");
  scs_addr = strcat("/scsa", name);
  uscsa_addr = strcat("/scsa_ruck", name);  
  
  [dlv, ulv, decv, compv] = work_with_scheme(scs_addr, q, Field, m, n, p, ordner);
  scsv = [dlv, ulv, decv, compv];
  
  [dlv, ulv, decv, compv] = work_with_scheme(uscsa_addr, q, Field, m, n, p, ordner);
  uscsav = [dlv, ulv, decv, compv];
  
  scsv
  uscsav
  
  
endfunction

#work_with_schemes(q, Field, matr_size);


function work_with_experiment(q, Field, m, n, p, ordner, nomer, coeff, title)
  scs_dl = [];
  scs_ul = [];
  scs_dec = [];
  scs_comp = [];
  uscsa_dl = [];
  uscsa_ul = [];
  uscsa_dec = [];
  uscsa_comp = [];
  matr = [];

  for i = 1:nomer
    [scsv, uscsav] = work_with_schemes(q, Field, m, n, p, ordner);
    matr = [matr, p];
    m
    n
    p
    scs_dl = [scs_dl, scsv(1)];
    scs_ul = [scs_ul, scsv(3)];
    scs_dec = [scs_dec, scsv(5)];
    scs_comp = [scs_comp, scsv(7)];
    uscsa_dl = [uscsa_dl, uscsav(1)];
    uscsa_ul = [uscsa_ul, uscsav(3)];
    uscsa_dec = [uscsa_dec, uscsav(5)];
    uscsa_comp = [uscsa_comp, uscsav(7)];
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
  subplot(3,2,1)
   plot(x, scs_dl, 'Color', 'r')
  hold on;
  plot(x, uscsa_dl, 'Color', 'g')
  xlabel ("Matrix Size");
  ylabel ("Time");
  title("Download");
  
  hold off;
  subplot(3,2,2);
  plot(x, scs_ul, 'Color', 'r');
  hold on;
  plot(x, uscsa_ul, 'Color', 'g');
 # legend({'gasp','ass', 'scs'});
  xlabel ("Matrix Size");
  ylabel ("Time");
  title("Upload");
  hold off;
  subplot(3,2,3);
  

  plot(x, scs_dec, 'Color', 'r');
  hold on;
  plot(x, uscsa_dec, 'Color', 'g');
 # legend({'gasp','ass', 'scs'});
  xlabel ("Matrix Size");
  ylabel ("Time");
  title("Decoding");
  hold off;
  subplot(3,2,4);
  
  
  plot(x, scs_comp, 'Color', 'r');
  hold on;
  plot(x, uscsa_comp, 'Color', 'g');
  #legend({'gasp','ass', 'scs'});
  xlabel ("Matrix Size");
  ylabel ("Time");
  title("Slaves");
  hold off;
  subplot(3,2,5);
  

  res_scs = scs_dl + scs_ul + scs_dec + scs_comp;
  res_uscsa = uscsa_dl + uscsa_ul + uscsa_dec + uscsa_comp;
  
  plot(x, res_scs, 'Color', 'r');
  hold on;
  plot(x, res_uscsa, 'Color', 'g');
 # legend({'gasp','ass', 'scs'});
  xlabel ("Matrix Size");
  ylabel ("Time");
  title("Total");
  hold off;
  
  scs_dec
  scs_comp
  scs_ul
  scs_dl
  res_scs
  
  uscsa_dec
  uscsa_comp
  uscsa_ul
  uscsa_dl
  res_uscsa
  
  #saveas(fig, res);
  
  #subplot(3,2,6);
  #p3 = plot(x, scs_comp, 'Color', 'r');
  #hold on;
  #p4 = plot(x, uscsa_comp, 'Color', 'g');
  #legend({'scsa', 'scsa bad'});
  
  #set(p3, 'visible', 'off');
  #set(p4, 'visible', 'off');
 
 # xlabel ("Matrix Size");
 # ylabel ("Time");
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
ordner = "332"
[title, q, Field, m, n, p, nomer, coeff] = load_title(ordner)
#q = str2num(argv(){1});
#Field = str2num(argv(){2});
#start_matr_size = str2num(argv(){3});
#ordner = argv(){4};
#nomer = str2num(argv(){5});
#coeff = str2double(argv(){6});
title
work_with_experiment(q, Field, m, n, p, ordner, nomer, coeff, title)



