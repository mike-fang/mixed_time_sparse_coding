let SessionLoad = 1
let s:so_save = &so | let s:siso_save = &siso | set so=0 siso=0
let v:this_session=expand("<sfile>:p")
silent only
cd ~/python_projects/mixed_time_sparse_coding
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
set shortmess=aoO
badd +41 ~/python_projects/mixed_time_sparse_coding/bars_lca_u0_sparsity.py
badd +20 ~/python_projects/mixed_time_sparse_coding/bars_dsc_l1_sparsity.py
badd +66 ~/python_projects/mixed_time_sparse_coding/bars_dsc_l1_sweep.py
badd +14 ~/python_projects/mixed_time_sparse_coding/bars_sparsity.py
badd +2 ~/python_projects/mixed_time_sparse_coding/results/bars_dsc/l1_0p10/params.yaml
badd +85 ~/python_projects/mixed_time_sparse_coding/ctsc.py
badd +60 ~/python_projects/mixed_time_sparse_coding/soln_analysis.py
badd +14 ~/python_projects/mixed_time_sparse_coding/lca_sweep_u0.py
badd +66 ~/python_projects/mixed_time_sparse_coding/bars_lsc_pi_sweep.py
badd +37 ~/python_projects/mixed_time_sparse_coding/bars.py
badd +0 ~/python_projects/mixed_time_sparse_coding/bars_lsc_pi_sparsity.py
badd +34 term://.//11576:python\ bars_lsc_pi_sparsity.py
badd +0 ~/python_projects/mixed_time_sparse_coding/results/bars_lsc/PI_0p10_pi_0p45/params.yaml
argglobal
%argdel
$argadd bars_lca_u0_sparsity.py
edit ~/python_projects/mixed_time_sparse_coding/bars_lsc_pi_sparsity.py
set splitbelow splitright
wincmd t
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
argglobal
setlocal fdm=expr
setlocal fde=SimpylFold#FoldExpr(v:lnum)
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 14 - ((13 * winheight(0) + 26) / 53)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
14
normal! 07|
tabnext 1
if exists('s:wipebuf') && getbufvar(s:wipebuf, '&buftype') isnot# 'terminal'
  silent exe 'bwipe ' . s:wipebuf
endif
unlet! s:wipebuf
set winheight=1 winwidth=20 winminheight=1 winminwidth=1 shortmess=filnxtToOF
let s:sx = expand("<sfile>:p:r")."x.vim"
if file_readable(s:sx)
  exe "source " . fnameescape(s:sx)
endif
let &so = s:so_save | let &siso = s:siso_save
doautoall SessionLoadPost
unlet SessionLoad
" vim: set ft=vim :
