let SessionLoad = 1
let s:so_save = &so | let s:siso_save = &siso | set so=0 siso=0
let v:this_session=expand("<sfile>:p")
silent only
cd ~/python_projects/mixed_time_sparse_coding
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
set shortmess=aoO
badd +195 ~/python_projects/mixed_time_sparse_coding/ctsc.py
badd +43 ~/python_projects/mixed_time_sparse_coding/euler_maruyama.py
badd +31 ~/python_projects/mixed_time_sparse_coding/bars.py
badd +14 ~/python_projects/mixed_time_sparse_coding/logpx.py
badd +126 ~/python_projects/mixed_time_sparse_coding/vh_dkl.py
badd +394 ~/python_projects/mixed_time_sparse_coding/visualization.py
badd +82 ~/python_projects/mixed_time_sparse_coding/loaders.py
badd +185 ~/python_projects/mixed_time_sparse_coding/soln_analysis.py
badd +108 ~/python_projects/mixed_time_sparse_coding/bars_dkl.py
badd +2 term://.//11352:python\ bars.py
badd +2 term://.//11385:python\ bars.py
badd +729 term://.//11400:python\ bars.py
badd +703 term://.//11452:python\ bars.py
badd +1364 term://.//11507:python\ bars.py
badd +44 ~/python_projects/mixed_time_sparse_coding/no_norm_A.py
badd +2 term://.//12279:python\ no_norm_A.py
badd +8 ~/python_projects/mixed_time_sparse_coding/plt_env.py
badd +32 ~/python_projects/mixed_time_sparse_coding/lca_bars.py
badd +59 ~/python_projects/mixed_time_sparse_coding/lca.py
badd +9 ~/python_projects/mixed_time_sparse_coding/vh_patches.py
badd +29 ~/python_projects/mixed_time_sparse_coding/vh_no_norm.py
badd +0 term://.//18201:python\ no_norm_A.py
badd +21 ~/python_projects/mixed_time_sparse_coding/bars_oc_lsc_pi_sweep.py
badd +17 ~/python_projects/mixed_time_sparse_coding/bars_no_norm_corr.py
badd +2 ~/python_projects/mixed_time_sparse_coding/__doc__
badd +76 ~/python_projects/mixed_time_sparse_coding/bars_learn_pi.py
badd +0 ~/python_projects/mixed_time_sparse_coding/vh_learn_pi.py
argglobal
%argdel
$argadd ctsc.py
edit ~/python_projects/mixed_time_sparse_coding/vh_learn_pi.py
set splitbelow splitright
wincmd _ | wincmd |
vsplit
1wincmd h
wincmd w
wincmd t
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
exe 'vert 1resize ' . ((&columns * 102 + 102) / 204)
exe 'vert 2resize ' . ((&columns * 101 + 102) / 204)
argglobal
setlocal fdm=expr
setlocal fde=SimpylFold#FoldExpr(v:lnum)
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 22 - ((21 * winheight(0) + 26) / 53)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
22
normal! 011|
wincmd w
argglobal
if bufexists("~/python_projects/mixed_time_sparse_coding/vh_no_norm.py") | buffer ~/python_projects/mixed_time_sparse_coding/vh_no_norm.py | else | edit ~/python_projects/mixed_time_sparse_coding/vh_no_norm.py | endif
setlocal fdm=expr
setlocal fde=SimpylFold#FoldExpr(v:lnum)
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 9 - ((8 * winheight(0) + 26) / 53)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
9
normal! 01|
wincmd w
2wincmd w
exe 'vert 1resize ' . ((&columns * 102 + 102) / 204)
exe 'vert 2resize ' . ((&columns * 101 + 102) / 204)
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
