let SessionLoad = 1
let s:so_save = &so | let s:siso_save = &siso | set so=0 siso=0
let v:this_session=expand("<sfile>:p")
silent only
cd ~/python_projects/mixed_time_sparse_coding
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
set shortmess=aoO
badd +11 ~/python_projects/mixed_time_sparse_coding/vh_learn_pi.py
badd +19 ~/python_projects/mixed_time_sparse_coding/vh_no_norm.py
badd +433 ~/python_projects/mixed_time_sparse_coding/visualization.py
badd +109 ~/python_projects/mixed_time_sparse_coding/loaders.py
badd +202 ~/python_projects/mixed_time_sparse_coding/ctsc.py
badd +53 ~/python_projects/mixed_time_sparse_coding/vh_ease_pi.py
badd +30 ~/python_projects/mixed_time_sparse_coding/bars.py
badd +36 ~/python_projects/mixed_time_sparse_coding/no_norm_A.py
badd +23 ~/python_projects/mixed_time_sparse_coding/bars_sparsity.py
badd +73 ~/python_projects/mixed_time_sparse_coding/vh_patches.py
badd +1 term://python\ fugitive:///home/michael/python_projects/mixed_time_sparse_coding/.git//d132f86ba0fda3ed57a498cf34a5fd918347be2e/vh_learn_pi.py
badd +4 ~/python_projects/mixed_time_sparse_coding/scratch_pad.py
badd +301 ~/python_projects/mixed_time_sparse_coding/soln_analysis.py
badd +33 ~/python_projects/mixed_time_sparse_coding/vh_laplace.py
badd +19 ~/python_projects/mixed_time_sparse_coding/vh_low_sig.py
badd +43 ~/python_projects/mixed_time_sparse_coding/oc_vs_pi.py
badd +2 term://~/python_projects/mixed_time_sparse_coding//21529:python\ oc_vs_pi.py
badd +21 ~/python_projects/mixed_time_sparse_coding/learn_pi.py
badd +13 ~/python_projects/mixed_time_sparse_coding/bars_learn_pi.py
badd +10052 term://~/python_projects/mixed_time_sparse_coding//24741:python\ learn_pi.py
badd +0 term://~/python_projects/mixed_time_sparse_coding//24757:python\ learn_pi.py
badd +4 term://~/python_projects/mixed_time_sparse_coding//24773:python\ learn_pi.py
badd +261 term://~/python_projects/mixed_time_sparse_coding//24810:python\ learn_pi.py
badd +3 term://~/python_projects/mixed_time_sparse_coding//24856:python\ learn_pi.py
badd +7 term://~/python_projects/mixed_time_sparse_coding//24895:python\ learn_pi.py
badd +89 ~/python_projects/mixed_time_sparse_coding/euler_maruyama.py
badd +314 term://~/python_projects/mixed_time_sparse_coding//24961:python\ learn_pi.py
badd +0 term://~/python_projects/mixed_time_sparse_coding//25076:python\ learn_pi.py
badd +20 ~/python_projects/mixed_time_sparse_coding/dict_A_vs_sigma.py
badd +20 term://~/python_projects/mixed_time_sparse_coding//3420:python\ vh_low_sig.py
badd +177 term://~/python_projects/mixed_time_sparse_coding//4580:python\ vh_no_norm.py
badd +88 term://~/python_projects/mixed_time_sparse_coding//25170:python\ oc_vs_pi.py
badd +108 ~/python_projects/mixed_time_sparse_coding/bars_dkl.py
badd +14 term://~/python_projects/mixed_time_sparse_coding//26265:python\ bars_sparsity.py
badd +2 ~/python_projects/mixed_time_sparse_coding/oc_vs_pi.pdf
argglobal
%argdel
$argadd vh_learn_pi.py
edit ~/python_projects/mixed_time_sparse_coding/bars.py
set splitbelow splitright
wincmd _ | wincmd |
vsplit
wincmd _ | wincmd |
vsplit
2wincmd h
wincmd w
wincmd w
wincmd t
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
exe 'vert 1resize ' . ((&columns * 67 + 40) / 80)
exe 'vert 2resize ' . ((&columns * 10 + 40) / 80)
exe 'vert 3resize ' . ((&columns * 1 + 40) / 80)
argglobal
setlocal fdm=expr
setlocal fde=SimpylFold#FoldExpr(v:lnum)
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
1
normal! zo
let s:l = 32 - ((8 * winheight(0) + 11) / 22)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
32
normal! 07|
wincmd w
argglobal
if bufexists("~/python_projects/mixed_time_sparse_coding/oc_vs_pi.py") | buffer ~/python_projects/mixed_time_sparse_coding/oc_vs_pi.py | else | edit ~/python_projects/mixed_time_sparse_coding/oc_vs_pi.py | endif
if &buftype ==# 'terminal'
  silent file ~/python_projects/mixed_time_sparse_coding/oc_vs_pi.py
endif
setlocal fdm=expr
setlocal fde=SimpylFold#FoldExpr(v:lnum)
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
1
normal! zo
let s:l = 67 - ((1 * winheight(0) + 11) / 22)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
67
normal! 035|
wincmd w
argglobal
if bufexists("~/python_projects/mixed_time_sparse_coding/ctsc.py") | buffer ~/python_projects/mixed_time_sparse_coding/ctsc.py | else | edit ~/python_projects/mixed_time_sparse_coding/ctsc.py | endif
if &buftype ==# 'terminal'
  silent file ~/python_projects/mixed_time_sparse_coding/ctsc.py
endif
setlocal fdm=expr
setlocal fde=SimpylFold#FoldExpr(v:lnum)
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
21
normal! zo
35
normal! zo
160
normal! zo
202
normal! zo
266
normal! zo
438
normal! zo
439
normal! zo
let s:l = 48 - ((12 * winheight(0) + 11) / 22)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
48
normal! 034|
wincmd w
2wincmd w
exe 'vert 1resize ' . ((&columns * 67 + 40) / 80)
exe 'vert 2resize ' . ((&columns * 10 + 40) / 80)
exe 'vert 3resize ' . ((&columns * 1 + 40) / 80)
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
